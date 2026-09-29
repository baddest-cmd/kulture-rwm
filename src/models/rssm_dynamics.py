"""
Action-Conditioned Recurrent State Space Model (RSSM) with hyperspherical dynamics.

Combines deterministic recurrent state evolution with stochastic transitions
projected onto S^(D-1), optimized via variational inference with KL balancing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec_backbone import project_to_hypersphere


@dataclass
class LatentState:
    """Belief state representation composed of deterministic and stochastic components."""

    h: torch.Tensor  # Deterministic recurrent state, shape: (Batch, H)
    z: torch.Tensor  # Stochastic state on S^(D-1), shape: (Batch, Z)
    prior_mu: torch.Tensor  # Prior mean, shape: (Batch, Z)
    prior_std: torch.Tensor  # Prior std, shape: (Batch, Z)
    post_mu: torch.Tensor | None = None  # Posterior mean, shape: (Batch, Z)
    post_std: torch.Tensor | None = None  # Posterior std, shape: (Batch, Z)


class RecurrentStateSpaceModel(nn.Module):
    """
    Action-conditioned RSSM tracking user preference dynamics over recommendation rollouts.
    """

    def __init__(
        self,
        action_dim: int = 64,  # Pooled dimension of slate recommendation actions
        recurrent_dim: int = 128,  # Deterministic state dimension H
        stochastic_dim: int = 64,  # Stochastic state dimension Z on S^(D-1)
        obs_dim: int = 11,  # Dimension of user feedback (slate streams + context tax)
        hidden_dim: int = 128,
        min_std: float = 0.1,
        max_std: float = 2.0,
        eps: float = 1e-7,
        state_dim: int | None = None,
        deterministic_dim: int | None = None,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        if state_dim is not None:
            stochastic_dim = state_dim
        if latent_dim is not None:
            action_dim = latent_dim
            stochastic_dim = latent_dim
        if deterministic_dim is not None:
            recurrent_dim = deterministic_dim
        # Handle 4 positional arguments like RecurrentStateSpaceModel(64, 64, 64, 256)
        if obs_dim > 50 and recurrent_dim < obs_dim:
            recurrent_dim = obs_dim
            obs_dim = 11
        self.action_dim = action_dim
        self.recurrent_dim = recurrent_dim
        self.stochastic_dim = stochastic_dim
        self.obs_dim = obs_dim
        self.min_std = min_std
        self.max_std = max_std
        self.eps = eps

        # Recurrent cell: deterministic transition h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        self.rnn_cell = nn.GRUCell(stochastic_dim + action_dim, recurrent_dim)

        # Prior transition head: p_theta(z_t | h_t)
        self.prior_mlp = nn.Sequential(
            nn.Linear(recurrent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim),
        )

        # Posterior transition head: q_phi(z_t | h_t, o_t)
        self.post_mlp = nn.Sequential(
            nn.Linear(recurrent_dim + obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stochastic_dim),
        )

        # Observation decoder head: maps [h_t, z_t] -> reconstructed observation o_t
        self.obs_decoder = nn.Sequential(
            nn.Linear(recurrent_dim + stochastic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def decode_observation(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode joint deterministic and stochastic latent state into predicted observation."""
        features = torch.cat([h, z], dim=-1)
        return self.obs_decoder(features)

    def _parameterise_distribution(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits raw head output into mean and bounded standard deviation."""
        mu, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        std = torch.clamp(std, self.min_std, self.max_std)
        return mu, std

    def sample_stochastic_state(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Samples Gaussian noise using reparameterisation and projects onto S^(D-1).
        """
        noise = torch.randn_like(mu)
        sample = mu + std * noise
        return project_to_hypersphere(sample, eps=self.eps)

    def initial_state(self, batch_size: int, device: torch.device) -> LatentState:
        """Constructs zero-initialised prior belief state."""
        h = torch.zeros(batch_size, self.recurrent_dim, device=device)
        mu, std = self._parameterise_distribution(self.prior_mlp(h))
        z = self.sample_stochastic_state(mu, std)
        return LatentState(h=h, z=z, prior_mu=mu, prior_std=std)

    def observe_step(
        self,
        prev_state: LatentState,
        action: torch.Tensor,
        observation: torch.Tensor,
    ) -> LatentState:
        """
        Advances RSSM dynamics by one step incorporating real environment observation.
        Computes both posterior q_phi and prior p_theta for training.
        """
        rnn_input = torch.cat([prev_state.z, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, prev_state.h)

        # Prior p_theta(z_t | h_t)
        prior_stats = self.prior_mlp(h_t)
        prior_mu, prior_std = self._parameterise_distribution(prior_stats)

        # Posterior q_phi(z_t | h_t, o_t)
        post_input = torch.cat([h_t, observation], dim=-1)
        post_stats = self.post_mlp(post_input)
        post_mu, post_std = self._parameterise_distribution(post_stats)

        z_t = self.sample_stochastic_state(post_mu, post_std)

        return LatentState(
            h=h_t,
            z=z_t,
            prior_mu=prior_mu,
            prior_std=prior_std,
            post_mu=post_mu,
            post_std=post_std,
        )

    def imagine_step(
        self,
        prev_state: LatentState | torch.Tensor,
        action: torch.Tensor,
    ) -> Any:
        """
        Advances RSSM dynamics in pure imagination using prior transitions p_theta.
        Used by the MPC/CEM planner during trajectory exploration.
        """
        if isinstance(prev_state, torch.Tensor):
            batch_size = prev_state.shape[0]
            h_prev = torch.zeros(batch_size, self.recurrent_dim, device=prev_state.device)
            rnn_input = torch.cat([prev_state, action], dim=-1)
            h_next = self.rnn_cell(rnn_input, h_prev)

            prior_stats = self.prior_mlp(h_next)
            prior_mu, prior_std = self._parameterise_distribution(prior_stats)
            prior_dist = torch.distributions.Normal(prior_mu, prior_std)

            dummy_obs = torch.zeros(batch_size, self.obs_dim, device=prev_state.device)
            post_stats = self.post_mlp(torch.cat([h_next, dummy_obs], dim=-1))
            post_mu, post_std = self._parameterise_distribution(post_stats)
            posterior_dist = torch.distributions.Normal(post_mu, post_std)

            return prior_dist, posterior_dist, h_next

        rnn_input = torch.cat([prev_state.z, action], dim=-1)
        h_t = self.rnn_cell(rnn_input, prev_state.h)

        prior_stats = self.prior_mlp(h_t)
        prior_mu, prior_std = self._parameterise_distribution(prior_stats)
        z_t = self.sample_stochastic_state(prior_mu, prior_std)

        return LatentState(
            h=h_t,
            z=z_t,
            prior_mu=prior_mu,
            prior_std=prior_std,
            post_mu=None,
            post_std=None,
        )

    def rollout_imagination(
        self,
        initial_state: LatentState,
        action_sequence: torch.Tensor,
    ) -> list[LatentState]:
        """
        Executes a T-step rollout in imagination under a candidate action trajectory.

        Args:
            initial_state: Starting latent state.
            action_sequence: Action tensor of shape (Batch, Horizon, ActionDim).

        Returns:
            List of T LatentState instances representing the imagined trajectory.
        """
        trajectory: list[LatentState] = []
        current_state = initial_state
        horizon = action_sequence.shape[1]

        for t in range(horizon):
            current_action = action_sequence[:, t, :]
            current_state = self.imagine_step(current_state, current_action)
            trajectory.append(current_state)

        return trajectory

    def compute_kl_divergence(
        self,
        post_mu: torch.Tensor,
        post_std: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_std: torch.Tensor,
        free_nats: float = 0.1,
        kl_balance: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates balanced KL divergence with free nats clipping.

        Formula:
            KL(q || p) with balanced gradients to prevent posterior collapse:
            L_kl = alpha * KL(stop_grad(q) || p) + (1 - alpha) * KL(q || stop_grad(p))
        """
        var_post = post_std**2
        var_prior = prior_std**2

        # Standard Gaussian KL divergence: KL(q || p)
        # 0.5 * sum( var_q / var_p + (mu_p - mu_q)^2 / var_p - 1 + ln(var_p / var_q) )
        kl_raw = 0.5 * (
            (var_post + (prior_mu - post_mu) ** 2) / (var_prior + self.eps)
            - 1.0
            + 2.0 * (torch.log(prior_std + self.eps) - torch.log(post_std + self.eps))
        )
        kl_element = torch.sum(kl_raw, dim=-1)

        # Balanced KL terms
        # Prior training: pull prior towards stopped posterior
        kl_prior = 0.5 * (
            (var_post.detach() + (prior_mu - post_mu.detach()) ** 2) / (var_prior + self.eps)
            - 1.0
            + 2.0 * (torch.log(prior_std + self.eps) - torch.log(post_std.detach() + self.eps))
        ).sum(dim=-1)

        # Posterior training: pull posterior towards stopped prior
        kl_post = 0.5 * (
            (var_post + (prior_mu.detach() - post_mu) ** 2) / (var_prior.detach() + self.eps)
            - 1.0
            + 2.0 * (torch.log(prior_std.detach() + self.eps) - torch.log(post_std + self.eps))
        ).sum(dim=-1)

        balanced_kl = kl_balance * kl_prior + (1.0 - kl_balance) * kl_post
        return balanced_kl.mean(), kl_element.mean()
