# src/policy/mpc_planner.py
"""Cross‑Entropy Method (CEM) Model‑Predictive Control planner.

The planner samples a population of action sequences, rolls each through the
trained Recurrent State‑Space Model (RSSM) in imagination, evaluates a scalar
reward that combines predicted engagement, a Context‑Tax penalty and a smooth
Gini exposure penalty, then refits a Gaussian distribution over the elite
samples.  The first action of the best trajectory is returned.

All operations are fully vectorised on the specified ``device`` (CPU by default).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

# Local model imports
from src.models.rssm_dynamics import RecurrentStateSpaceModel, LatentState
from src.models.predictors import (
    EngagementPredictor,
    ContextTaxPredictor,
    SmoothGiniLoss,
)

class CEMPMPPlanner:
    """Cross‑Entropy Method planner for the RSSM world model.

    Parameters
    ----------
    rssm: RecurrentStateSpaceModel
        Trained world‑model used for imagination rollouts.
    engagement_head: EngagementPredictor
        Predicts per‑track engagement probabilities.
    context_tax_head: ContextTaxPredictor
        Predicts the Context Tax (τ_c) for a given latent state and action.
    gini_loss: SmoothGiniLoss
        Differentiable smooth‑Gini exposure loss.
    horizon: int, default 10
        Planning horizon (T).
    pop_size: int, default 256
        Number of sampled action sequences per CEM iteration.
    elite_frac: float, default 0.1
        Fraction of top‑scoring samples used as elite set.
    cem_iters: int, default 3
        Number of CEM refinement iterations.
    action_dim: int, default 64
        Dimensionality of the action representation.
    tax_threshold: float, default 0.0
        Threshold above which the planner penalises high Context Tax.
    device: torch.device, default ``cpu``
        Device on which all tensors are allocated.
    """

    def __init__(
        self,
        rssm: RecurrentStateSpaceModel,
        engagement_head: EngagementPredictor,
        context_tax_head: ContextTaxPredictor,
        gini_loss: SmoothGiniLoss,
        *,
        horizon: int = 10,
        pop_size: int = 256,
        elite_frac: float = 0.1,
        cem_iters: int = 3,
        action_dim: int = 64,
        slate_size: int | None = None,
        item_dim: int | None = None,
        tax_threshold: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.rssm = rssm
        self.engagement_head = engagement_head
        self.context_tax_head = context_tax_head
        self.gini_loss = gini_loss
        self.horizon = horizon
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.cem_iters = cem_iters
        self.slate_size = slate_size
        self.item_dim = item_dim
        if slate_size is not None and item_dim is not None:
            self.action_dim = slate_size * item_dim
        else:
            self.action_dim = action_dim
        self.tax_threshold = tax_threshold
        self.device = device
        # Initialise Gaussian distribution (mean=0, std=1) on the chosen device
        self.mean = torch.zeros(pop_size, horizon, self.action_dim, device=self.device)
        self.std = torch.ones(pop_size, horizon, self.action_dim, device=self.device)
        # If planner action_dim differs from RSSM's expected action_dim, create a linear projection
        if self.action_dim != self.rssm.action_dim:
            self.action_proj = torch.nn.Linear(self.action_dim, self.rssm.action_dim).to(self.device)
        else:
            self.action_proj = None

    def _sample_actions(self) -> torch.Tensor:
        """Sample a batch of action sequences from the current Gaussian.

        Returns
        -------
        actions: torch.Tensor of shape ``(pop_size, horizon, action_dim)``.
        """
        return self.mean + self.std * torch.randn_like(self.mean, device=self.device)

    def _evaluate(self, actions: torch.Tensor, init_state: LatentState) -> torch.Tensor:
        """Rollout ``actions`` through the RSSM and compute a scalar reward.

        Parameters
        ----------
        actions: torch.Tensor of shape ``(pop_size, horizon, action_dim)``.
        init_state: LatentState – the current belief state.

        Returns
        -------
        rewards: torch.Tensor of shape ``(pop_size,)`` – higher is better.
        """
        # Expand the initial latent state across the population dimension
        def _expand_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            if t.dim() == 1:
                return t.unsqueeze(0).repeat(self.pop_size, 1)
            if t.shape[0] == 1:
                return t.repeat(self.pop_size, 1)
            # If t has batch > 1, slice the first state and repeat across pop_size
            return t[0:1].repeat(self.pop_size, 1)

        state = LatentState(
            h=_expand_tensor(init_state.h),
            z=_expand_tensor(init_state.z),
            prior_mu=_expand_tensor(init_state.prior_mu),
            prior_std=_expand_tensor(init_state.prior_std),
            post_mu=_expand_tensor(init_state.post_mu),
            post_std=_expand_tensor(init_state.post_std),
        )

        cum_retention = torch.zeros(self.pop_size, device=self.device)
        cum_tax = torch.zeros(self.pop_size, device=self.device)
        exposures = []  # collect proxy exposure tensors for Gini loss

        for t in range(self.horizon):
            a_t = actions[:, t, :]
            # Project actions to RSSM's action dimension if needed
            proj_a_t = self.action_proj(a_t) if self.action_proj is not None else a_t
            state = self.rssm.imagine_step(state, proj_a_t)
            # Dummy track embeddings for engagement prediction (batch, K, D)
            dummy_tracks = torch.randn(self.pop_size, 10, self.rssm.action_dim, device=self.device)
            engagement = self.engagement_head(state.h, state.z, dummy_tracks)
            cum_retention += engagement.mean(dim=-1)  # scalar per trajectory
            tax = self.context_tax_head(state.h, state.z, proj_a_t).squeeze(-1)
            cum_tax += tax
            exposures.append(dummy_tracks.mean(dim=1).norm(dim=-1))

        exposures_tensor = torch.stack(exposures, dim=1)  # (pop, horizon)
        gini = self.gini_loss(exposures_tensor)
        # Reward = high retention – penalty on tax (absolute) – gini
        reward = cum_retention - torch.abs(cum_tax) - gini
        return reward

    def plan(
        self,
        init_state: LatentState,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Return the optimal action or trajectory of action slates.

        Parameters
        ----------
        init_state: LatentState – current RSSM belief.
        return_trajectory: bool, default False
            If True, returns the full planned trajectory matching
            ``(horizon, slate_size, item_dim)`` if slate_size and item_dim are set,
            or ``(horizon, action_dim)``.
            If False, returns the first action of shape ``(action_dim,)``.

        Returns
        -------
        torch.Tensor
            Optimal action of shape ``(action_dim,)`` or trajectory of shape
            ``(horizon, slate_size, item_dim)`` / ``(horizon, action_dim)``.
        """
        for _ in range(self.cem_iters):
            actions = self._sample_actions()
            rewards = self._evaluate(actions, init_state)
            k = max(1, int(self.elite_frac * self.pop_size))
            elite_idx = rewards.topk(k, dim=0).indices
            elite_actions = actions[elite_idx]
            # Re‑fit Gaussian over elite actions
            self.mean = elite_actions.mean(dim=0, keepdim=True).repeat(self.pop_size, 1, 1)
            self.std = elite_actions.std(dim=0, unbiased=False, keepdim=True).repeat(self.pop_size, 1, 1)
            self.std = torch.clamp(self.std, min=1e-5)
        # Best sequence from the mean trajectory
        best_seq = self.mean[0]  # (horizon, action_dim)
        if return_trajectory:
            if self.slate_size is not None and self.item_dim is not None:
                return best_seq.view(self.horizon, self.slate_size, self.item_dim)
            return best_seq
        return best_seq[0]

# End of mpc_planner.py
