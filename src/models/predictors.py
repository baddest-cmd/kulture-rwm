"""
Predictor heads and cultural loss formulations for tidal-kulture-rwm.

Implements engagement prediction, Context Tax estimation, regular simplex
prototype spanning, and the scalable O(M log M) differentiable smooth Gini loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec_backbone import project_to_hypersphere


class EngagementPredictor(nn.Module):
    """
    Predicts per-track stream probability given latent user state and slate embeddings.
    """

    def __init__(
        self,
        recurrent_dim: int = 128,
        stochastic_dim: int = 64,
        track_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = recurrent_dim + stochastic_dim + track_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        track_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: Recurrent deterministic state, shape: (Batch, H).
            z: Stochastic state, shape: (Batch, Z).
            track_embeddings: Slate track embeddings, shape: (Batch, SlateSize, TrackDim).

        Returns:
            Stream probabilities, shape: (Batch, SlateSize).
        """
        batch_size, slate_size, track_dim = track_embeddings.shape
        user_state = torch.cat([h, z], dim=-1)  # Shape: (Batch, H + Z)
        user_state_expanded = user_state.unsqueeze(1).expand(-1, slate_size, -1)
        pair_features = torch.cat([user_state_expanded, track_embeddings], dim=-1)
        probs = self.mlp(pair_features).squeeze(-1)
        return probs


class ContextTaxPredictor(nn.Module):
    """
    Predicts the cultural Context Tax tau_c = S_global - D_local in [-1, 1].
    """

    def __init__(
        self,
        recurrent_dim: int = 128,
        stochastic_dim: int = 64,
        action_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = recurrent_dim + stochastic_dim + action_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Bounded strictly in [-1, 1]
        )

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: Recurrent state, shape: (Batch, H).
            z: Stochastic state, shape: (Batch, Z).
            action_repr: Slate action representation, shape: (Batch, ActionDim).

        Returns:
            Predicted tau_c, shape: (Batch, 1).
        """
        features = torch.cat([h, z, action_repr], dim=-1)
        return self.mlp(features)


class PrototypeLossOutput(torch.Tensor):
    """Tensor subclass enabling both scalar arithmetic and 3-tuple unpacking."""

    def __new__(cls, total: torch.Tensor, attraction: torch.Tensor, separation: torch.Tensor):
        res = total.as_subclass(cls)
        res.attraction = attraction
        res.separation = separation
        return res

    def __iter__(self):
        yield self
        yield self.attraction
        yield self.separation


class PrototypeSimplexLoss(nn.Module):
    """
    Enforces regular simplex spanning and subgenre clustering on S^(D-1).

    Equations:
        theta_target = arccos(-1 / (K - 1))
        L_proto = 1/M * sum(1 - max_k <v_i, p_k>)
                + lambda_sep / (K * (K - 1)) * sum_{j != k} max(0, P P^T - cos(theta_target))
    """

    def __init__(
        self,
        num_prototypes: int = 4,
        dim: int = 64,
        lambda_sep: float = 0.1,
        eps: float = 1e-7,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim is not None:
            dim = embed_dim
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.lambda_sep = lambda_sep
        self.eps = eps

        # Prototypes matrix P on S^(D-1)
        init_prototypes = torch.randn(num_prototypes, dim)
        init_prototypes = project_to_hypersphere(init_prototypes, eps=eps)
        self.prototypes = nn.Parameter(init_prototypes)

        # Theoretical optimal simplex angle: cos(theta_target) = -1 / (K - 1)
        self.cos_theta_target = -1.0 / (num_prototypes - 1)

    def get_normalised_prototypes(self) -> torch.Tensor:
        """Returns unit-normalised subgenre prototypes on S^(D-1)."""
        return project_to_hypersphere(self.prototypes, eps=self.eps)

    def forward(self, track_embeddings: torch.Tensor) -> PrototypeLossOutput:
        """
        Computes prototype attraction and mutual simplex separation penalty.

        Args:
            track_embeddings: Candidate track embeddings, shape: (M, D).

        Returns:
            PrototypeLossOutput: (total_loss, attraction_loss, separation_loss)
        """
        k = self.num_prototypes
        norm_prototypes = self.get_normalised_prototypes()  # Shape: (K, D)

        # 1. Track-prototype attraction: 1/M sum_i (1 - max_k <v_i, p_k>)
        # Dot product matrix: (M, K)
        cosine_sims = torch.matmul(track_embeddings, norm_prototypes.t())
        max_sims, _ = torch.max(cosine_sims, dim=-1)
        attraction_loss = torch.mean(1.0 - max_sims)

        # 2. Simplex mutual repulsion penalty: sum_{j != k} max(0, P P^T - cos(theta_target))
        # Gram matrix P P^T: (K, K)
        gram = torch.matmul(norm_prototypes, norm_prototypes.t())

        # Mask out diagonal (self-similarity p_k . p_k = 1.0)
        off_diag_mask = ~torch.eye(k, dtype=torch.bool, device=track_embeddings.device)
        off_diag_sims = gram[off_diag_mask]

        # Penalise correlation exceeding the regular simplex target angle
        excess_correlation = F.relu(off_diag_sims - self.cos_theta_target)
        separation_loss = self.lambda_sep * torch.sum(excess_correlation) / (k * (k - 1))

        total_loss = attraction_loss + separation_loss
        return PrototypeLossOutput(total_loss, attraction_loss, separation_loss)


def differentiable_smooth_gini(
    exposures: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes catalog exposure Gini coefficient with scalable O(M log M) complexity.

    Formula:
        e_sorted = sort(e)
        L_gini = (2 * sum_{i=1}^M i * e_sorted,i) / (M * sum_{i=1}^M e_i + eps) - (M + 1) / M

    Ensures zero allocation of O(M^2) dense pairwise difference matrices.
    """
    m = exposures.shape[-1]
    if m <= 1:
        return torch.tensor(0.0, device=exposures.device, dtype=exposures.dtype)

    # Ascending sort: O(M log M)
    sorted_exposures, _ = torch.sort(exposures, dim=-1)

    # 1-indexed rank weights: [1, 2, ..., M]
    ranks = torch.arange(1, m + 1, device=exposures.device, dtype=exposures.dtype)

    # Vectorised weighted rank sum
    weighted_sum = torch.sum(ranks * sorted_exposures, dim=-1)
    total_exposure = torch.sum(exposures, dim=-1)

    numerator = 2.0 * weighted_sum
    denominator = float(m) * (total_exposure + eps)

    gini = (numerator / denominator) - (float(m + 1) / float(m))
    return torch.clamp(gini, 0.0, 1.0)


class SmoothGiniLoss(nn.Module):
    """
    Differentiable exposure equality loss wrapper for training loops.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, predicted_exposures: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_exposures: Tensor of item exposure frequencies, shape: (M,) or (Batch, M).

        Returns:
            Scalar Gini coefficient penalty in [0, 1].
        """
        return differentiable_smooth_gini(predicted_exposures, eps=self.eps).mean()
