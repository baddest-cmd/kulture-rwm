"""
Causal Alignment Framing (CAFL) utilities.

The framing layer adds a regularisation term that encourages the model
to predict a low cultural Context Tax (tau_c) for the actions it selects.
This mirrors the "CAFL" framing described in the design specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FramingConfig:
    """Configuration for the causal alignment framing layer.

    Attributes
    ----------
    weight: float
        Scaling factor for the framing loss term.
    target_tax: float, default 0.0
        Desired Context Tax (tau_c) value. Typically zero indicates perfect
        alignment between global satisfaction and local discovery.
    tax_bound: float, default 1.0
        Maximum allowable magnitude for Context Tax (tau_c).
    """

    weight: float = 0.8
    target_tax: float = 0.0
    tax_bound: float = 1.0


class CausalAlignmentFramingLayer(nn.Module):
    """Computes a framing regularisation loss on predicted Context Tax.

    The layer is deliberately lightweight – it uses a pre‑trained
    ``ContextTaxPredictor`` to estimate ``tau_c`` from the latent RSSM
    state and the current action representation, then penalises deviation
    from the target tax.
    """

    def __init__(
        self,
        context_tax_predictor: nn.Module,
        config: FramingConfig | None = None,
    ) -> None:
        super().__init__()
        self.predictor = context_tax_predictor
        self.config = config if config is not None else FramingConfig()

    def bound_tax(self, tax: torch.Tensor, bound: float | None = None) -> torch.Tensor:
        """Apply monotonic bounding to Context Tax tau_c.

        Parameters
        ----------
        tax : torch.Tensor
            Raw predicted context tax tensor.
        bound : float | None
            Upper bound limit. If None, uses config.tax_bound.

        Returns
        -------
        torch.Tensor
            Monotonically bounded Context Tax within [-bound, bound].
        """
        b = bound if bound is not None else self.config.tax_bound
        return torch.clamp(tax, min=-b, max=b)

    def forward(self, h: torch.Tensor, z: torch.Tensor, action_repr: torch.Tensor) -> torch.Tensor:
        """Return the framing loss for a single time‑step.

        Parameters
        ----------
        h : torch.Tensor
            Deterministic recurrent state of shape ``(B, H)``.
        z : torch.Tensor
            Stochastic hyperspherical state of shape ``(B, Z)``.
        action_repr : torch.Tensor
            Action embedding (e.g. pooled slate representation) of shape ``(B, A)``.

        Returns
        -------
        torch.Tensor
            Scalar loss (already reduced) that can be added to the overall
            training objective.
        """
        # Predict tau_c in the range [-1, 1] via tanh output.
        pred_tax = self.predictor(h, z, action_repr).squeeze(-1)
        target = torch.full_like(pred_tax, self.config.target_tax)
        loss = F.mse_loss(pred_tax, target) * self.config.weight
        return loss

    def forward_tax_and_loss(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded Context Tax and associated framing regularisation loss.

        Returns
        -------
        bounded_tax : torch.Tensor
            Context Tax constrained within [-tax_bound, tax_bound].
        loss : torch.Tensor
            Framing regularisation loss penalising deviation from target tax.
        """
        pred_tax = self.predictor(h, z, action_repr).squeeze(-1)
        bounded_tax = self.bound_tax(pred_tax)
        target = torch.full_like(pred_tax, self.config.target_tax)
        loss = F.mse_loss(pred_tax, target) * self.config.weight
        return bounded_tax, loss
