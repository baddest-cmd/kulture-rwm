"""
Neural models and representation architectures for tidal-kulture-rwm.
"""

from .sasrec_backbone import SASRecBackbone, project_to_hypersphere
from .rssm_dynamics import RecurrentStateSpaceModel, LatentState
from .predictors import (
    EngagementPredictor,
    ContextTaxPredictor,
    PrototypeSimplexLoss,
    SmoothGiniLoss,
    differentiable_smooth_gini,
)

__all__ = [
    "SASRecBackbone",
    "project_to_hypersphere",
    "RecurrentStateSpaceModel",
    "LatentState",
    "EngagementPredictor",
    "ContextTaxPredictor",
    "PrototypeSimplexLoss",
    "SmoothGiniLoss",
    "differentiable_smooth_gini",
]
