"""
Neural models and representation architectures for tidal-kulture-rwm.
"""

from .predictors import (
    ContextTaxPredictor,
    EngagementPredictor,
    PrototypeSimplexLoss,
    SmoothGiniLoss,
    differentiable_smooth_gini,
)
from .rssm_dynamics import LatentState, RecurrentStateSpaceModel
from .sasrec_backbone import SASRecBackbone, project_to_hypersphere

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
