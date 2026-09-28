"""
Planning and alignment policy package for tidal-kulture-rwm.
"""

from .cafl_framing import CausalAlignmentFramingLayer, FramingConfig
from .mpc_planner import CEMPMPPlanner, CEMPMPPlanner as CrossEntropyMethodPlanner

# PlannerConfig placeholder for compatibility; can be extended if needed.
class PlannerConfig:
    """Configuration holder for the planner (placeholder)."""
    pass

__all__ = [
    "CausalAlignmentFramingLayer",
    "FramingConfig",
    "CEMPMPPlanner",
    "CrossEntropyMethodPlanner",
    "PlannerConfig",
]
