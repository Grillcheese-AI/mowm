# mowm/__init__.py
"""MoWM — General-Purpose Mixture of World Models.

Self-organizing recursive expert worlds with a living axiom library,
14 domain modules, discovery loop via symbolic regression, and
dynamic expert spawn/prune.

Public API:
    MoWMPipeline — End-to-end pipeline orchestrator
    AxiomLibrary — Global world axiom storage
    Axiom — Single axiom dataclass
    World — Recursive expert world model
    MoWMRouter — Dynamic expert routing
    DiscoveryLoop — Symbolic regression discovery
"""

__version__ = "0.1.0"

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.discovery import DiscoveryLoop
from mowm.pipeline import MoWMPipeline, StepResult
from mowm.router import MoWMRouter
from mowm.world import World

__all__ = [
    "Axiom",
    "AxiomLibrary",
    "DiscoveryLoop",
    "MoWMPipeline",
    "MoWMRouter",
    "StepResult",
    "World",
]
