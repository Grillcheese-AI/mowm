# mowm/__init__.py
"""MoWM — General-Purpose Mixture of World Models.

Self-organizing recursive expert worlds with a living axiom library,
14 domain modules, discovery loop via symbolic regression, and
dynamic expert spawn/prune.

Public API (v0.2.0):
    MoWMPipeline — End-to-end pipeline orchestrator
    AxiomLibrary — Global world axiom storage
    Axiom — Single axiom dataclass
    World — Recursive expert world model
    MoWMRouter — Dynamic expert routing
    DiscoveryLoop — Symbolic regression discovery
    SafeFormulaExecutor — Safe formula evaluation executor
    CausalGraph — VSA-backed causal graph
    LogicEngine — Dual-mode deductive/probabilistic logic
    SequenceEncoder — DNA / temporal / token sequence encoder
    PhasorEngine — Harmonic phasor bank for frequency-domain encoding
    IdentityState — Personal world model with Plutchik emotions
    GridWorld — 2-D simulation environment
    MoWMTrainer — Training loop with prediction + confidence loss
"""

__version__ = "0.2.0"

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.discovery import DiscoveryLoop
from mowm.executor import SafeFormulaExecutor
from mowm.identity_state import IdentityState
from mowm.logic_engine import LogicEngine
from mowm.phasor_engine import PhasorEngine
from mowm.pipeline import MoWMPipeline, StepResult
from mowm.router import MoWMRouter
from mowm.sequence_encoder import SequenceEncoder
from mowm.sim_environment import GridWorld
from mowm.training import MoWMTrainer
from mowm.world import World

__all__ = [
    "Axiom",
    "AxiomLibrary",
    "CausalGraph",
    "DiscoveryLoop",
    "GridWorld",
    "IdentityState",
    "LogicEngine",
    "MoWMPipeline",
    "MoWMRouter",
    "MoWMTrainer",
    "PhasorEngine",
    "SafeFormulaExecutor",
    "SequenceEncoder",
    "StepResult",
    "World",
]
