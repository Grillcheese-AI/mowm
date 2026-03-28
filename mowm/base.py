# mowm/base.py
"""Abstract base classes for MoWM — GoF patterns for swappable components.

Patterns implemented:
  - Strategy: MergeStrategy, SearchStrategy, CleanupStrategy
  - Abstract Factory: WorldFactory
  - Composite: BaseWorld (recursive z-axis tree)
  - Template Method: BaseDomain.seed() with _register_axioms()
  - Observer: DiscoveryObserver
  - Registry: @register_domain (in domains/__init__.py)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary


class MergeStrategy(ABC):
    """Strategy for merging multiple HYLA transition deltas within a world."""

    @abstractmethod
    def merge(
        self,
        deltas: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """Merge deltas into a single transition vector.

        Args:
            deltas: List of (k, l) block-code transition deltas.
            weights: Per-delta weights (same length as deltas).

        Returns:
            Merged (k, l) block-code vector.
        """
        ...


class SearchStrategy(ABC):
    """Strategy for symbolic regression search in the discovery loop."""

    @abstractmethod
    def search(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        operators: list[str],
        max_complexity: int,
    ) -> tuple[str, float]:
        """Search for a formula relating inputs to outputs.

        Args:
            inputs: Training inputs (n_samples, n_features).
            outputs: Training outputs (n_samples,).
            operators: Grammar of allowed operators.
            max_complexity: Maximum AST node count.

        Returns:
            (formula_str, mse) — best formula found and its error.
        """
        ...


class CleanupStrategy(ABC):
    """Strategy for cleaning up noisy block-code vectors."""

    @abstractmethod
    def cleanup(self, vec: np.ndarray) -> np.ndarray:
        """Clean up a block-code vector.

        Args:
            vec: Noisy block-code vector (k, l).

        Returns:
            Cleaned block-code vector (k, l).
        """
        ...


class DiscoveryObserver(ABC):
    """Observer notified when the discovery loop finds a new axiom."""

    @abstractmethod
    def on_axiom_discovered(self, axiom: "Axiom") -> None:
        """Called when a new axiom is discovered and registered.

        Args:
            axiom: The newly discovered Axiom instance.
        """
        ...


class WorldFactory(ABC):
    """Abstract factory for creating World instances with configurable params."""

    @abstractmethod
    def create_world(
        self,
        world_id: int,
        axioms: list["Axiom"],
        z_depth: int,
    ) -> "BaseWorld":
        """Create a new world instance.

        Args:
            world_id: Unique identifier for the world.
            axioms: List of axioms the world operates with.
            z_depth: Depth in the recursive z-axis tree.

        Returns:
            A concrete BaseWorld instance.
        """
        ...


class BaseWorld(ABC):
    """Abstract base for recursive world models (Composite pattern).

    Each world can contain sub-worlds at z+1, forming a recursive tree.
    """

    @abstractmethod
    def predict(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict next state given current state and action.

        Args:
            state_vec: Current state block-code vector (k, l).
            action_vec: Action block-code vector (k, l).

        Returns:
            (predicted_state, confidence) — predicted (k, l) and scalar.
        """
        ...

    @abstractmethod
    def spawn_subworld(self, axioms: list["Axiom"]) -> "BaseWorld":
        """Spawn a child sub-world at z+1."""
        ...

    @abstractmethod
    def consolidate(self, other: "BaseWorld") -> None:
        """Merge another world into this one via Oja's rule."""
        ...


class BaseDomain(ABC):
    """Template method base for domain modules.

    Concrete domains override _register_axioms() to seed axioms.
    The public seed() method provides the skeleton.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain name (e.g. 'physics', 'logic')."""
        ...

    @property
    @abstractmethod
    def operators(self) -> list[str]:
        """Operator grammar for symbolic regression in this domain."""
        ...

    def seed(self, library: "AxiomLibrary") -> None:
        """Seed the axiom library with this domain's axioms (template method)."""
        self._register_axioms(library)

    @abstractmethod
    def _register_axioms(self, library: "AxiomLibrary") -> None:
        """Register all axioms for this domain. Subclasses must implement."""
        ...


class BaseRouter(ABC):
    """Abstract base for routing inputs to expert worlds."""

    @abstractmethod
    def route(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
    ) -> list[tuple[Any, float]]:
        """Route to top-k worlds with weights.

        Args:
            state_vec: State block-code vector (k, l).
            action_vec: Action block-code vector (k, l).

        Returns:
            List of (world, weight) tuples.
        """
        ...

    @abstractmethod
    def maybe_spawn(
        self,
        residual_vec: np.ndarray,
        library: "AxiomLibrary",
    ) -> Any | None:
        """Conditionally spawn a new world if residual is high."""
        ...

    @abstractmethod
    def prune(self, current_step: int) -> list[int]:
        """Prune inactive worlds. Returns list of pruned world IDs."""
        ...


# ── v0.2.0 ABCs ─────────────────────────────────────────────────────────────


class BaseExecutor(ABC):
    """Abstract base for executing axiom formula strings."""

    @abstractmethod
    def execute(
        self,
        formula_str: str,
        inputs: dict[str, float],
    ) -> float | np.ndarray:
        """Evaluate a formula string with named variable inputs."""
        ...

    @abstractmethod
    def parse(self, formula_str: str):
        """Parse a formula string into an internal representation."""
        ...


class BaseCausalGraph(ABC):
    """Abstract base for signed directed causal graphs."""

    @abstractmethod
    def add_node(self, name: str, domain: str, vector: np.ndarray) -> None:
        ...

    @abstractmethod
    def add_edge(
        self, source: str, target: str, polarity: str, strength: float,
    ) -> None:
        ...

    @abstractmethod
    def remove_node(self, name: str) -> None:
        ...

    @abstractmethod
    def remove_edge(self, source: str, target: str) -> None:
        ...

    @abstractmethod
    def propagate(self, node: str, delta: float) -> dict[str, float]:
        ...

    @abstractmethod
    def counterfactual(self, node: str, value: float) -> dict[str, float]:
        ...

    @abstractmethod
    def find_paths(self, source: str, target: str) -> list[list[str]]:
        ...

    @abstractmethod
    def feedback_loops(self) -> list[list[str]]:
        ...

    @abstractmethod
    def encode_to_vsa(self) -> np.ndarray:
        ...


class BaseLogicEngine(ABC):
    """Abstract base for dual-mode logic evaluation."""

    @abstractmethod
    def evaluate(
        self,
        formula_str: str,
        facts: dict[str, bool | float],
        mode: str | None = None,
    ) -> bool | float:
        ...

    @abstractmethod
    def check_obligation(
        self,
        conditions: list[str],
        facts: dict[str, bool],
        exceptions: list[str] | None = None,
    ) -> bool:
        ...
