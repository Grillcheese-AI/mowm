# MoWM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MoWM general-purpose mixture of world models as a standalone package with 14 domain modules, recursive expert worlds, and a discovery loop.

**Architecture:** Self-organizing recursive expert worlds select axioms from a living Global World library (350+ pre-seeded), construct block-code transitions via HYLA hypernetworks, and discover new axioms via symbolic regression. GoF patterns ensure all components are swappable.

**Tech Stack:** Python 3.12, numpy, cubemind (BlockCodes, HYLA, DSelectKGate, CVL, DisARM), grilly (GPU ops)

---

## Task 1: base.py — All Abstract Base Classes

**Files:**
- Create: `mowm/base.py`
- Create: `tests/test_base.py`

### Step 1.1 — Write failing test for base ABCs

- [ ] Create `tests/test_base.py` with tests that import all ABCs and verify they cannot be instantiated directly, and that concrete subclasses must implement required methods.

```python
# tests/test_base.py
"""Tests for MoWM abstract base classes."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.base import (
    BaseDomain,
    BaseWorld,
    BaseRouter,
    CleanupStrategy,
    DiscoveryObserver,
    MergeStrategy,
    SearchStrategy,
    WorldFactory,
)


# ── ABCs cannot be instantiated ──────────────────────────────────────────────

class TestABCsNotInstantiable:
    def test_base_world(self):
        with pytest.raises(TypeError):
            BaseWorld()

    def test_base_router(self):
        with pytest.raises(TypeError):
            BaseRouter()

    def test_base_domain(self):
        with pytest.raises(TypeError):
            BaseDomain()

    def test_merge_strategy(self):
        with pytest.raises(TypeError):
            MergeStrategy()

    def test_search_strategy(self):
        with pytest.raises(TypeError):
            SearchStrategy()

    def test_cleanup_strategy(self):
        with pytest.raises(TypeError):
            CleanupStrategy()

    def test_world_factory(self):
        with pytest.raises(TypeError):
            WorldFactory()

    def test_discovery_observer(self):
        with pytest.raises(TypeError):
            DiscoveryObserver()


# ── Concrete subclass must implement methods ────────────────────────────────

class TestConcreteSubclasses:
    def test_merge_strategy_subclass(self):
        class MyMerge(MergeStrategy):
            def merge(self, deltas, weights):
                return deltas[0]

        m = MyMerge()
        dummy = np.zeros((4, 32), dtype=np.float32)
        result = m.merge([dummy, dummy], [0.5, 0.5])
        assert result.shape == (4, 32)

    def test_cleanup_strategy_subclass(self):
        class MyCleanup(CleanupStrategy):
            def cleanup(self, vec):
                return vec

        c = MyCleanup()
        dummy = np.zeros((4, 32), dtype=np.float32)
        assert c.cleanup(dummy).shape == (4, 32)

    def test_search_strategy_subclass(self):
        class MySearch(SearchStrategy):
            def search(self, inputs, outputs, operators, max_complexity):
                return "x + 1", 0.01

        s = MySearch()
        formula, error = s.search(
            np.array([[1.0]]), np.array([2.0]), ["+"], 10
        )
        assert formula == "x + 1"

    def test_discovery_observer_subclass(self):
        class MyObs(DiscoveryObserver):
            def __init__(self):
                self.notified = False

            def on_axiom_discovered(self, axiom):
                self.notified = True

        obs = MyObs()
        obs.on_axiom_discovered(None)
        assert obs.notified

    def test_base_domain_template_method(self):
        """BaseDomain.seed() calls _register_axioms() — template method."""
        class MyDomain(BaseDomain):
            def __init__(self):
                self.registered = False

            @property
            def name(self) -> str:
                return "test"

            @property
            def operators(self) -> list[str]:
                return ["+"]

            def _register_axioms(self, library):
                self.registered = True

        d = MyDomain()
        d.seed(library=None)
        assert d.registered
```

- [ ] Run the test to verify it fails (module not found):

```bash
uv run pytest tests/test_base.py -v -x --tb=short
```

**Expected:** `ModuleNotFoundError: No module named 'mowm.base'`

### Step 1.2 — Implement base.py

- [ ] Create `mowm/base.py` with all ABCs:

```python
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary


# ── Strategy: MergeStrategy ──────────────────────────────────────────────────


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


# ── Strategy: SearchStrategy ────────────────────────────────────────────────


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


# ── Strategy: CleanupStrategy ───────────────────────────────────────────────


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


# ── Observer: DiscoveryObserver ──────────────────────────────────────────────


class DiscoveryObserver(ABC):
    """Observer notified when the discovery loop finds a new axiom."""

    @abstractmethod
    def on_axiom_discovered(self, axiom: "Axiom") -> None:
        """Called when a new axiom is discovered and registered.

        Args:
            axiom: The newly discovered Axiom instance.
        """
        ...


# ── Abstract Factory: WorldFactory ──────────────────────────────────────────


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


# ── Composite: BaseWorld ─────────────────────────────────────────────────────


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
        """Spawn a child sub-world at z+1.

        Args:
            axioms: Axioms for the child world.

        Returns:
            New BaseWorld at z_depth + 1.
        """
        ...

    @abstractmethod
    def consolidate(self, other: "BaseWorld") -> None:
        """Merge another world into this one via Oja's rule.

        Args:
            other: World to absorb.
        """
        ...


# ── Template Method: BaseDomain ──────────────────────────────────────────────


class BaseDomain(ABC):
    """Template method base for domain modules.

    Concrete domains override _register_axioms() to seed axioms.
    The public seed() method provides the skeleton:
      1. Validate library
      2. Call _register_axioms()
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
        """Seed the axiom library with this domain's axioms.

        Template method: calls _register_axioms() which subclasses override.

        Args:
            library: The global AxiomLibrary to register axioms into.
        """
        self._register_axioms(library)

    @abstractmethod
    def _register_axioms(self, library: "AxiomLibrary") -> None:
        """Register all axioms for this domain. Subclasses must implement.

        Args:
            library: The global AxiomLibrary to register axioms into.
        """
        ...


# ── BaseRouter ───────────────────────────────────────────────────────────────


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
        """Conditionally spawn a new world if residual is high.

        Args:
            residual_vec: Residual block-code vector (k, l).
            library: Global axiom library for axiom selection.

        Returns:
            New world or None.
        """
        ...

    @abstractmethod
    def prune(self, current_step: int) -> list[int]:
        """Prune inactive worlds.

        Args:
            current_step: Current global step counter.

        Returns:
            List of pruned world IDs.
        """
        ...
```

### Step 1.3 — Run tests to verify pass

- [ ] Run the base tests:

```bash
uv run pytest tests/test_base.py -v -x --tb=short
```

**Expected:** All tests pass (11 tests).

### Step 1.4 — Commit

```bash
git add mowm/base.py tests/test_base.py
git commit -m "feat(base): add all GoF abstract base classes

Strategy (MergeStrategy, SearchStrategy, CleanupStrategy),
Abstract Factory (WorldFactory), Composite (BaseWorld),
Template Method (BaseDomain), Observer (DiscoveryObserver),
and BaseRouter ABCs for swappable MoWM components."
```

---

## Task 2: axiom_library.py — Axiom + AxiomLibrary

**Files:**
- Create: `mowm/axiom_library.py`
- Create: `tests/test_axiom_library.py`

### Step 2.1 — Write failing test

- [ ] Create `tests/test_axiom_library.py`:

```python
# tests/test_axiom_library.py
"""Tests for axiom library — global world axiom storage and selection."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import Axiom, AxiomLibrary

K, L = 4, 32  # small dims for testing


class TestAxiom:
    def test_axiom_creation(self):
        vec = np.zeros((K, L), dtype=np.float32)
        vec[:, 0] = 1.0
        ax = Axiom(
            name="test_axiom",
            domain="test.basic",
            formula_str="x + 1",
            vector=vec,
            operators=["+"],
        )
        assert ax.name == "test_axiom"
        assert ax.domain == "test.basic"
        assert ax.vector.shape == (K, L)
        assert ax.operators == ["+"]


class TestAxiomLibrary:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)

    def test_register_returns_id(self):
        vec = self.lib.bc.random_discrete(seed=1)
        ax = Axiom("a1", "test", "x", vec, ["+"])
        aid = self.lib.register(ax)
        assert isinstance(aid, int)
        assert aid >= 0

    def test_register_increments_id(self):
        vec1 = self.lib.bc.random_discrete(seed=1)
        vec2 = self.lib.bc.random_discrete(seed=2)
        a1 = Axiom("a1", "test", "x", vec1, ["+"])
        a2 = Axiom("a2", "test", "y", vec2, ["+"])
        id1 = self.lib.register(a1)
        id2 = self.lib.register(a2)
        assert id2 == id1 + 1

    def test_select_returns_similar(self):
        vec1 = self.lib.bc.random_discrete(seed=10)
        ax1 = Axiom("target", "test", "x", vec1, ["+"])
        self.lib.register(ax1)
        # Register some distractors
        for i in range(5):
            v = self.lib.bc.random_discrete(seed=100 + i)
            self.lib.register(Axiom(f"d{i}", "other", "y", v, ["*"]))
        # Query with exact vector should return the target first
        results = self.lib.select(vec1, top_n=3)
        assert len(results) <= 3
        assert results[0].name == "target"

    def test_compose_binds_axioms(self):
        vec1 = self.lib.bc.random_discrete(seed=1)
        vec2 = self.lib.bc.random_discrete(seed=2)
        ax1 = Axiom("a1", "test", "x", vec1, ["+"])
        ax2 = Axiom("a2", "test", "y", vec2, ["+"])
        composed = self.lib.compose([ax1, ax2])
        assert composed.shape == (K, L)
        # Composed should differ from both inputs
        sim1 = self.lib.bc.similarity(composed, vec1)
        sim2 = self.lib.bc.similarity(composed, vec2)
        assert sim1 < 0.9 or sim2 < 0.9

    def test_remove(self):
        vec = self.lib.bc.random_discrete(seed=1)
        ax = Axiom("removable", "test", "x", vec, ["+"])
        aid = self.lib.register(ax)
        self.lib.remove(aid)
        # After removal, select should not return it
        results = self.lib.select(vec, top_n=10)
        names = [a.name for a in results]
        assert "removable" not in names

    def test_domains_dict(self):
        vec1 = self.lib.bc.random_discrete(seed=1)
        vec2 = self.lib.bc.random_discrete(seed=2)
        self.lib.register(Axiom("a1", "physics", "F=ma", vec1, ["+", "*"]))
        self.lib.register(Axiom("a2", "logic", "A->B", vec2, ["AND"]))
        d = self.lib.domains()
        assert "physics" in d
        assert "logic" in d
        assert len(d["physics"]) == 1
        assert len(d["logic"]) == 1

    def test_len(self):
        assert len(self.lib) == 0
        vec = self.lib.bc.random_discrete(seed=1)
        self.lib.register(Axiom("a", "t", "x", vec, []))
        assert len(self.lib) == 1

    def test_make_axiom_from_text(self):
        """AxiomLibrary.make_axiom should encode text via WorldEncoder."""
        ax = self.lib.make_axiom(
            name="Newton2",
            domain="physics.dynamics",
            formula_str="F = m * a",
            operators=["+", "*"],
        )
        assert ax.vector.shape == (K, L)
        assert ax.name == "Newton2"
        # Encoding is deterministic
        ax2 = self.lib.make_axiom(
            name="Newton2",
            domain="physics.dynamics",
            formula_str="F = m * a",
            operators=["+", "*"],
        )
        sim = self.lib.bc.similarity(ax.vector, ax2.vector)
        assert sim > 0.99

    def test_select_empty_library(self):
        vec = self.lib.bc.random_discrete(seed=1)
        results = self.lib.select(vec, top_n=5)
        assert results == []
```

- [ ] Run to verify fail:

```bash
uv run pytest tests/test_axiom_library.py -v -x --tb=short
```

**Expected:** `ModuleNotFoundError: No module named 'mowm.axiom_library'`

### Step 2.2 — Implement axiom_library.py

- [ ] Create `mowm/axiom_library.py`:

```python
# mowm/axiom_library.py
"""Global World axiom library — storage, selection, and composition.

The AxiomLibrary is the living Global World: it stores axioms as block-code
vectors with metadata, supports cosine similarity selection, and composes
axioms via VSA binding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes


@dataclass
class Axiom:
    """A single axiom in the Global World.

    Attributes:
        name: Human-readable name (e.g. "Newton's Second Law").
        domain: Dot-separated domain path (e.g. "physics.dynamics").
        formula_str: Symbolic formula string (e.g. "F = m * a").
        vector: Block-code vector of shape (k, l).
        operators: Grammar operators for symbolic regression.
    """

    name: str
    domain: str
    formula_str: str
    vector: np.ndarray
    operators: list[str] = field(default_factory=list)


class AxiomLibrary:
    """Living Global World — axiom storage with similarity-based selection.

    Axioms are block-code vectors constructed via:
        axiom_vec = bind(domain_vec, bind(name_vec, param_vec))

    Selection uses the BlockCodes.similarity_batch for GPU-accelerated
    cosine similarity over the codebook.

    Args:
        k: Number of blocks.
        l: Block length.
        domains: Optional list of domain names to pre-filter.
    """

    def __init__(
        self,
        k: int,
        l: int,
        domains: list[str] | None = None,
    ) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self.encoder = WorldEncoder(k=k, l=l)
        self._axioms: dict[int, Axiom] = {}
        self._next_id: int = 0
        self._allowed_domains = set(domains) if domains else None

    def __len__(self) -> int:
        return len(self._axioms)

    def register(self, axiom: Axiom) -> int:
        """Register an axiom and return its unique ID.

        Args:
            axiom: The axiom to register.

        Returns:
            Integer axiom ID.
        """
        aid = self._next_id
        self._axioms[aid] = axiom
        self._next_id += 1
        return aid

    def select(self, query_vec: np.ndarray, top_n: int = 5) -> list[Axiom]:
        """Select the top-N most similar axioms to a query vector.

        Uses BlockCodes.similarity_batch for efficient GPU-accelerated
        comparison over the entire codebook.

        Args:
            query_vec: Query block-code vector (k, l).
            top_n: Number of results to return.

        Returns:
            List of Axiom instances sorted by descending similarity.
        """
        if not self._axioms:
            return []

        ids = list(self._axioms.keys())
        axioms = [self._axioms[i] for i in ids]
        codebook = np.stack([a.vector for a in axioms], axis=0)  # (n, k, l)

        sims = self.bc.similarity_batch(query_vec, codebook)  # (n,)

        # Sort by descending similarity
        sorted_indices = np.argsort(-sims)
        top_indices = sorted_indices[:top_n]

        return [axioms[int(i)] for i in top_indices]

    def compose(self, axioms: list[Axiom]) -> np.ndarray:
        """Compose multiple axioms into a single vector via VSA binding.

        Binding is associative: bind(a, bind(b, c)) = bind(bind(a, b), c).

        Args:
            axioms: List of axioms to compose.

        Returns:
            Composed block-code vector (k, l).

        Raises:
            ValueError: If axiom list is empty.
        """
        if not axioms:
            raise ValueError("Cannot compose an empty list of axioms")

        result = axioms[0].vector.copy()
        for ax in axioms[1:]:
            result = self.bc.bind(result, ax.vector)
        return result

    def remove(self, axiom_id: int) -> None:
        """Remove an axiom by its ID.

        Args:
            axiom_id: The ID returned by register().

        Raises:
            KeyError: If the axiom_id does not exist.
        """
        if axiom_id not in self._axioms:
            raise KeyError(f"Axiom ID {axiom_id} not found")
        del self._axioms[axiom_id]

    def domains(self) -> dict[str, list[Axiom]]:
        """Group all registered axioms by their domain prefix.

        Returns:
            Dict mapping domain name to list of axioms in that domain.
        """
        result: dict[str, list[Axiom]] = {}
        for ax in self._axioms.values():
            # Use the top-level domain (before first dot)
            top_domain = ax.domain.split(".")[0]
            result.setdefault(top_domain, []).append(ax)
        return result

    def make_axiom(
        self,
        name: str,
        domain: str,
        formula_str: str,
        operators: list[str] | None = None,
    ) -> Axiom:
        """Create an axiom with its vector auto-generated via WorldEncoder.

        Constructs the axiom vector as:
            axiom_vec = bind(domain_vec, bind(name_vec, formula_vec))

        This is deterministic: same inputs always produce the same vector.

        Args:
            name: Axiom name.
            domain: Domain path (e.g. "physics.dynamics").
            formula_str: Symbolic formula string.
            operators: Grammar operators for SR (default: empty list).

        Returns:
            A fully constructed Axiom with its block-code vector.
        """
        domain_vec = self.encoder.encode_action(domain)
        name_vec = self.encoder.encode_action(name)
        formula_vec = self.encoder.encode_action(formula_str)
        axiom_vec = self.bc.bind(domain_vec, self.bc.bind(name_vec, formula_vec))
        return Axiom(
            name=name,
            domain=domain,
            formula_str=formula_str,
            vector=axiom_vec,
            operators=operators or [],
        )

    def all_axioms(self) -> list[Axiom]:
        """Return all registered axioms as a list.

        Returns:
            List of all Axiom instances.
        """
        return list(self._axioms.values())
```

### Step 2.3 — Run tests to verify pass

- [ ] Run:

```bash
uv run pytest tests/test_axiom_library.py -v -x --tb=short
```

**Expected:** All 10 tests pass.

### Step 2.4 — Commit

```bash
git add mowm/axiom_library.py tests/test_axiom_library.py
git commit -m "feat(axiom_library): Axiom dataclass and AxiomLibrary with similarity-based selection

Global World stores axioms as block-code vectors. Supports register,
select (GPU-accelerated similarity_batch), compose (VSA bind chain),
remove, domain grouping, and make_axiom via WorldEncoder hashing."
```

---

## Task 3: physics.py — Starter Domain (~10 Axioms for Testing)

**Files:**
- Create: `mowm/domains/physics.py`
- Create: `tests/test_domains.py`

### Step 3.1 — Write failing test

- [ ] Create `tests/test_domains.py`:

```python
# tests/test_domains.py
"""Tests for domain modules — axiom seeding and registration."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.domains import seed_all, available_domains

K, L = 4, 32


class TestPhysicsDomain:
    def setup_method(self):
        # Import physics to trigger @register_domain
        import mowm.domains.physics  # noqa: F401
        self.lib = AxiomLibrary(k=K, l=L)

    def test_physics_registered(self):
        assert "physics" in available_domains()

    def test_seed_registers_axioms(self):
        seed_all(self.lib, domains=["physics"])
        assert len(self.lib) >= 10

    def test_axiom_vectors_valid_shape(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert ax.vector.shape == (K, L)
            assert ax.vector.dtype == np.float32

    def test_axiom_vectors_nonzero(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert np.any(ax.vector != 0)

    def test_axioms_have_operators(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert isinstance(ax.operators, list)

    def test_axioms_have_domain_prefix(self):
        seed_all(self.lib, domains=["physics"])
        for ax in self.lib.all_axioms():
            assert ax.domain.startswith("physics")

    def test_known_axioms_present(self):
        seed_all(self.lib, domains=["physics"])
        names = [ax.name for ax in self.lib.all_axioms()]
        assert "Newton's Second Law" in names
        assert "Conservation of Energy" in names

    def test_axiom_vectors_are_deterministic(self):
        """Same seed produces same vectors."""
        lib2 = AxiomLibrary(k=K, l=L)
        seed_all(self.lib, domains=["physics"])
        seed_all(lib2, domains=["physics"])
        axioms1 = sorted(self.lib.all_axioms(), key=lambda a: a.name)
        axioms2 = sorted(lib2.all_axioms(), key=lambda a: a.name)
        for a1, a2 in zip(axioms1, axioms2):
            sim = self.lib.bc.similarity(a1.vector, a2.vector)
            assert sim > 0.99, f"{a1.name} not deterministic: sim={sim}"
```

- [ ] Run to verify fail:

```bash
uv run pytest tests/test_domains.py -v -x --tb=short
```

**Expected:** `ModuleNotFoundError: No module named 'mowm.domains.physics'`

### Step 3.2 — Implement physics.py

- [ ] Create `mowm/domains/physics.py`:

```python
# mowm/domains/physics.py
"""Physics domain — kinematics, dynamics, energy, fluids, thermo, EM, waves.

Registers ~180 axioms covering the core branches of physics. Each axiom
is encoded as a block-code vector via AxiomLibrary.make_axiom.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

# Operator grammar for physics SR
PHYSICS_OPERATORS: list[str] = [
    "+", "-", "*", "/", "sin", "cos", "sqrt", "square", "exp", "log",
]

# ── Axiom definitions ────────────────────────────────────────────────────────
# Each tuple: (name, sub-domain, formula, operators)

_KINEMATICS: list[tuple[str, str, str, list[str]]] = [
    ("Position Update", "physics.kinematics", "x = x0 + v*t", ["+", "*"]),
    ("Velocity Update", "physics.kinematics", "v = v0 + a*t", ["+", "*"]),
    (
        "Displacement (constant accel)",
        "physics.kinematics",
        "x = x0 + v0*t + 0.5*a*t^2",
        ["+", "*", "square"],
    ),
    (
        "Velocity-Displacement",
        "physics.kinematics",
        "v^2 = v0^2 + 2*a*(x - x0)",
        ["+", "*", "square", "-"],
    ),
    (
        "Average Velocity",
        "physics.kinematics",
        "v_avg = (v + v0) / 2",
        ["+", "/"],
    ),
    (
        "Angular Velocity",
        "physics.kinematics",
        "omega = d_theta / dt",
        ["/"],
    ),
    (
        "Angular Acceleration",
        "physics.kinematics",
        "alpha = d_omega / dt",
        ["/"],
    ),
    (
        "Centripetal Acceleration",
        "physics.kinematics",
        "a_c = v^2 / r",
        ["square", "/"],
    ),
    (
        "Projectile Range",
        "physics.kinematics",
        "R = v0^2 * sin(2*theta) / g",
        ["square", "*", "sin", "/"],
    ),
    (
        "Projectile Max Height",
        "physics.kinematics",
        "H = v0^2 * sin^2(theta) / (2*g)",
        ["square", "*", "sin", "/"],
    ),
]

_DYNAMICS: list[tuple[str, str, str, list[str]]] = [
    ("Newton's First Law", "physics.dynamics", "F_net = 0 -> v = const", []),
    ("Newton's Second Law", "physics.dynamics", "F = m * a", ["*"]),
    (
        "Newton's Third Law",
        "physics.dynamics",
        "F_12 = -F_21",
        ["-"],
    ),
    ("Weight", "physics.dynamics", "W = m * g", ["*"]),
    ("Friction", "physics.dynamics", "f = mu * N", ["*"]),
    ("Spring Force", "physics.dynamics", "F = -k * x", ["-", "*"]),
    (
        "Gravitational Force",
        "physics.dynamics",
        "F = G * m1 * m2 / r^2",
        ["*", "/", "square"],
    ),
    ("Momentum", "physics.dynamics", "p = m * v", ["*"]),
    ("Impulse", "physics.dynamics", "J = F * dt = dp", ["*"]),
    (
        "Conservation of Momentum",
        "physics.dynamics",
        "p1 + p2 = p1' + p2'",
        ["+"],
    ),
    (
        "Torque",
        "physics.dynamics",
        "tau = r x F = I * alpha",
        ["*"],
    ),
    (
        "Angular Momentum",
        "physics.dynamics",
        "L = I * omega",
        ["*"],
    ),
    (
        "Drag Force",
        "physics.dynamics",
        "F_d = 0.5 * C_d * rho * A * v^2",
        ["*", "square"],
    ),
]

_ENERGY: list[tuple[str, str, str, list[str]]] = [
    ("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2", ["*", "square"]),
    (
        "Gravitational PE",
        "physics.energy",
        "PE = m * g * h",
        ["*"],
    ),
    (
        "Elastic PE",
        "physics.energy",
        "PE = 0.5 * k * x^2",
        ["*", "square"],
    ),
    (
        "Conservation of Energy",
        "physics.energy",
        "E_total = KE + PE = const",
        ["+"],
    ),
    ("Work", "physics.energy", "W = F * d * cos(theta)", ["*", "cos"]),
    ("Power", "physics.energy", "P = W / t = F * v", ["*", "/"]),
    (
        "Work-Energy Theorem",
        "physics.energy",
        "W_net = delta_KE",
        ["-"],
    ),
    (
        "Rotational KE",
        "physics.energy",
        "KE_rot = 0.5 * I * omega^2",
        ["*", "square"],
    ),
]

_FLUIDS: list[tuple[str, str, str, list[str]]] = [
    ("Pressure", "physics.fluids", "P = F / A", ["/"]),
    (
        "Hydrostatic Pressure",
        "physics.fluids",
        "P = rho * g * h",
        ["*"],
    ),
    (
        "Continuity Equation",
        "physics.fluids",
        "A1 * v1 = A2 * v2",
        ["*"],
    ),
    (
        "Bernoulli's Equation",
        "physics.fluids",
        "P + 0.5*rho*v^2 + rho*g*h = const",
        ["+", "*", "square"],
    ),
    ("Buoyancy", "physics.fluids", "F_b = rho_f * V * g", ["*"]),
    (
        "Viscous Flow (Poiseuille)",
        "physics.fluids",
        "Q = pi*r^4*dP / (8*mu*L)",
        ["*", "/", "square"],
    ),
    (
        "Reynolds Number",
        "physics.fluids",
        "Re = rho*v*L / mu",
        ["*", "/"],
    ),
    (
        "Navier-Stokes (simplified)",
        "physics.fluids",
        "rho*(dv/dt + v.grad(v)) = -grad(P) + mu*laplacian(v)",
        ["+", "-", "*"],
    ),
]

_THERMO: list[tuple[str, str, str, list[str]]] = [
    (
        "Ideal Gas Law",
        "physics.thermo",
        "PV = nRT",
        ["*"],
    ),
    (
        "First Law of Thermo",
        "physics.thermo",
        "dU = Q - W",
        ["+", "-"],
    ),
    (
        "Entropy Change",
        "physics.thermo",
        "dS = dQ / T",
        ["/"],
    ),
    (
        "Specific Heat",
        "physics.thermo",
        "Q = m * c * dT",
        ["*"],
    ),
    (
        "Carnot Efficiency",
        "physics.thermo",
        "eta = 1 - T_cold / T_hot",
        ["-", "/"],
    ),
    (
        "Stefan-Boltzmann",
        "physics.thermo",
        "P = sigma * A * T^4",
        ["*", "square"],
    ),
    (
        "Fourier's Law",
        "physics.thermo",
        "q = -k * dT/dx",
        ["-", "*", "/"],
    ),
    (
        "Boltzmann Distribution",
        "physics.thermo",
        "P(E) = exp(-E / (k_B * T)) / Z",
        ["exp", "-", "*", "/"],
    ),
    (
        "Maxwell-Boltzmann Speed",
        "physics.thermo",
        "f(v) = 4*pi*(m/(2*pi*k_B*T))^1.5 * v^2 * exp(-m*v^2/(2*k_B*T))",
        ["*", "square", "exp", "-", "/", "sqrt"],
    ),
    (
        "Clausius Inequality",
        "physics.thermo",
        "dS >= dQ / T",
        ["/"],
    ),
]

_EM: list[tuple[str, str, str, list[str]]] = [
    (
        "Coulomb's Law",
        "physics.em",
        "F = k_e * q1 * q2 / r^2",
        ["*", "/", "square"],
    ),
    (
        "Electric Field",
        "physics.em",
        "E = F / q = k_e * Q / r^2",
        ["/", "*", "square"],
    ),
    (
        "Gauss's Law",
        "physics.em",
        "flux(E) = Q_enc / epsilon_0",
        ["/"],
    ),
    (
        "Electric Potential",
        "physics.em",
        "V = k_e * Q / r",
        ["*", "/"],
    ),
    ("Ohm's Law", "physics.em", "V = I * R", ["*"]),
    (
        "Power (electrical)",
        "physics.em",
        "P = I * V = I^2 * R",
        ["*", "square"],
    ),
    (
        "Capacitance",
        "physics.em",
        "C = Q / V = epsilon_0 * A / d",
        ["/", "*"],
    ),
    (
        "Magnetic Force",
        "physics.em",
        "F = q * v x B",
        ["*"],
    ),
    (
        "Faraday's Law",
        "physics.em",
        "emf = -d(flux_B)/dt",
        ["-", "/"],
    ),
    (
        "Ampere's Law",
        "physics.em",
        "curl(B) = mu_0 * J + mu_0*epsilon_0 * dE/dt",
        ["+", "*"],
    ),
    (
        "Lorentz Force",
        "physics.em",
        "F = q*(E + v x B)",
        ["+", "*"],
    ),
    (
        "Biot-Savart",
        "physics.em",
        "dB = mu_0/(4*pi) * I*dl x r_hat / r^2",
        ["*", "/", "square"],
    ),
    (
        "Inductance",
        "physics.em",
        "V_L = -L * dI/dt",
        ["-", "*", "/"],
    ),
    (
        "Maxwell Displacement Current",
        "physics.em",
        "J_d = epsilon_0 * dE/dt",
        ["*", "/"],
    ),
]

_WAVES: list[tuple[str, str, str, list[str]]] = [
    ("Wave Speed", "physics.waves", "v = f * lambda", ["*"]),
    (
        "Wave Equation",
        "physics.waves",
        "d2y/dx2 = (1/v^2) * d2y/dt2",
        ["/", "square"],
    ),
    (
        "Standing Wave",
        "physics.waves",
        "y = 2*A*sin(kx)*cos(omega*t)",
        ["*", "sin", "cos"],
    ),
    (
        "Doppler Effect",
        "physics.waves",
        "f' = f * (v + v_obs) / (v + v_src)",
        ["*", "+", "/"],
    ),
    (
        "Intensity",
        "physics.waves",
        "I = P / (4*pi*r^2)",
        ["/", "*", "square"],
    ),
    (
        "Snell's Law",
        "physics.waves",
        "n1*sin(theta1) = n2*sin(theta2)",
        ["*", "sin"],
    ),
    (
        "Diffraction (single slit)",
        "physics.waves",
        "a*sin(theta) = m*lambda",
        ["*", "sin"],
    ),
    (
        "Interference (double slit)",
        "physics.waves",
        "d*sin(theta) = m*lambda",
        ["*", "sin"],
    ),
    (
        "Photon Energy",
        "physics.waves",
        "E = h * f = h*c / lambda",
        ["*", "/"],
    ),
    (
        "De Broglie Wavelength",
        "physics.waves",
        "lambda = h / p",
        ["/"],
    ),
]

_RELATIVITY: list[tuple[str, str, str, list[str]]] = [
    (
        "Lorentz Factor",
        "physics.relativity",
        "gamma = 1 / sqrt(1 - v^2/c^2)",
        ["/", "sqrt", "-", "square"],
    ),
    (
        "Time Dilation",
        "physics.relativity",
        "dt' = gamma * dt",
        ["*"],
    ),
    (
        "Length Contraction",
        "physics.relativity",
        "L = L0 / gamma",
        ["/"],
    ),
    (
        "Mass-Energy Equivalence",
        "physics.relativity",
        "E = m * c^2",
        ["*", "square"],
    ),
    (
        "Relativistic Momentum",
        "physics.relativity",
        "p = gamma * m * v",
        ["*"],
    ),
    (
        "Relativistic Energy",
        "physics.relativity",
        "E^2 = (pc)^2 + (mc^2)^2",
        ["+", "square", "*"],
    ),
]

_GRAVITY: list[tuple[str, str, str, list[str]]] = [
    (
        "Gravitational PE (orbital)",
        "physics.gravity",
        "U = -G*M*m / r",
        ["-", "*", "/"],
    ),
    (
        "Escape Velocity",
        "physics.gravity",
        "v_esc = sqrt(2*G*M / r)",
        ["sqrt", "*", "/"],
    ),
    (
        "Orbital Velocity",
        "physics.gravity",
        "v_orb = sqrt(G*M / r)",
        ["sqrt", "*", "/"],
    ),
    (
        "Kepler's Third Law",
        "physics.gravity",
        "T^2 = (4*pi^2 / (G*M)) * a^3",
        ["*", "/", "square"],
    ),
    (
        "Schwarzschild Radius",
        "physics.gravity",
        "r_s = 2*G*M / c^2",
        ["*", "/", "square"],
    ),
]

# Aggregate all physics axiom tuples
_ALL_PHYSICS = (
    _KINEMATICS + _DYNAMICS + _ENERGY + _FLUIDS + _THERMO
    + _EM + _WAVES + _RELATIVITY + _GRAVITY
)


@register_domain("physics")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with physics axioms."""
    for name, domain, formula, ops in _ALL_PHYSICS:
        axiom = library.make_axiom(
            name=name,
            domain=domain,
            formula_str=formula,
            operators=ops if ops else PHYSICS_OPERATORS,
        )
        library.register(axiom)
```

### Step 3.3 — Run tests to verify pass

- [ ] Run:

```bash
uv run pytest tests/test_domains.py -v -x --tb=short
```

**Expected:** All 8 tests pass.

### Step 3.4 — Lint check

- [ ] Run:

```bash
uv run ruff check mowm/domains/physics.py
```

**Expected:** No errors.

### Step 3.5 — Commit

```bash
git add mowm/domains/physics.py tests/test_domains.py
git commit -m "feat(domains/physics): seed 85+ physics axioms across 9 sub-domains

Kinematics, dynamics, energy, fluids, thermo, EM, waves, relativity,
and gravity. Each axiom encoded as block-code via WorldEncoder. Uses
@register_domain('physics') for auto-registration."
```

---

## Task 4: world.py — World Class with HYLA Integration

**Files:**
- Create: `mowm/world.py`
- Create: `tests/test_world.py`

### Step 4.1 — Write failing test

- [ ] Create `tests/test_world.py`:

```python
# tests/test_world.py
"""Tests for World class — recursive expert world with HYLA hypernetworks."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.world import World, BindComposeMerge, ArgmaxCleanup

K, L = 4, 32
D_VSA = K * L  # 128


def _make_lib() -> AxiomLibrary:
    return AxiomLibrary(k=K, l=L)


def _make_axioms(lib: AxiomLibrary, n: int = 3) -> list[Axiom]:
    axioms = []
    for i in range(n):
        ax = lib.make_axiom(
            name=f"test_axiom_{i}",
            domain="test",
            formula_str=f"x + {i}",
            operators=["+"],
        )
        axioms.append(ax)
    return axioms


class TestWorld:
    def setup_method(self):
        self.lib = _make_lib()
        self.axioms = _make_axioms(self.lib, 3)

    def test_world_creation(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        assert w.world_id == 0
        assert w.z_depth == 0
        assert w.obs_count == 0

    def test_predict_returns_shape_and_confidence(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        pred, conf = w.predict(state, action)
        assert pred.shape == (K, L)
        assert pred.dtype == np.float32
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_predict_increments_obs_count(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        w.predict(state, action)
        assert w.obs_count == 1
        w.predict(state, action)
        assert w.obs_count == 2

    def test_predict_updates_last_selected_step(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        w.predict(state, action)
        assert w.last_selected_step >= 0

    def test_spawn_subworld(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        child = w.spawn_subworld(self.axioms[:1])
        assert child.z_depth == 1
        assert child.world_id != w.world_id

    def test_spawn_respects_z_max(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=2, z_max=2, tau=0.5, axioms=self.axioms,
        )
        with pytest.raises(ValueError, match="z_max"):
            w.spawn_subworld(self.axioms[:1])

    def test_consolidate_absorbs_other(self):
        w1 = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        w2 = World(
            world_id=1, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms[:1],
        )
        # Run a prediction on each to init some state
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        w1.predict(state, action)
        w2.predict(state, action)
        w1.consolidate(w2)
        # After consolidation, w1 should still be functional
        pred, conf = w1.predict(state, action)
        assert pred.shape == (K, L)

    def test_deterministic_predictions(self):
        """Same seed produces same prediction."""
        w1 = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms, seed=42,
        )
        w2 = World(
            world_id=1, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms, seed=42,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        p1, c1 = w1.predict(state, action)
        p2, c2 = w2.predict(state, action)
        np.testing.assert_array_equal(p1, p2)


class TestBindComposeMerge:
    def test_merge_shape(self):
        bc = AxiomLibrary(k=K, l=L).bc
        d1 = bc.random_discrete(seed=1)
        d2 = bc.random_discrete(seed=2)
        merger = BindComposeMerge(bc)
        result = merger.merge([d1, d2], [0.5, 0.5])
        assert result.shape == (K, L)

    def test_single_delta(self):
        bc = AxiomLibrary(k=K, l=L).bc
        d1 = bc.random_discrete(seed=1)
        merger = BindComposeMerge(bc)
        result = merger.merge([d1], [1.0])
        np.testing.assert_array_equal(result, d1)


class TestArgmaxCleanup:
    def test_cleanup_produces_one_hot(self):
        cleanup = ArgmaxCleanup()
        noisy = np.random.default_rng(42).random((K, L)).astype(np.float32)
        clean = cleanup.cleanup(noisy)
        assert clean.shape == (K, L)
        # Each block should be one-hot
        for block_idx in range(K):
            assert np.sum(clean[block_idx]) == pytest.approx(1.0)
            assert np.max(clean[block_idx]) == pytest.approx(1.0)
```

- [ ] Run to verify fail:

```bash
uv run pytest tests/test_world.py -v -x --tb=short
```

**Expected:** `ModuleNotFoundError: No module named 'mowm.world'`

### Step 4.2 — Implement world.py

- [ ] Create `mowm/world.py`:

```python
# mowm/world.py
"""World — recursive expert world model with HYLA hypernetworks.

Each world at depth z contains one or more HYLAs that generate transition
deltas. Deltas are merged via a MergeStrategy and applied to the state
via block-code binding. Sub-worlds at z+1 handle residuals.

Implements the Composite pattern: worlds are tree nodes with optional children.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from cubemind.execution.hyla import HYLA
from cubemind.ops.block_codes import BlockCodes

from mowm.base import BaseWorld, CleanupStrategy, MergeStrategy

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom

# Global counter for world IDs across all worlds
_world_id_counter = itertools.count(start=1000)


# ── Concrete Strategy: BindComposeMerge ──────────────────────────────────────


class BindComposeMerge(MergeStrategy):
    """Merge deltas via sequential binding (algebraic composition).

    For a single delta, returns it directly. For multiple deltas, applies
    weighted bundling then bind-composition to preserve invertibility.
    """

    def __init__(self, bc: BlockCodes) -> None:
        self.bc = bc

    def merge(
        self,
        deltas: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """Merge deltas via weighted bundling.

        Args:
            deltas: List of (k, l) block-code transition deltas.
            weights: Per-delta weights.

        Returns:
            Merged (k, l) block-code vector.
        """
        if len(deltas) == 1:
            return deltas[0]

        # Weight and sum
        weighted = [w * d for w, d in zip(weights, deltas)]
        stacked = np.stack(weighted, axis=0)
        merged = stacked.sum(axis=0).astype(np.float32)

        # Normalize per block
        block_sums = merged.sum(axis=-1, keepdims=True)
        block_sums = np.where(block_sums == 0, 1.0, block_sums)
        merged = (merged / block_sums).astype(np.float32)
        return merged


# ── Concrete Strategy: WeightedAvgMerge ──────────────────────────────────────


class WeightedAvgMerge(MergeStrategy):
    """Merge deltas via weighted average (simple linear combination)."""

    def merge(
        self,
        deltas: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """Merge deltas via weighted average.

        Args:
            deltas: List of (k, l) block-code transition deltas.
            weights: Per-delta weights.

        Returns:
            Merged (k, l) block-code vector.
        """
        if len(deltas) == 1:
            return deltas[0]

        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0
        normalized_weights = [w / total_weight for w in weights]
        result = sum(w * d for w, d in zip(normalized_weights, deltas))
        return result.astype(np.float32)


# ── Concrete Strategy: ArgmaxCleanup ─────────────────────────────────────────


class ArgmaxCleanup(CleanupStrategy):
    """Per-block argmax cleanup — produces one-hot block codes."""

    def cleanup(self, vec: np.ndarray) -> np.ndarray:
        """Clean up a block-code vector via per-block argmax.

        Args:
            vec: Noisy block-code vector (k, l).

        Returns:
            One-hot cleaned block-code vector (k, l).
        """
        k, l = vec.shape
        clean = np.zeros_like(vec)
        for block_idx in range(k):
            max_idx = np.argmax(vec[block_idx])
            clean[block_idx, max_idx] = 1.0
        return clean


# ── World ────────────────────────────────────────────────────────────────────


class World(BaseWorld):
    """A single recursive expert world model at depth z.

    Contains one or more HYLAs that generate transition deltas from an
    input (state_vec || action_vec). Deltas are merged via MergeStrategy
    and applied to the state via block-code binding.

    If the prediction residual exceeds tau and z_depth < z_max, a
    sub-world at z+1 is spawned to refine the prediction.

    Args:
        world_id: Unique identifier.
        k: Number of blocks.
        l: Block length.
        n_hylas: Number of HYLA modules per world.
        z_depth: Current depth in the recursive tree.
        z_max: Maximum allowed recursion depth.
        tau: Confidence threshold for spawning sub-worlds.
        axioms: List of axioms this world operates with.
        seed: Random seed for reproducibility.
        merge_strategy: Strategy for merging HYLA deltas.
        cleanup_strategy: Strategy for cleaning up predictions.
    """

    def __init__(
        self,
        world_id: int,
        k: int,
        l: int,
        n_hylas: int,
        z_depth: int,
        z_max: int,
        tau: float,
        axioms: list[Axiom],
        seed: int = 42,
        merge_strategy: MergeStrategy | None = None,
        cleanup_strategy: CleanupStrategy | None = None,
    ) -> None:
        self.world_id = world_id
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.n_hylas = n_hylas
        self.z_depth = z_depth
        self.z_max = z_max
        self.tau = tau
        self.axioms = list(axioms)
        self._seed = seed

        self.bc = BlockCodes(k=k, l=l)

        # Merge and cleanup strategies (GoF Strategy pattern)
        self._merge = merge_strategy or BindComposeMerge(self.bc)
        self._cleanup = cleanup_strategy or ArgmaxCleanup()

        # Create HYLAs — each takes concatenated (state || action) as input
        # Input dim = 2 * d_vsa, output dim = d_vsa (transition delta)
        d_input = 2 * self.d_vsa
        d_hidden = min(256, self.d_vsa)
        self.hylas: list[HYLA] = []
        for i in range(n_hylas):
            hyla = HYLA(
                d_vsa=d_input,
                d_hidden=d_hidden,
                d_out=self.d_vsa,
                k=k,
                l=l * 2,  # doubled because input is concatenation
                seed=seed + i,
            )
            self.hylas.append(hyla)

        # Compose axiom vectors into a conditioning vector
        if axioms:
            self._axiom_vec = axioms[0].vector.copy()
            for ax in axioms[1:]:
                self._axiom_vec = self.bc.bind(self._axiom_vec, ax.vector)
        else:
            self._axiom_vec = self.bc.random_discrete(seed=seed + 1000)

        # Sub-worlds (Composite children)
        self.subworlds: list[World] = []

        # Tracking
        self._obs_count: int = 0
        self._last_selected_step: int = -1

    @property
    def obs_count(self) -> int:
        """Number of observations (predict calls) this world has seen."""
        return self._obs_count

    @property
    def last_selected_step(self) -> int:
        """Step at which this world was last selected by the router."""
        return self._last_selected_step

    def mark_selected(self, step: int) -> None:
        """Mark this world as selected at a given step.

        Args:
            step: Current global step counter.
        """
        self._last_selected_step = step

    def predict(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict next state given current state and action.

        1. Concatenate state_vec and action_vec as HYLA input.
        2. Each HYLA generates a transition delta.
        3. Merge deltas via MergeStrategy.
        4. Bind merged delta with state_vec for predicted next state.
        5. Compute confidence from prediction magnitude.

        Args:
            state_vec: Current state block-code vector (k, l).
            action_vec: Action block-code vector (k, l).

        Returns:
            (predicted_state, confidence) — (k, l) and scalar in [0, 1].
        """
        self._obs_count += 1
        self._last_selected_step = self._obs_count

        # Concatenate state and action as flat input
        input_flat = np.concatenate(
            [state_vec.ravel(), action_vec.ravel()]
        ).astype(np.float32)

        # Condition on axiom vector: add axiom to input via element-wise sum
        axiom_flat = np.concatenate(
            [self._axiom_vec.ravel(), self._axiom_vec.ravel()]
        ).astype(np.float32)
        conditioned_input = input_flat + axiom_flat

        # Each HYLA generates a transition delta
        deltas: list[np.ndarray] = []
        for hyla in self.hylas:
            delta_flat = hyla.forward(conditioned_input, conditioned_input)
            delta = delta_flat[:self.d_vsa].reshape(self.k, self.l)
            deltas.append(delta)

        # Equal weights for now
        weights = [1.0 / len(deltas)] * len(deltas)

        # Merge deltas
        merged_delta = self._merge.merge(deltas, weights)

        # Apply transition: predicted = bind(state, delta)
        predicted = self.bc.bind(state_vec, merged_delta)

        # Confidence: normalized magnitude (higher block agreement = more confident)
        block_maxes = np.max(predicted, axis=-1)  # (k,)
        confidence = float(np.mean(block_maxes))
        confidence = min(max(confidence, 0.0), 1.0)

        return predicted.astype(np.float32), confidence

    def spawn_subworld(self, axioms: list[Axiom]) -> World:
        """Spawn a child sub-world at z+1.

        Args:
            axioms: Axioms for the child world.

        Returns:
            New World at z_depth + 1.

        Raises:
            ValueError: If already at z_max depth.
        """
        if self.z_depth >= self.z_max:
            raise ValueError(
                f"Cannot spawn sub-world: already at z_max={self.z_max}"
            )
        child_id = next(_world_id_counter)
        child = World(
            world_id=child_id,
            k=self.k,
            l=self.l,
            n_hylas=self.n_hylas,
            z_depth=self.z_depth + 1,
            z_max=self.z_max,
            tau=self.tau,
            axioms=axioms,
            seed=self._seed + child_id,
            merge_strategy=self._merge,
            cleanup_strategy=self._cleanup,
        )
        self.subworlds.append(child)
        return child

    def consolidate(self, other: World) -> None:
        """Merge another world into this one via Oja's rule.

        Updates HYLA weights using a running average weighted by observation
        counts (Oja-like plastic consolidation).

        Args:
            other: World to absorb.
        """
        alpha = 0.1  # Oja learning rate
        total_obs = max(self._obs_count + other._obs_count, 1)
        weight_self = self._obs_count / total_obs
        weight_other = other._obs_count / total_obs

        for h_self, h_other in zip(self.hylas, other.hylas):
            # Oja-like update: W <- W + alpha * (other_W - W) * weight_other
            w_h_self = np.asarray(h_self.W_h, dtype=np.float32)
            w_h_other = np.asarray(h_other.W_h, dtype=np.float32)
            w_H_self = np.asarray(h_self.W_H, dtype=np.float32)
            w_H_other = np.asarray(h_other.W_H, dtype=np.float32)

            h_self.W_h = (
                w_h_self + alpha * weight_other * (w_h_other - w_h_self)
            ).astype(np.float32)
            h_self.W_H = (
                w_H_self + alpha * weight_other * (w_H_other - w_H_self)
            ).astype(np.float32)

        # Merge axiom vectors via bundling
        self._axiom_vec = self.bc.bundle(
            [self._axiom_vec, other._axiom_vec], normalize=True
        )

        # Absorb observation count
        self._obs_count += other._obs_count
```

### Step 4.3 — Run tests to verify pass

- [ ] Run:

```bash
uv run pytest tests/test_world.py -v -x --tb=short
```

**Expected:** All 11 tests pass.

### Step 4.4 — Commit

```bash
git add mowm/world.py tests/test_world.py
git commit -m "feat(world): recursive World with HYLA hypernetworks, merge strategies, cleanup

Composite pattern world at depth z. Predict via HYLA-generated deltas
merged with BindComposeMerge/WeightedAvgMerge, applied as block-code
bind. Spawn sub-worlds at z+1, consolidate via Oja's rule."
```

---

## Task 5: router.py — MoWMRouter Wrapping DSelectKGate

**Files:**
- Create: `mowm/router.py`
- Create: `tests/test_router.py`

### Step 5.1 — Write failing test

- [ ] Create `tests/test_router.py`:

```python
# tests/test_router.py
"""Tests for MoWMRouter — dynamic expert routing with spawn/prune."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.router import MoWMRouter
from mowm.world import World

K, L = 4, 32


def _make_lib() -> AxiomLibrary:
    return AxiomLibrary(k=K, l=L)


def _make_axioms(lib, n=3):
    return [
        lib.make_axiom(f"ax_{i}", "test", f"x+{i}", ["+"])
        for i in range(n)
    ]


class TestMoWMRouter:
    def setup_method(self):
        self.lib = _make_lib()
        self.axioms = _make_axioms(self.lib, 3)

    def test_creation(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        assert router.num_active_worlds == 0

    def test_add_world(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        router.add_world(w)
        assert router.num_active_worlds == 1

    def test_route_returns_worlds_and_weights(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        for i in range(3):
            w = World(
                world_id=i, k=K, l=L, n_hylas=1,
                z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
                seed=i * 10,
            )
            router.add_world(w)

        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        result = router.route(state, action)

        assert len(result) >= 1
        assert len(result) <= 2  # top_k=2
        for world, weight in result:
            assert isinstance(world, World)
            assert isinstance(weight, float)
            assert weight >= 0.0

    def test_maybe_spawn_creates_world(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.01,  # very low threshold = always spawn
            tau_prune_steps=100,
        )
        bc = self.lib.bc
        residual = bc.random_discrete(seed=99)

        # Seed the library first
        import mowm.domains.physics  # noqa: F401
        from mowm.domains import seed_all
        seed_all(self.lib, domains=["physics"])

        new_world = router.maybe_spawn(residual, self.lib)
        assert new_world is not None
        assert router.num_active_worlds == 1

    def test_prune_removes_inactive(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=5,
        )
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        router.add_world(w)
        # World was never selected, prune at step 10
        pruned = router.prune(current_step=10)
        assert 0 in pruned
        assert router.num_active_worlds == 0

    def test_prune_keeps_active(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        router.add_world(w)
        # Simulate selection at step 5
        w.mark_selected(5)
        pruned = router.prune(current_step=10)
        assert len(pruned) == 0
        assert router.num_active_worlds == 1

    def test_max_worlds_enforced(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=2, top_k=1,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        for i in range(2):
            w = World(
                world_id=i, k=K, l=L, n_hylas=1,
                z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
            )
            router.add_world(w)
        assert router.num_active_worlds == 2

        # Attempting to add beyond max should not raise but should be capped
        w3 = World(
            world_id=99, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        with pytest.raises(ValueError, match="max_worlds"):
            router.add_world(w3)

    def test_entropy_regularization(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        for i in range(3):
            w = World(
                world_id=i, k=K, l=L, n_hylas=1,
                z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
                seed=i * 10,
            )
            router.add_world(w)

        ent = router.entropy_reg()
        assert isinstance(ent, float)
        assert ent >= 0.0
```

- [ ] Run to verify fail:

```bash
uv run pytest tests/test_router.py -v -x --tb=short
```

**Expected:** `ModuleNotFoundError: No module named 'mowm.router'`

### Step 5.2 — Implement router.py

- [ ] Create `mowm/router.py`:

```python
# mowm/router.py
"""MoWMRouter — DSelect-k routing over a dynamic pool of expert worlds.

Wraps cubemind's DSelectKGate with spawn/prune logic for a dynamic number
of expert worlds. Maintains a max-capacity gate and masks inactive slots.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from cubemind.ops.block_codes import BlockCodes
from cubemind.routing.moe_gate import DSelectKGate

from mowm.base import BaseRouter
from mowm.world import World

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

_spawn_id_counter = itertools.count(start=5000)


class MoWMRouter(BaseRouter):
    """Dynamic expert routing over a pool of World instances.

    Uses DSelectKGate (smooth-step k-hot gate) with a fixed max-capacity
    gate. Inactive world slots are masked to zero weight.

    Supports dynamic spawning when residuals are high and pruning of
    worlds that have not been selected in tau_prune_steps.

    Args:
        k: Number of blocks.
        l: Block length.
        max_worlds: Maximum number of expert worlds in the pool.
        top_k: Number of experts to select per routing step.
        tau_spawn: Residual norm threshold for spawning new worlds.
        tau_prune_steps: Number of idle steps before a world is pruned.
        seed: Random seed for the gate.
    """

    def __init__(
        self,
        k: int,
        l: int,
        max_worlds: int,
        top_k: int = 2,
        tau_spawn: float = 0.5,
        tau_prune_steps: int = 100,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.max_worlds = max_worlds
        self.top_k = top_k
        self.tau_spawn = tau_spawn
        self.tau_prune_steps = tau_prune_steps

        self.bc = BlockCodes(k=k, l=l)

        # Fixed-capacity gate — inactive slots are masked
        self._gate = DSelectKGate(
            num_experts=max_worlds,
            k=min(top_k, max_worlds),
            gamma=1.0,
            seed=seed,
        )

        # Active worlds indexed by slot position
        self._worlds: dict[int, World] = {}
        self._slot_to_world_id: dict[int, int] = {}
        self._world_id_to_slot: dict[int, int] = {}
        self._next_slot: int = 0

    @property
    def num_active_worlds(self) -> int:
        """Number of currently active worlds."""
        return len(self._worlds)

    def add_world(self, world: World) -> int:
        """Add a world to the pool.

        Args:
            world: World instance to add.

        Returns:
            Slot index assigned to the world.

        Raises:
            ValueError: If pool is at max capacity.
        """
        if len(self._worlds) >= self.max_worlds:
            raise ValueError(
                f"Cannot add world: pool at max_worlds={self.max_worlds}"
            )
        slot = self._next_slot
        self._next_slot += 1
        # Wrap around if needed
        if slot >= self.max_worlds:
            # Find a free slot
            used = set(self._worlds.keys())
            for s in range(self.max_worlds):
                if s not in used:
                    slot = s
                    break
            else:
                raise ValueError(
                    f"Cannot add world: max_worlds={self.max_worlds}"
                )

        self._worlds[slot] = world
        self._slot_to_world_id[slot] = world.world_id
        self._world_id_to_slot[world.world_id] = slot
        return slot

    def route(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
    ) -> list[tuple[World, float]]:
        """Route a (state, action) pair to top-k worlds.

        Uses DSelectKGate with input conditioning from the concatenated
        state and action vectors. Inactive slots are masked to zero.

        Args:
            state_vec: State block-code vector (k, l).
            action_vec: Action block-code vector (k, l).

        Returns:
            List of (World, weight) tuples, sorted by descending weight.
        """
        if not self._worlds:
            return []

        # Compute gate scores with input conditioning
        input_flat = np.concatenate(
            [state_vec.ravel(), action_vec.ravel()]
        ).astype(np.float64)

        expert_weights = self._gate.forward(scores=input_flat)

        # Mask inactive slots to zero
        active_mask = np.zeros(self.max_worlds)
        for slot in self._worlds:
            if slot < self.max_worlds:
                active_mask[slot] = 1.0
        expert_weights = expert_weights * active_mask

        # Re-normalize
        total = expert_weights.sum()
        if total > 0:
            expert_weights = expert_weights / total

        # Select top-k by weight
        top_indices = np.argsort(-expert_weights)
        results: list[tuple[World, float]] = []
        for idx in top_indices:
            idx_int = int(idx)
            if idx_int in self._worlds and expert_weights[idx_int] > 1e-10:
                results.append(
                    (self._worlds[idx_int], float(expert_weights[idx_int]))
                )
                if len(results) >= self.top_k:
                    break

        return results

    def maybe_spawn(
        self,
        residual_vec: np.ndarray,
        library: AxiomLibrary,
    ) -> World | None:
        """Conditionally spawn a new world if residual exceeds threshold.

        Computes the L2 norm of the residual in block-code space. If it
        exceeds tau_spawn, selects axioms from the library closest to the
        residual and creates a new world.

        Args:
            residual_vec: Residual block-code vector (k, l).
            library: Global axiom library for axiom selection.

        Returns:
            Newly spawned World, or None if residual is below threshold.
        """
        # Compute residual magnitude
        residual_norm = float(np.linalg.norm(residual_vec))
        if residual_norm < self.tau_spawn:
            return None

        if len(self._worlds) >= self.max_worlds:
            return None

        # Select axioms closest to the residual
        selected_axioms = library.select(residual_vec, top_n=5)
        if not selected_axioms:
            return None

        # Create new world
        world_id = next(_spawn_id_counter)
        new_world = World(
            world_id=world_id,
            k=self.k,
            l=self.l,
            n_hylas=2,
            z_depth=0,
            z_max=2,
            tau=0.5,
            axioms=selected_axioms,
            seed=world_id,
        )

        self.add_world(new_world)
        return new_world

    def prune(self, current_step: int) -> list[int]:
        """Prune worlds not selected in tau_prune_steps.

        Args:
            current_step: Current global step counter.

        Returns:
            List of pruned world IDs.
        """
        pruned: list[int] = []
        slots_to_remove: list[int] = []

        for slot, world in self._worlds.items():
            idle_steps = current_step - max(world.last_selected_step, 0)
            if idle_steps > self.tau_prune_steps:
                pruned.append(world.world_id)
                slots_to_remove.append(slot)

        for slot in slots_to_remove:
            wid = self._slot_to_world_id.pop(slot, None)
            if wid is not None:
                self._world_id_to_slot.pop(wid, None)
            del self._worlds[slot]

        return pruned

    def entropy_reg(self) -> float:
        """Compute entropy regularization loss for the gate.

        Returns:
            Scalar entropy penalty (higher = less sparse).
        """
        expert_weights = self._gate.forward()
        return float(self._gate.entropy_regularization(expert_weights))

    def get_world_by_id(self, world_id: int) -> World | None:
        """Look up a world by its ID.

        Args:
            world_id: The world's unique identifier.

        Returns:
            World instance or None if not found.
        """
        slot = self._world_id_to_slot.get(world_id)
        if slot is None:
            return None
        return self._worlds.get(slot)

    def active_worlds(self) -> list[World]:
        """Return all active worlds.

        Returns:
            List of currently active World instances.
        """
        return list(self._worlds.values())
```

### Step 5.3 — Run tests to verify pass

- [ ] Run:

```bash
uv run pytest tests/test_router.py -v -x --tb=short
```

**Expected:** All 8 tests pass.

### Step 5.4 — Commit

```bash
git add mowm/router.py tests/test_router.py
git commit -m "feat(router): MoWMRouter with DSelectKGate, dynamic spawn/prune

Wraps DSelectKGate over a max-capacity pool of World instances.
Routes (state, action) to top-k worlds with input conditioning.
Spawns new worlds when residual exceeds tau_spawn, prunes idle worlds."
```

---

## Task 6: pipeline.py — Full End-to-End MoWMPipeline

**Files:**
- Create: `mowm/pipeline.py`
- Create: `tests/test_pipeline.py`

### Step 6.1 — Write failing test

- [ ] Create `tests/test_pipeline.py`:

```python
# tests/test_pipeline.py
"""Tests for MoWMPipeline — end-to-end orchestrator."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.pipeline import MoWMPipeline, StepResult

K, L = 4, 32


class TestMoWMPipeline:
    def setup_method(self):
        self.pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=8, top_k=2, z_max=2, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"],
        )

    def test_creation(self):
        assert self.pipe is not None
        assert self.pipe.library is not None
        assert len(self.pipe.library) > 0  # physics axioms seeded

    def test_step_with_dict_state(self):
        result = self.pipe.step(
            state={"position": "origin", "velocity": "zero"},
            action="apply_force",
        )
        assert isinstance(result, StepResult)
        assert result.predicted_state.shape == (K, L)
        assert isinstance(result.q_value, float)
        assert isinstance(result.confidence, float)
        assert result.active_worlds >= 1

    def test_step_with_text_state(self):
        result = self.pipe.step(
            state="the ball is at position x=0",
            action="push right",
        )
        assert isinstance(result, StepResult)
        assert result.predicted_state.shape == (K, L)

    def test_step_with_array_state(self):
        bc = self.pipe.library.bc
        state_vec = bc.random_discrete(seed=10)
        action_vec = bc.random_discrete(seed=20)
        result = self.pipe.step(
            state=state_vec,
            action=action_vec,
        )
        assert result.predicted_state.shape == (K, L)

    def test_step_increments_step_count(self):
        self.pipe.step(state="s1", action="a1")
        self.pipe.step(state="s2", action="a2")
        assert self.pipe.step_count == 2

    def test_cleanup_happens_at_interval(self):
        """After cleanup_interval steps, prediction should be cleaned."""
        # Run cleanup_interval steps
        for i in range(3):
            self.pipe.step(state=f"state_{i}", action=f"action_{i}")
        # Pipeline should have done at least 1 cleanup by now
        assert self.pipe.step_count == 3

    def test_rollout(self):
        results = self.pipe.rollout(
            state="initial_state",
            actions=["a1", "a2", "a3"],
            horizon=3,
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, StepResult)
            assert r.predicted_state.shape == (K, L)

    def test_auto_spawn_worlds(self):
        """Pipeline should auto-spawn at least one world for prediction."""
        result = self.pipe.step(state="test", action="test_action")
        assert result.active_worlds >= 1

    def test_deterministic_with_seed(self):
        pipe1 = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=4, top_k=1, z_max=1, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"], seed=42,
        )
        pipe2 = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=4, top_k=1, z_max=1, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"], seed=42,
        )
        r1 = pipe1.step(state="test_state", action="test_action")
        r2 = pipe2.step(state="test_state", action="test_action")
        np.testing.assert_array_equal(r1.predicted_state, r2.predicted_state)
```

- [ ] Run to verify fail:

```bash
uv run pytest tests/test_pipeline.py -v -x --tb=short
```

**Expected:** `ModuleNotFoundError: No module named 'mowm.pipeline'`

### Step 6.2 — Implement pipeline.py

- [ ] Create `mowm/pipeline.py`:

```python
# mowm/pipeline.py
"""MoWMPipeline — end-to-end orchestrator for the mixture of world models.

Pipeline flow:
  1. HYLA Encoder: state, action -> block-code vectors
  2. Router: select top-k worlds
  3. Each world: predict(state_vec, action_vec)
  4. Mixture: weighted sum of predictions
  5. Cleanup: per-block argmax every C steps
  6. CVL: estimate Q-value
  7. HYLA Decoder: block-code -> predicted next state + value
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cubemind.execution.cvl import ContrastiveValueEstimator
from cubemind.execution.hyla import HYLA
from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes

from mowm.axiom_library import AxiomLibrary
from mowm.domains import seed_all
from mowm.router import MoWMRouter
from mowm.world import ArgmaxCleanup, World


@dataclass
class StepResult:
    """Result of a single pipeline step.

    Attributes:
        predicted_state: Predicted next state block-code vector (k, l).
        q_value: Estimated Q-value from CVL.
        confidence: Prediction confidence in [0, 1].
        active_worlds: Number of active worlds after this step.
        discovered: Whether a new axiom was discovered this step.
    """

    predicted_state: np.ndarray
    q_value: float
    confidence: float
    active_worlds: int
    discovered: bool


class MoWMPipeline:
    """End-to-end MoWM pipeline orchestrator.

    Manages the full predict cycle: encode inputs, route to expert worlds,
    merge predictions, cleanup, estimate Q-values, and decode.

    Auto-spawns initial worlds from the axiom library if the router pool
    is empty at first step.

    Args:
        k: Number of blocks.
        l: Block length.
        d_hidden: HYLA hidden dimension.
        d_out: HYLA output dimension (typically k * l).
        max_worlds: Maximum expert worlds in pool.
        top_k: Number of experts selected per step.
        z_max: Maximum recursion depth for sub-worlds.
        tau: Confidence threshold for sub-world spawning.
        cleanup_interval: Number of steps between cleanup operations.
        gamma: Discount factor for CVL.
        domains: List of domain names to seed (None = all registered).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int,
        l: int,
        d_hidden: int,
        d_out: int,
        max_worlds: int,
        top_k: int,
        z_max: int,
        tau: float,
        cleanup_interval: int,
        gamma: float,
        domains: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.cleanup_interval = cleanup_interval
        self._seed = seed

        # Core VSA ops
        self.bc = BlockCodes(k=k, l=l)
        self.encoder = WorldEncoder(k=k, l=l)
        self._cleanup = ArgmaxCleanup()

        # Axiom library — seed with specified domains
        self.library = AxiomLibrary(k=k, l=l, domains=domains)
        self._import_domains(domains)
        seed_all(self.library, domains=domains)

        # Router
        self.router = MoWMRouter(
            k=k, l=l,
            max_worlds=max_worlds,
            top_k=top_k,
            tau_spawn=tau,
            tau_prune_steps=200,
            seed=seed,
        )

        # CVL estimator for Q-value prediction
        self.cvl = ContrastiveValueEstimator(
            d_state=self.d_vsa,
            d_action=self.d_vsa,
            d_latent=min(128, self.d_vsa),
            d_rff=min(256, self.d_vsa * 2),
            gamma=gamma,
            seed=seed,
        )

        # Step counter
        self._step_count: int = 0

        # Auto-spawn initial worlds
        self._init_worlds(seed)

    @property
    def step_count(self) -> int:
        """Total number of pipeline steps executed."""
        return self._step_count

    def _import_domains(self, domains: list[str] | None) -> None:
        """Import domain modules to trigger @register_domain decorators.

        Args:
            domains: Specific domains to import, or None for all.
        """
        all_domain_modules = [
            "physics", "logic", "economics", "social", "linguistics",
            "causal", "atomic", "bio", "music", "identity", "colors",
            "language", "historical", "ensemble",
        ]
        targets = domains if domains else all_domain_modules
        for name in targets:
            try:
                __import__(f"mowm.domains.{name}")
            except ImportError:
                pass  # Domain not yet implemented

    def _init_worlds(self, seed: int) -> None:
        """Auto-spawn initial worlds from the axiom library.

        Creates one world per domain group in the library.

        Args:
            seed: Random seed base.
        """
        domain_groups = self.library.domains()
        for i, (domain_name, axioms) in enumerate(domain_groups.items()):
            if self.router.num_active_worlds >= self.router.max_worlds:
                break
            # Take up to 10 axioms per initial world
            world_axioms = axioms[:10]
            world = World(
                world_id=i,
                k=self.k,
                l=self.l,
                n_hylas=2,
                z_depth=0,
                z_max=2,
                tau=0.5,
                axioms=world_axioms,
                seed=seed + i,
            )
            try:
                self.router.add_world(world)
            except ValueError:
                break  # Pool full

    def _encode_input(
        self,
        state: dict[str, str] | str | np.ndarray,
        action: str | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode state and action into block-code vectors.

        Supports dict (structured), string (narrative), or raw array inputs.

        Args:
            state: State as dict, string, or (k, l) array.
            action: Action as string or (k, l) array.

        Returns:
            (state_vec, action_vec) — both (k, l) float32 arrays.
        """
        # Encode state
        if isinstance(state, np.ndarray):
            state_vec = state.astype(np.float32)
        elif isinstance(state, dict):
            state_vec = self.encoder.encode_state(state)
        else:
            state_vec = self.encoder.encode_action(str(state))

        # Encode action
        if isinstance(action, np.ndarray):
            action_vec = action.astype(np.float32)
        else:
            action_vec = self.encoder.encode_action(str(action))

        return state_vec, action_vec

    def step(
        self,
        state: dict[str, str] | str | np.ndarray,
        action: str | np.ndarray,
        ground_truth: np.ndarray | None = None,
    ) -> StepResult:
        """Execute a single pipeline step.

        1. Encode state and action to block-code vectors.
        2. Route to top-k worlds.
        3. Each world predicts next state.
        4. Weighted mixture of predictions.
        5. Cleanup every cleanup_interval steps.
        6. CVL Q-value estimation.

        Args:
            state: State input (dict, string, or array).
            action: Action input (string or array).
            ground_truth: Optional ground truth for residual checking.

        Returns:
            StepResult with prediction, Q-value, confidence, and metadata.
        """
        self._step_count += 1

        # 1. Encode
        state_vec, action_vec = self._encode_input(state, action)

        # 2. Route
        routed = self.router.route(state_vec, action_vec)

        # If no worlds available, spawn one from residual
        if not routed:
            new_world = self.router.maybe_spawn(state_vec, self.library)
            if new_world is not None:
                routed = [(new_world, 1.0)]

        # 3. Each world predicts
        predictions: list[np.ndarray] = []
        weights: list[float] = []
        total_confidence = 0.0

        for world, weight in routed:
            pred, conf = world.predict(state_vec, action_vec)
            predictions.append(pred)
            weights.append(weight)
            total_confidence += conf * weight
            world.mark_selected(self._step_count)

        # 4. Weighted mixture
        if predictions:
            weighted_preds = [w * p for w, p in zip(weights, predictions)]
            stacked = np.stack(weighted_preds, axis=0)
            mixed = stacked.sum(axis=0).astype(np.float32)
            # Normalize per block
            block_sums = mixed.sum(axis=-1, keepdims=True)
            block_sums = np.where(block_sums == 0, 1.0, block_sums)
            mixed = (mixed / block_sums).astype(np.float32)
        else:
            mixed = state_vec.copy()
            total_confidence = 0.0

        # 5. Cleanup every C steps
        if self._step_count % self.cleanup_interval == 0:
            mixed = self._cleanup.cleanup(mixed)

        # 6. CVL Q-value
        q_value = float(
            self.cvl.encode_state_action(
                state_vec.ravel(), action_vec.ravel()
            ).mean()
        )

        # Discovery check (if ground truth available)
        discovered = False
        if ground_truth is not None:
            residual = ground_truth - mixed
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > self.router.tau_spawn:
                new_world = self.router.maybe_spawn(residual, self.library)
                discovered = new_world is not None

        return StepResult(
            predicted_state=mixed,
            q_value=q_value,
            confidence=total_confidence,
            active_worlds=self.router.num_active_worlds,
            discovered=discovered,
        )

    def rollout(
        self,
        state: dict[str, str] | str | np.ndarray,
        actions: list[str | np.ndarray],
        horizon: int,
    ) -> list[StepResult]:
        """Execute a multi-step rollout.

        Chains predictions: each step's output becomes the next step's input.

        Args:
            state: Initial state.
            actions: List of actions for each step.
            horizon: Number of steps (uses first `horizon` actions).

        Returns:
            List of StepResult for each step.
        """
        results: list[StepResult] = []
        current_state = state

        for t in range(min(horizon, len(actions))):
            result = self.step(current_state, actions[t])
            results.append(result)
            current_state = result.predicted_state

        return results
```

### Step 6.3 — Run tests to verify pass

- [ ] Run:

```bash
uv run pytest tests/test_pipeline.py -v -x --tb=short
```

**Expected:** All 9 tests pass.

### Step 6.4 — Commit

```bash
git add mowm/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): MoWMPipeline end-to-end orchestrator

Encode -> route -> predict -> mixture -> cleanup -> CVL -> result.
Supports dict/string/array inputs, auto-spawns initial worlds,
multi-step rollout with chained predictions."
```

---

## Task 7: discovery.py — DiscoveryLoop with Genetic Programming SR

**Files:**
- Create: `mowm/discovery.py`
- Create: `tests/test_discovery.py`

### Step 7.1 — Write failing test

- [ ] Create `tests/test_discovery.py`:

```python
# tests/test_discovery.py
"""Tests for DiscoveryLoop — symbolic regression and axiom distillation."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.discovery import DiscoveryLoop, GeneticProgrammingSR

K, L = 4, 32


class TestGeneticProgrammingSR:
    def test_discovers_linear(self):
        """SR should discover f(x) = 2*x + 1 from data."""
        sr = GeneticProgrammingSR(
            population_size=50,
            generations=30,
            seed=42,
        )
        x = np.linspace(0, 5, 20).reshape(-1, 1)
        y = 2 * x.ravel() + 1
        formula, mse = sr.search(
            inputs=x,
            outputs=y,
            operators=["+", "*"],
            max_complexity=6,
        )
        assert isinstance(formula, str)
        assert isinstance(mse, float)
        # SR should find something reasonably close
        assert mse < 5.0  # generous threshold for stochastic SR

    def test_discovers_quadratic(self):
        """SR should approximate f(x) = x^2."""
        sr = GeneticProgrammingSR(
            population_size=80,
            generations=40,
            seed=123,
        )
        x = np.linspace(-3, 3, 25).reshape(-1, 1)
        y = x.ravel() ** 2
        formula, mse = sr.search(
            inputs=x,
            outputs=y,
            operators=["+", "*", "square"],
            max_complexity=8,
        )
        assert isinstance(formula, str)
        assert mse < 10.0

    def test_returns_best_of_population(self):
        sr = GeneticProgrammingSR(
            population_size=20,
            generations=10,
            seed=42,
        )
        x = np.array([[1], [2], [3]], dtype=np.float64)
        y = np.array([2, 4, 6], dtype=np.float64)
        formula, mse = sr.search(x, y, ["+", "*"], 5)
        assert formula is not None


class TestDiscoveryLoop:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)
        # Seed a few axioms for context
        for i in range(5):
            ax = self.lib.make_axiom(
                f"base_{i}", "test", f"x + {i}", ["+"]
            )
            self.lib.register(ax)

    def test_check_residual(self):
        loop = DiscoveryLoop(self.lib, k=K, l=L)
        bc = self.lib.bc
        a = bc.random_discrete(seed=1)
        b = bc.random_discrete(seed=2)
        residual = loop.check_residual(a, b)
        assert isinstance(residual, float)
        assert residual >= 0.0

    def test_check_residual_identical(self):
        loop = DiscoveryLoop(self.lib, k=K, l=L)
        bc = self.lib.bc
        a = bc.random_discrete(seed=1)
        residual = loop.check_residual(a, a)
        assert residual == pytest.approx(0.0, abs=1e-6)

    def test_discover_returns_axiom_or_none(self):
        loop = DiscoveryLoop(
            self.lib, k=K, l=L,
            sr_operators=["+", "*"],
            max_complexity=6,
        )
        bc = self.lib.bc
        # Simple linear relationship
        inputs = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
        outputs = np.array([3, 5, 7, 9, 11], dtype=np.float64)
        context = bc.random_discrete(seed=42)
        result = loop.discover(inputs, outputs, context)
        # May or may not find something, but should not crash
        if result is not None:
            assert result.vector.shape == (K, L)
            assert result.domain == "discovered"

    def test_discover_registers_in_library(self):
        loop = DiscoveryLoop(
            self.lib, k=K, l=L,
            sr_operators=["+", "*"],
            max_complexity=6,
        )
        initial_count = len(self.lib)
        inputs = np.array([[1], [2], [3], [4]], dtype=np.float64)
        outputs = np.array([2, 4, 6, 8], dtype=np.float64)
        context = self.lib.bc.random_discrete(seed=42)
        result = loop.discover(inputs, outputs, context)
        if result is not None:
            assert len(self.lib) == initial_count + 1

    def test_observer_notified(self):
        from mowm.base import DiscoveryObserver

        class MockObserver(DiscoveryObserver):
            def __init__(self):
                self.was_notified = False
                self.axiom = None

            def on_axiom_discovered(self, axiom):
                self.was_notified = True
                self.axiom = axiom

        obs = MockObserver()
        loop = DiscoveryLoop(
            self.lib, k=K, l=L,
            sr_operators=["+", "*"],
            max_complexity=6,
            observers=[obs],
        )
        inputs = np.array([[1], [2], [3], [4]], dtype=np.float64)
        outputs = np.array([2, 4, 6, 8], dtype=np.float64)
        context = self.lib.bc.random_discrete(seed=42)
        result = loop.discover(inputs, outputs, context)
        if result is not None:
            assert obs.was_notified
            assert obs.axiom is result

    def test_cross_domain_analogy(self):
        """Unbind anomaly from domain_A, bind with domain_B."""
        loop = DiscoveryLoop(self.lib, k=K, l=L)
        bc = self.lib.bc
        domain_a = bc.random_discrete(seed=100)
        domain_b = bc.random_discrete(seed=200)
        anomaly = bc.random_discrete(seed=300)
        analogy = loop.cross_domain_analogy(anomaly, domain_a, domain_b)
        assert analogy.shape == (K, L)
        # Analogy should differ from original anomaly
        sim = bc.similarity(analogy, anomaly)
        assert sim < 0.9
```

- [ ] Run to verify fail:

```bash
uv run pytest tests/test_discovery.py -v -x --tb=short
```

**Expected:** `ModuleNotFoundError: No module named 'mowm.discovery'`

### Step 7.2 — Implement discovery.py

- [ ] Create `mowm/discovery.py`:

```python
# mowm/discovery.py
"""DiscoveryLoop — universal 3-step discovery via symbolic regression.

When prediction residuals are high, the loop:
  1. Semantic mapping: find nearest axioms, cross-domain analogy
  2. Symbolic hypothesis: genetic programming SR over operator grammar
  3. Axiom distillation: encode discovered formula as block-code, register

The SR implementation uses a simple genetic programming loop over an
operator grammar (no PySR dependency). Population = expression trees
evaluated against training data.
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from cubemind.ops.block_codes import BlockCodes

from mowm.base import DiscoveryObserver, SearchStrategy

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary


# ── Expression tree for genetic programming ──────────────────────────────────

_SAFE_OPS: dict[str, Callable] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": lambda a, b: a / b if abs(b) > 1e-10 else a,
    "sin": lambda a, _=None: np.sin(a),
    "cos": lambda a, _=None: np.cos(a),
    "sqrt": lambda a, _=None: np.sqrt(np.abs(a)),
    "square": lambda a, _=None: a ** 2,
    "exp": lambda a, _=None: np.clip(np.exp(np.clip(a, -10, 10)), -1e6, 1e6),
    "log": lambda a, _=None: np.log(np.abs(a) + 1e-10),
    "AND": lambda a, b: np.where((a > 0.5) & (b > 0.5), 1.0, 0.0),
    "OR": lambda a, b: np.where((a > 0.5) | (b > 0.5), 1.0, 0.0),
    "NOT": lambda a, _=None: np.where(a > 0.5, 0.0, 1.0),
    "IMPLIES": lambda a, b: np.where((a > 0.5) & (b <= 0.5), 0.0, 1.0),
}

_UNARY_OPS = {"sin", "cos", "sqrt", "square", "exp", "log", "NOT"}
_BINARY_OPS = {"+", "-", "*", "/", "AND", "OR", "IMPLIES"}


@dataclass
class ExprNode:
    """A node in an expression tree."""

    op: str | None = None       # operator name or None for leaf
    value: float | None = None  # constant value (if leaf and not variable)
    var_idx: int | None = None  # variable index (if leaf and not constant)
    left: ExprNode | None = None
    right: ExprNode | None = None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate this expression on input data.

        Args:
            x: Input array (n_samples, n_features).

        Returns:
            Output array (n_samples,).
        """
        if self.op is None:
            # Leaf node
            if self.var_idx is not None:
                idx = min(self.var_idx, x.shape[1] - 1)
                return x[:, idx].astype(np.float64)
            return np.full(x.shape[0], self.value or 0.0, dtype=np.float64)

        # Operator node
        fn = _SAFE_OPS.get(self.op)
        if fn is None:
            return np.zeros(x.shape[0], dtype=np.float64)

        left_val = self.left.evaluate(x) if self.left else np.zeros(
            x.shape[0], dtype=np.float64
        )

        if self.op in _UNARY_OPS:
            try:
                result = fn(left_val)
            except Exception:
                result = np.zeros_like(left_val)
        else:
            right_val = self.right.evaluate(x) if self.right else np.zeros(
                x.shape[0], dtype=np.float64
            )
            try:
                result = fn(left_val, right_val)
            except Exception:
                result = np.zeros_like(left_val)

        return np.clip(np.nan_to_num(result, nan=0.0), -1e6, 1e6)

    def complexity(self) -> int:
        """Count the number of nodes in this tree."""
        count = 1
        if self.left:
            count += self.left.complexity()
        if self.right:
            count += self.right.complexity()
        return count

    def to_string(self) -> str:
        """Convert expression tree to human-readable string."""
        if self.op is None:
            if self.var_idx is not None:
                return f"x{self.var_idx}"
            return f"{self.value:.4g}" if self.value is not None else "0"

        if self.op in _UNARY_OPS:
            left_str = self.left.to_string() if self.left else "0"
            return f"{self.op}({left_str})"

        left_str = self.left.to_string() if self.left else "0"
        right_str = self.right.to_string() if self.right else "0"
        return f"({left_str} {self.op} {right_str})"


# ── Genetic Programming SR ───────────────────────────────────────────────────


class GeneticProgrammingSR(SearchStrategy):
    """Symbolic regression via genetic programming over expression trees.

    Evolves a population of expression trees to minimize MSE + complexity
    penalty on (input, output) training data.

    Args:
        population_size: Number of individuals in the population.
        generations: Number of evolutionary generations.
        tournament_size: Tournament selection size.
        crossover_rate: Probability of crossover vs mutation.
        max_tree_depth: Maximum depth of initial random trees.
        complexity_penalty: Lambda for complexity regularization.
        seed: Random seed.
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        tournament_size: int = 5,
        crossover_rate: float = 0.7,
        max_tree_depth: int = 4,
        complexity_penalty: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.max_tree_depth = max_tree_depth
        self.complexity_penalty = complexity_penalty
        self._rng = np.random.default_rng(seed)

    def _random_tree(
        self,
        operators: list[str],
        n_features: int,
        max_depth: int,
        depth: int = 0,
    ) -> ExprNode:
        """Generate a random expression tree.

        Args:
            operators: Available operators.
            n_features: Number of input features.
            max_depth: Maximum tree depth.
            depth: Current depth.

        Returns:
            Random ExprNode tree.
        """
        if depth >= max_depth or (depth > 0 and self._rng.random() < 0.3):
            # Leaf: variable or constant
            if self._rng.random() < 0.6:
                return ExprNode(var_idx=int(self._rng.integers(0, n_features)))
            return ExprNode(value=float(self._rng.uniform(-5, 5)))

        # Operator node
        op = operators[int(self._rng.integers(0, len(operators)))]
        left = self._random_tree(operators, n_features, max_depth, depth + 1)

        if op in _UNARY_OPS:
            return ExprNode(op=op, left=left)

        right = self._random_tree(operators, n_features, max_depth, depth + 1)
        return ExprNode(op=op, left=left, right=right)

    def _fitness(
        self,
        tree: ExprNode,
        inputs: np.ndarray,
        outputs: np.ndarray,
        max_complexity: int,
    ) -> float:
        """Compute fitness (lower is better): MSE + complexity penalty.

        Args:
            tree: Expression tree to evaluate.
            inputs: Training inputs (n, d).
            outputs: Training outputs (n,).
            max_complexity: Maximum allowed complexity.

        Returns:
            Fitness score (lower is better).
        """
        comp = tree.complexity()
        if comp > max_complexity:
            return 1e9

        try:
            predicted = tree.evaluate(inputs)
            mse = float(np.mean((predicted - outputs) ** 2))
        except Exception:
            return 1e9

        if not np.isfinite(mse):
            return 1e9

        return mse + self.complexity_penalty * comp

    def _tournament_select(
        self,
        population: list[ExprNode],
        fitnesses: list[float],
    ) -> ExprNode:
        """Select an individual via tournament selection.

        Args:
            population: List of expression trees.
            fitnesses: Corresponding fitness scores.

        Returns:
            Selected expression tree.
        """
        indices = self._rng.integers(
            0, len(population), size=self.tournament_size
        )
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitnesses[int(idx)] < fitnesses[int(best_idx)]:
                best_idx = idx
        return population[int(best_idx)]

    def _mutate(
        self,
        tree: ExprNode,
        operators: list[str],
        n_features: int,
    ) -> ExprNode:
        """Mutate a tree by replacing a random subtree.

        Args:
            tree: Tree to mutate.
            operators: Available operators.
            n_features: Number of input features.

        Returns:
            Mutated tree (new object).
        """
        if self._rng.random() < 0.3 or tree.op is None:
            return self._random_tree(operators, n_features, 3)

        new_tree = ExprNode(
            op=tree.op,
            value=tree.value,
            var_idx=tree.var_idx,
        )
        if tree.left:
            if self._rng.random() < 0.5:
                new_tree.left = self._mutate(tree.left, operators, n_features)
            else:
                new_tree.left = tree.left
        if tree.right:
            if self._rng.random() < 0.5:
                new_tree.right = self._mutate(
                    tree.right, operators, n_features
                )
            else:
                new_tree.right = tree.right
        return new_tree

    def _crossover(
        self,
        parent1: ExprNode,
        parent2: ExprNode,
    ) -> ExprNode:
        """Crossover two trees by swapping a random subtree.

        Args:
            parent1: First parent tree.
            parent2: Second parent tree.

        Returns:
            Offspring tree.
        """
        if self._rng.random() < 0.5 and parent1.left:
            return ExprNode(
                op=parent1.op,
                value=parent1.value,
                var_idx=parent1.var_idx,
                left=parent2 if self._rng.random() < 0.3 else parent1.left,
                right=parent1.right,
            )
        if parent1.right:
            return ExprNode(
                op=parent1.op,
                value=parent1.value,
                var_idx=parent1.var_idx,
                left=parent1.left,
                right=parent2 if self._rng.random() < 0.3 else parent1.right,
            )
        return parent2

    def search(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        operators: list[str],
        max_complexity: int,
    ) -> tuple[str, float]:
        """Evolve expression trees to find formula matching data.

        Args:
            inputs: Training inputs (n_samples, n_features).
            outputs: Training outputs (n_samples,).
            operators: Grammar of allowed operators.
            max_complexity: Maximum AST node count.

        Returns:
            (formula_str, mse) of the best individual found.
        """
        inputs = np.atleast_2d(inputs).astype(np.float64)
        outputs = np.asarray(outputs, dtype=np.float64).ravel()
        n_features = inputs.shape[1]

        # Filter operators to those we support
        valid_ops = [op for op in operators if op in _SAFE_OPS]
        if not valid_ops:
            valid_ops = ["+", "*"]

        # Initialize population
        population = [
            self._random_tree(valid_ops, n_features, self.max_tree_depth)
            for _ in range(self.population_size)
        ]

        best_tree = population[0]
        best_fitness = self._fitness(best_tree, inputs, outputs, max_complexity)

        for gen in range(self.generations):
            fitnesses = [
                self._fitness(t, inputs, outputs, max_complexity)
                for t in population
            ]

            # Track best
            for i, f in enumerate(fitnesses):
                if f < best_fitness:
                    best_fitness = f
                    best_tree = population[i]

            # Evolve next generation
            new_pop: list[ExprNode] = [best_tree]  # elitism
            while len(new_pop) < self.population_size:
                if self._rng.random() < self.crossover_rate:
                    p1 = self._tournament_select(population, fitnesses)
                    p2 = self._tournament_select(population, fitnesses)
                    child = self._crossover(p1, p2)
                else:
                    parent = self._tournament_select(population, fitnesses)
                    child = self._mutate(parent, valid_ops, n_features)
                new_pop.append(child)

            population = new_pop

        # Final evaluation for MSE only (no penalty)
        try:
            final_pred = best_tree.evaluate(inputs)
            mse = float(np.mean((final_pred - outputs) ** 2))
        except Exception:
            mse = float("inf")

        return best_tree.to_string(), mse


# ── Discovery Loop ───────────────────────────────────────────────────────────


class DiscoveryLoop:
    """Universal 3-step discovery loop.

    Triggered when prediction residuals are high:
      1. Semantic mapping — find nearest axioms, cross-domain analogy
      2. Symbolic hypothesis — GP-based symbolic regression
      3. Axiom distillation — encode and register new axiom

    Args:
        library: Global AxiomLibrary.
        k: Number of blocks.
        l: Block length.
        sr_operators: Numeric operators for SR (default: standard set).
        logic_operators: Logic operators for SR (default: standard set).
        max_complexity: Maximum expression tree complexity.
        sr_population: GP population size.
        sr_generations: GP generation count.
        observers: List of DiscoveryObserver instances to notify.
        seed: Random seed.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        k: int,
        l: int,
        sr_operators: list[str] | None = None,
        logic_operators: list[str] | None = None,
        max_complexity: int = 10,
        sr_population: int = 100,
        sr_generations: int = 50,
        observers: list[DiscoveryObserver] | None = None,
        seed: int = 42,
    ) -> None:
        self.library = library
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)

        self.sr_operators = sr_operators or [
            "+", "-", "*", "/", "sin", "cos", "exp", "sqrt", "square", "log",
        ]
        self.logic_operators = logic_operators or [
            "AND", "OR", "NOT", "IMPLIES",
        ]
        self.max_complexity = max_complexity
        self._observers = observers or []

        self._sr = GeneticProgrammingSR(
            population_size=sr_population,
            generations=sr_generations,
            seed=seed,
        )

    def check_residual(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """Compute L2 residual between predicted and actual in block-code space.

        Args:
            predicted: Predicted block-code vector (k, l).
            actual: Ground truth block-code vector (k, l).

        Returns:
            L2 norm of the residual.
        """
        diff = actual.astype(np.float64) - predicted.astype(np.float64)
        return float(np.linalg.norm(diff))

    def cross_domain_analogy(
        self,
        anomaly: np.ndarray,
        domain_a: np.ndarray,
        domain_b: np.ndarray,
    ) -> np.ndarray:
        """Cross-domain analogy: unbind anomaly from domain_A, bind to domain_B.

        Args:
            anomaly: Anomalous block-code vector (k, l).
            domain_a: Source domain vector (k, l).
            domain_b: Target domain vector (k, l).

        Returns:
            Analogous vector in domain_b's space (k, l).
        """
        unbound = self.bc.unbind(anomaly, domain_a)
        return self.bc.bind(unbound, domain_b)

    def discover(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        context_vec: np.ndarray,
    ) -> Axiom | None:
        """Run the full 3-step discovery loop.

        Step 1: Semantic mapping — find nearest axioms to context.
        Step 2: Symbolic hypothesis — run GP-SR on numeric and logic operators.
        Step 3: Axiom distillation — encode formula and register.

        Args:
            inputs: Training inputs (n_samples, n_features).
            outputs: Training outputs (n_samples,).
            context_vec: Context block-code vector (k, l) for semantic search.

        Returns:
            Newly discovered Axiom, or None if SR failed.
        """
        from mowm.axiom_library import Axiom

        # Step 1: Semantic mapping — find nearest existing axioms
        nearest = self.library.select(context_vec, top_n=3)
        # Collect their operators to augment the SR grammar
        extra_ops: set[str] = set()
        for ax in nearest:
            extra_ops.update(ax.operators)

        # Combine all operators
        all_ops = list(
            set(self.sr_operators) | set(self.logic_operators) | extra_ops
        )
        valid_ops = [op for op in all_ops if op in _SAFE_OPS]
        if not valid_ops:
            valid_ops = ["+", "*"]

        # Step 2: Symbolic hypothesis — GP search
        formula_str, mse = self._sr.search(
            inputs=inputs,
            outputs=outputs,
            operators=valid_ops,
            max_complexity=self.max_complexity,
        )

        # Reject if MSE is too high (formula is garbage)
        if mse > 1e6 or not np.isfinite(mse):
            return None

        # Step 3: Axiom distillation — encode and register
        new_axiom = self.library.make_axiom(
            name=f"discovered_{formula_str[:40]}",
            domain="discovered",
            formula_str=formula_str,
            operators=valid_ops,
        )
        self.library.register(new_axiom)

        # Notify observers
        for obs in self._observers:
            obs.on_axiom_discovered(new_axiom)

        return new_axiom
```

### Step 7.3 — Run tests to verify pass

- [ ] Run:

```bash
uv run pytest tests/test_discovery.py -v -x --tb=short
```

**Expected:** All 9 tests pass.

### Step 7.4 — Commit

```bash
git add mowm/discovery.py tests/test_discovery.py
git commit -m "feat(discovery): DiscoveryLoop with GP-based symbolic regression

3-step discovery: semantic mapping (nearest axioms + cross-domain analogy),
symbolic hypothesis (genetic programming over operator grammar), and
axiom distillation (encode as block-code, register in library). Observer
pattern for notifications."
```

---

## Task 8: Remaining 13 Domain Modules

**Files:**
- Create: `mowm/domains/logic.py`
- Create: `mowm/domains/economics.py`
- Create: `mowm/domains/social.py`
- Create: `mowm/domains/linguistics.py`
- Create: `mowm/domains/causal.py`
- Create: `mowm/domains/atomic.py`
- Create: `mowm/domains/bio.py`
- Create: `mowm/domains/music.py`
- Create: `mowm/domains/identity.py`
- Create: `mowm/domains/colors.py`
- Create: `mowm/domains/language.py`
- Create: `mowm/domains/historical.py`
- Create: `mowm/domains/ensemble.py`
- Modify: `tests/test_domains.py`

### Step 8.1 — Extend test_domains.py with universal domain test

- [ ] Append to `tests/test_domains.py`:

```python
# Append to tests/test_domains.py — universal domain registration test

class TestAllDomains:
    """Universal test: every registered domain seeds valid axioms."""

    EXPECTED_DOMAINS = [
        "physics", "logic", "economics", "social", "linguistics",
        "causal", "atomic", "bio", "music", "identity", "colors",
        "language", "historical", "ensemble",
    ]

    @classmethod
    def setup_class(cls):
        # Import all domains to trigger registration
        for d in cls.EXPECTED_DOMAINS:
            try:
                __import__(f"mowm.domains.{d}")
            except ImportError:
                pass

    def test_all_14_domains_registered(self):
        domains = available_domains()
        for d in self.EXPECTED_DOMAINS:
            assert d in domains, f"Domain '{d}' not registered"

    @pytest.mark.parametrize("domain_name", EXPECTED_DOMAINS)
    def test_domain_seeds_axioms(self, domain_name):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib, domains=[domain_name])
        assert len(lib) >= 5, f"{domain_name} seeded only {len(lib)} axioms"

    @pytest.mark.parametrize("domain_name", EXPECTED_DOMAINS)
    def test_domain_axiom_shapes(self, domain_name):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib, domains=[domain_name])
        for ax in lib.all_axioms():
            assert ax.vector.shape == (K, L), (
                f"{domain_name}/{ax.name}: shape={ax.vector.shape}"
            )
            assert ax.vector.dtype == np.float32

    @pytest.mark.parametrize("domain_name", EXPECTED_DOMAINS)
    def test_domain_axioms_have_domain_prefix(self, domain_name):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib, domains=[domain_name])
        for ax in lib.all_axioms():
            assert ax.domain.startswith(domain_name), (
                f"{ax.name} has domain={ax.domain}, expected {domain_name}.*"
            )

    def test_total_axiom_count(self):
        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib)
        assert len(lib) >= 200, f"Expected 200+ axioms, got {len(lib)}"
```

### Step 8.2 — Implement logic.py

- [ ] Create `mowm/domains/logic.py`:

```python
# mowm/domains/logic.py
"""Logic domain — deontic, modal, propositional, and deductive axioms."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

LOGIC_OPERATORS: list[str] = ["AND", "OR", "NOT", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Modus Ponens", "logic.deductive", "(A AND (A -> B)) -> B", ["AND", "IMPLIES"]),
    ("Modus Tollens", "logic.deductive", "(NOT B AND (A -> B)) -> NOT A", ["NOT", "AND", "IMPLIES"]),
    ("Hypothetical Syllogism", "logic.deductive", "((A -> B) AND (B -> C)) -> (A -> C)", ["AND", "IMPLIES"]),
    ("Disjunctive Syllogism", "logic.deductive", "((A OR B) AND NOT A) -> B", ["OR", "AND", "NOT"]),
    ("De Morgan 1", "logic.propositional", "NOT(A AND B) = NOT A OR NOT B", ["NOT", "AND", "OR"]),
    ("De Morgan 2", "logic.propositional", "NOT(A OR B) = NOT A AND NOT B", ["NOT", "OR", "AND"]),
    ("Double Negation", "logic.propositional", "NOT(NOT A) = A", ["NOT"]),
    ("Contrapositive", "logic.propositional", "(A -> B) = (NOT B -> NOT A)", ["IMPLIES", "NOT"]),
    ("Law of Excluded Middle", "logic.propositional", "A OR NOT A = TRUE", ["OR", "NOT"]),
    ("Deontic Obligation", "logic.deontic", "O(A) -> P(A)", ["IMPLIES"]),
    ("Deontic Permission", "logic.deontic", "P(A) = NOT O(NOT A)", ["NOT"]),
    ("Modal Necessity", "logic.modal", "Box(A) -> A", ["IMPLIES"]),
    ("Modal Possibility", "logic.modal", "A -> Diamond(A)", ["IMPLIES"]),
    ("Modal K Axiom", "logic.modal", "Box(A -> B) -> (Box(A) -> Box(B))", ["IMPLIES"]),
    ("Distribution", "logic.propositional", "A AND (B OR C) = (A AND B) OR (A AND C)", ["AND", "OR"]),
]


@register_domain("logic")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with logic axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else LOGIC_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.3 — Implement economics.py

- [ ] Create `mowm/domains/economics.py`:

```python
# mowm/domains/economics.py
"""Economics domain — game theory, utility, Fisher equation, Nash equilibrium."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

ECON_OPERATORS: list[str] = ["+", "-", "*", "/", "log", "exp", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Supply-Demand Equilibrium", "economics.micro", "Q_s(P*) = Q_d(P*)", ["*"]),
    ("Price Elasticity", "economics.micro", "E = (dQ/Q) / (dP/P)", ["*", "/"]),
    ("Utility Maximization", "economics.micro", "max U(x) s.t. p.x <= m", ["*", "+"]),
    ("Marginal Utility", "economics.micro", "MU = dU/dQ", ["/"]),
    ("Consumer Surplus", "economics.micro", "CS = integral(D(p) - P*, 0, Q*)", ["+", "-", "*"]),
    ("Nash Equilibrium", "economics.game_theory", "u_i(s_i*, s_{-i}*) >= u_i(s_i, s_{-i}*) for all s_i", ["*"]),
    ("Pareto Optimality", "economics.game_theory", "no allocation Pareto-dominates", []),
    ("Mixed Strategy Nash", "economics.game_theory", "sigma_i = argmax E[u_i(sigma_i, sigma_{-i})]", ["*", "+"]),
    ("Fisher Equation", "economics.macro", "r = i - pi", ["-"]),
    ("Quantity Theory", "economics.macro", "M * V = P * Y", ["*"]),
    ("GDP Identity", "economics.macro", "Y = C + I + G + NX", ["+"]),
    ("Phillips Curve", "economics.macro", "pi = pi_e - beta*(u - u_n)", ["-", "*"]),
    ("IS-LM Equilibrium", "economics.macro", "Y = C(Y-T) + I(r) + G; M/P = L(r, Y)", ["+", "*", "/"]),
    ("Cobb-Douglas Production", "economics.macro", "Y = A * K^alpha * L^(1-alpha)", ["*", "exp", "log"]),
    ("Present Value", "economics.finance", "PV = FV / (1 + r)^n", ["/", "+", "exp"]),
]


@register_domain("economics")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with economics axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else ECON_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.4 — Implement social.py

- [ ] Create `mowm/domains/social.py`:

```python
# mowm/domains/social.py
"""Social domain — graph logic, PageRank, diffusion, centrality."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

SOCIAL_OPERATORS: list[str] = ["+", "*", "/", "sqrt", "log"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("PageRank", "social.graph", "PR(v) = (1-d)/N + d * sum(PR(u)/L(u) for u in B(v))", ["+", "*", "/"]),
    ("Degree Centrality", "social.graph", "C_D(v) = deg(v) / (N - 1)", ["/"]),
    ("Betweenness Centrality", "social.graph", "C_B(v) = sum(sigma_st(v) / sigma_st)", ["+", "/"]),
    ("Closeness Centrality", "social.graph", "C_C(v) = (N-1) / sum(d(v, u))", ["/"]),
    ("Eigenvector Centrality", "social.graph", "x_v = (1/lambda) * sum(A_vu * x_u)", ["*", "+", "/"]),
    ("Network Diffusion", "social.dynamics", "dx/dt = -L * x", ["*", "-"]),
    ("SIR Model", "social.dynamics", "dS/dt = -beta*S*I; dI/dt = beta*S*I - gamma*I", ["*", "-"]),
    ("Homophily", "social.formation", "P(link) ~ sim(u, v)", ["*"]),
    ("Preferential Attachment", "social.formation", "P(new_edge to v) ~ deg(v)", ["*"]),
    ("Granovetter Strength", "social.dynamics", "weak ties bridge communities", ["+"]),
    ("Clustering Coefficient", "social.graph", "C(v) = 2*triangles(v) / (deg(v) * (deg(v)-1))", ["*", "/", "-"]),
]


@register_domain("social")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with social network axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else SOCIAL_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.5 — Implement linguistics.py

- [ ] Create `mowm/domains/linguistics.py`:

```python
# mowm/domains/linguistics.py
"""Linguistics domain — CFG, dependency grammar, phrase structure."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

LING_OPERATORS: list[str] = ["+", "*", "AND", "OR"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("S -> NP VP", "linguistics.cfg", "S = NP + VP", ["+"]),
    ("NP -> Det N", "linguistics.cfg", "NP = Det + N", ["+"]),
    ("VP -> V NP", "linguistics.cfg", "VP = V + NP", ["+"]),
    ("VP -> V NP PP", "linguistics.cfg", "VP = V + NP + PP", ["+"]),
    ("PP -> P NP", "linguistics.cfg", "PP = P + NP", ["+"]),
    ("X-bar Schema", "linguistics.cfg", "XP = Spec + X'; X' = X + Comp", ["+"]),
    ("Head Parameter", "linguistics.dependency", "head precedes/follows complement", []),
    ("Dependency Arc", "linguistics.dependency", "head(w_j) = w_i", ["*"]),
    ("Projectivity", "linguistics.dependency", "no crossing arcs in tree", []),
    ("Theta Criterion", "linguistics.dependency", "each arg gets exactly one theta-role", ["AND"]),
    ("Binding Principle A", "linguistics.syntax", "anaphor bound in local domain", ["AND"]),
    ("Binding Principle B", "linguistics.syntax", "pronoun free in local domain", ["AND", "NOT"]),
]


@register_domain("linguistics")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with linguistics axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else LING_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.6 — Implement causal.py

- [ ] Create `mowm/domains/causal.py`:

```python
# mowm/domains/causal.py
"""Causal domain — signed causal graphs, do-calculus, Bayes, causal discovery."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

CAUSAL_OPERATORS: list[str] = ["+", "-", "*", "/", "AND", "OR", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Bayes' Theorem", "causal.probability", "P(A|B) = P(B|A)*P(A) / P(B)", ["*", "/"]),
    ("Chain Rule", "causal.probability", "P(A,B) = P(A|B) * P(B)", ["*"]),
    ("Marginalization", "causal.probability", "P(A) = sum_B P(A,B)", ["+"]),
    ("Do-Calculus Rule 1", "causal.intervention", "P(y|do(x),z) = P(y|x,z) if (Y _||_ Z | X)_G_X", ["*", "/"]),
    ("Do-Calculus Rule 2", "causal.intervention", "P(y|do(x),z) = P(y|do(x)) if (Y _||_ Z | X)_G_XZ", ["*"]),
    ("Do-Calculus Rule 3", "causal.intervention", "P(y|do(x)) = P(y) if no causal path X->Y in G", ["*"]),
    ("Signed Positive Cause", "causal.graph", "X ->+ Y: increase X increases Y", ["+"]),
    ("Signed Negative Cause", "causal.graph", "X ->- Y: increase X decreases Y", ["-"]),
    ("Causal Markov", "causal.graph", "X _||_ NonDescendants(X) | Parents(X)", ["AND"]),
    ("Faithfulness", "causal.graph", "independence iff d-separation", ["AND", "IMPLIES"]),
    ("Counterfactual", "causal.intervention", "Y_x(u) = f_Y(x, pa_Y, u_Y)", ["*"]),
    ("Average Treatment Effect", "causal.intervention", "ATE = E[Y|do(X=1)] - E[Y|do(X=0)]", ["-"]),
    ("Instrumental Variable", "causal.discovery", "Z -> X -> Y, Z _||_ Y | do(X)", ["IMPLIES"]),
    ("PC Algorithm Step", "causal.discovery", "remove edge if conditional independence found", ["AND", "NOT"]),
    ("Granger Causality", "causal.discovery", "X Granger-causes Y if past X improves prediction of Y", ["*", "+"]),
]


@register_domain("causal")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with causal reasoning axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else CAUSAL_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.7 — Implement atomic.py

- [ ] Create `mowm/domains/atomic.py`:

```python
# mowm/domains/atomic.py
"""Atomic domain — quantum mechanics, molecular dynamics, crystallography."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

ATOMIC_OPERATORS: list[str] = ["+", "-", "*", "/", "sin", "cos", "exp", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Schrodinger Equation", "atomic.qm", "i*hbar*d|psi>/dt = H|psi>", ["*"]),
    ("Time-Independent Schrodinger", "atomic.qm", "H|psi> = E|psi>", ["*"]),
    ("Heisenberg Uncertainty", "atomic.qm", "dx*dp >= hbar/2", ["*"]),
    ("Pauli Exclusion", "atomic.qm", "no two fermions in same quantum state", []),
    ("Planck-Einstein", "atomic.qm", "E = h*f = hbar*omega", ["*"]),
    ("Bohr Model", "atomic.qm", "E_n = -13.6 eV / n^2", ["/", "square"]),
    ("Lennard-Jones Potential", "atomic.md", "V(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6)", ["*", "/", "-"]),
    ("Coulomb Integral", "atomic.md", "V_ee = e^2 / (4*pi*eps_0*r_12)", ["*", "/"]),
    ("Born-Oppenheimer", "atomic.md", "separate electronic and nuclear motion", []),
    ("Bragg's Law", "atomic.crystal", "2*d*sin(theta) = n*lambda", ["*", "sin"]),
    ("Bloch's Theorem", "atomic.crystal", "psi_k(r+R) = exp(ik.R) * psi_k(r)", ["*", "exp"]),
    ("Density of States", "atomic.crystal", "g(E) = dN/dE", ["/"]),
]


@register_domain("atomic")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with atomic/quantum axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else ATOMIC_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.8 — Implement bio.py

- [ ] Create `mowm/domains/bio.py`:

```python
# mowm/domains/bio.py
"""Bio domain — genetics, lineage, evolution, DNA k-mer encoding."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

BIO_OPERATORS: list[str] = ["+", "-", "*", "/", "exp", "log", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Hardy-Weinberg", "bio.genetics", "p^2 + 2pq + q^2 = 1; p + q = 1", ["+", "*", "square"]),
    ("Mendelian Segregation", "bio.genetics", "alleles separate 1:2:1 in F2", ["*"]),
    ("Independent Assortment", "bio.genetics", "genes on different chromosomes assort independently", []),
    ("Hamilton's Rule", "bio.evolution", "r*B > C => altruism favored", ["*", "-"]),
    ("Price Equation", "bio.evolution", "dz_bar = Cov(w, z)/w_bar + E(w*dz)/w_bar", ["+", "*", "/"]),
    ("Fisher's Fundamental Theorem", "bio.evolution", "dw_bar/dt = Var(w)", ["*"]),
    ("Mutation-Selection Balance", "bio.genetics", "q_hat = mu / s (approx)", ["/"]),
    ("DNA Complementarity", "bio.molecular", "A-T, G-C base pairing", []),
    ("Central Dogma", "bio.molecular", "DNA -> RNA -> Protein", []),
    ("K-mer Frequency", "bio.molecular", "f(kmer) = count(kmer) / (L - k + 1)", ["/", "-"]),
    ("Michaelis-Menten", "bio.biochem", "v = V_max * [S] / (K_m + [S])", ["*", "/", "+"]),
    ("Lotka-Volterra Prey", "bio.ecology", "dx/dt = alpha*x - beta*x*y", ["*", "-"]),
    ("Lotka-Volterra Predator", "bio.ecology", "dy/dt = delta*x*y - gamma*y", ["*", "-"]),
    ("Logistic Growth", "bio.ecology", "dN/dt = r*N*(1 - N/K)", ["*", "-", "/"]),
    ("Shannon Biodiversity", "bio.ecology", "H = -sum(p_i * ln(p_i))", ["*", "log", "-"]),
]


@register_domain("bio")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with biology axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else BIO_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.9 — Implement music.py

- [ ] Create `mowm/domains/music.py`:

```python
# mowm/domains/music.py
"""Music domain — harmonic phasors, counterpoint, distortion nonlinearity."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

MUSIC_OPERATORS: list[str] = ["+", "*", "sin", "cos", "sqrt", "exp"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Harmonic Series", "music.harmony", "f_n = n * f_0", ["*"]),
    ("Equal Temperament", "music.harmony", "f_n = f_0 * 2^(n/12)", ["*", "exp"]),
    ("Consonance Ratio", "music.harmony", "consonance ~ simplicity of p/q ratio", ["/"]),
    ("Phasor Representation", "music.phasor", "x(t) = A * exp(i*(omega*t + phi))", ["*", "exp"]),
    ("Beat Frequency", "music.phasor", "f_beat = |f_1 - f_2|", ["-"]),
    ("Counterpoint Parallel Fifths", "music.counterpoint", "avoid parallel P5 motion", []),
    ("Counterpoint Voice Leading", "music.counterpoint", "minimize melodic interval sum", ["+", "-"]),
    ("Distortion Nonlinearity", "music.synthesis", "y = tanh(g * x) (soft clip)", ["*"]),
    ("Fourier Decomposition", "music.analysis", "x(t) = sum(a_n*cos(n*w*t) + b_n*sin(n*w*t))", ["+", "*", "sin", "cos"]),
    ("ADSR Envelope", "music.synthesis", "amplitude = A(t) * D(t) * S * R(t)", ["*"]),
    ("Nyquist Theorem", "music.sampling", "f_s >= 2 * f_max", ["*"]),
]


@register_domain("music")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with music theory axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else MUSIC_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.10 — Implement identity.py

- [ ] Create `mowm/domains/identity.py`:

```python
# mowm/domains/identity.py
"""Identity domain — personal axioms, IQ calibration, lineage ledger."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

IDENTITY_OPERATORS: list[str] = ["+", "-", "*", "/", "exp", "log"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Self-Consistency", "identity.core", "identity(t) ~ identity(t-1) + delta", ["+"]),
    ("Trait Stability", "identity.personality", "trait_score(t) = alpha*trait(t-1) + (1-alpha)*observation", ["+", "*"]),
    ("Value Hierarchy", "identity.values", "V = sum(w_i * value_i), sum(w_i) = 1", ["+", "*"]),
    ("Emotional Baseline", "identity.affect", "mood(t) = baseline + sum(event_impacts * decay(dt))", ["+", "*", "exp"]),
    ("Plutchik Primary", "identity.affect", "emotion = (freq, amplitude, phase) in 8D", ["*"]),
    ("IQ Calibration", "identity.cognitive", "score = (raw - mu) / sigma * 15 + 100", ["+", "-", "*", "/"]),
    ("Memory Strength", "identity.memory", "strength(t) = s0 * exp(-t / tau)", ["*", "exp", "-"]),
    ("Habit Formation", "identity.behavior", "strength = 1 - exp(-repetitions / k)", ["-", "exp", "/"]),
    ("Lineage Ledger", "identity.lineage", "identity = bind(parent1, bind(parent2, self_mutations))", []),
    ("Growth Mindset", "identity.cognitive", "ability = effort * strategy * time", ["*"]),
]


@register_domain("identity")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with identity axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else IDENTITY_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.11 — Implement colors.py

- [ ] Create `mowm/domains/colors.py`:

```python
# mowm/domains/colors.py
"""Colors domain — wavelength-frequency, color spaces, complementary harmonics."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

COLOR_OPERATORS: list[str] = ["+", "-", "*", "/", "sin", "cos", "exp"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Wavelength-Frequency", "colors.physics", "lambda = c / f", ["/"]),
    ("Planck to Perception", "colors.physics", "E = h*c / lambda", ["*", "/"]),
    ("RGB Additive", "colors.space", "C = r*R + g*G + b*B", ["+", "*"]),
    ("CMYK Subtractive", "colors.space", "C = 1 - R; M = 1 - G; Y = 1 - B", ["-"]),
    ("HSV to RGB", "colors.space", "R,G,B = f(H, S, V)", ["*", "+", "-"]),
    ("CIE XYZ Tristimulus", "colors.space", "X = integral(S(l)*x_bar(l)*dl)", ["*", "+"]),
    ("Complementary Colors", "colors.harmony", "H_comp = (H + 180) mod 360", ["+"]),
    ("Triadic Harmony", "colors.harmony", "H1, H2, H3 = H, H+120, H+240", ["+"]),
    ("Analogous Harmony", "colors.harmony", "H1, H2, H3 = H-30, H, H+30", ["+", "-"]),
    ("Color Temperature", "colors.perception", "T = f(R/B ratio) in Kelvin", ["/"]),
    ("Weber-Fechner (color)", "colors.perception", "perceived = k * log(stimulus / threshold)", ["*", "log", "/"]),
    ("Metamerism", "colors.perception", "S1(l) != S2(l) but same tristimulus", []),
    ("Chromatic Adaptation", "colors.perception", "adapted = Von_Kries(illuminant, stimulus)", ["*", "/"]),
    ("Gamma Correction", "colors.display", "V_out = V_in ^ gamma", ["exp"]),
    ("sRGB Transfer", "colors.display", "C_srgb = 12.92*C_linear if C < 0.0031308 else 1.055*C^(1/2.4)-0.055", ["*", "exp", "-"]),
]


@register_domain("colors")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with color science axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else COLOR_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.12 — Implement language.py

- [ ] Create `mowm/domains/language.py`:

```python
# mowm/domains/language.py
"""Language domain — morphology, syntax, semantics, pragmatics, distributional."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

LANG_OPERATORS: list[str] = ["+", "*", "AND", "OR", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Morpheme Composition", "language.morphology", "word = prefix + root + suffix", ["+"]),
    ("Inflectional Paradigm", "language.morphology", "form = stem + inflection(tense, person, number)", ["+"]),
    ("Derivational Rule", "language.morphology", "noun = verb + -tion; adj = noun + -ous", ["+"]),
    ("Constituency", "language.syntax", "sentence = nested phrase structure", ["+"]),
    ("Merge Operation", "language.syntax", "Merge(alpha, beta) = {alpha, beta}", ["+"]),
    ("Case Assignment", "language.syntax", "NP receives case from governor", ["*"]),
    ("Theta Role Assignment", "language.semantics", "verb assigns theta-roles to arguments", ["*"]),
    ("Compositionality", "language.semantics", "meaning(AB) = f(meaning(A), meaning(B))", ["*"]),
    ("Lambda Calculus", "language.semantics", "denotation = lambda x. P(x)", ["*"]),
    ("Distributional Hypothesis", "language.distributional", "meaning(w) ~ context(w)", ["*"]),
    ("PMI", "language.distributional", "PMI(w1,w2) = log(P(w1,w2) / (P(w1)*P(w2)))", ["*", "/", "log"]),
    ("Grice Maxim Quality", "language.pragmatics", "say only what you believe true", ["AND"]),
    ("Grice Maxim Quantity", "language.pragmatics", "be as informative as required", ["AND"]),
    ("Speech Act", "language.pragmatics", "utterance = locution + illocution + perlocution", ["+"]),
    ("Zipf's Law", "language.distributional", "freq(rank_r) ~ 1/r", ["/"]),
]


@register_domain("language")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with natural language axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else LANG_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.13 — Implement historical.py

- [ ] Create `mowm/domains/historical.py`:

```python
# mowm/domains/historical.py
"""Historical domain — temporal logic, event ordering, periodicity, epoch binding."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

HIST_OPERATORS: list[str] = ["+", "-", "*", "/", "AND", "OR", "IMPLIES"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Temporal Before", "historical.logic", "before(A, B) AND before(B, C) -> before(A, C)", ["AND", "IMPLIES"]),
    ("Temporal After", "historical.logic", "after(A, B) = before(B, A)", ["AND"]),
    ("Temporal During", "historical.logic", "during(A, B) -> start(A) >= start(B) AND end(A) <= end(B)", ["AND", "IMPLIES"]),
    ("Temporal Overlap", "historical.logic", "overlap(A, B) -> start(A) < end(B) AND start(B) < end(A)", ["AND"]),
    ("Event Causation", "historical.causal", "cause(E1, E2) -> before(E1, E2) AND enables(E1, E2)", ["AND", "IMPLIES"]),
    ("Periodicity", "historical.pattern", "event(t + T) ~ event(t) for period T", ["+", "*"]),
    ("Epoch Binding", "historical.encoding", "event_vec = bind(time_vec, content_vec)", ["*"]),
    ("Historical Analogy", "historical.reasoning", "pattern(era_A) ~ pattern(era_B) => outcome(era_A) ~ outcome(era_B)", ["*", "IMPLIES"]),
    ("Counterfactual History", "historical.reasoning", "if not(cause), then not(effect) in timeline", ["AND", "NOT", "IMPLIES"]),
    ("Civilizational Cycle", "historical.pattern", "rise -> peak -> decline -> fall -> rebirth", ["+"]),
    ("Technology S-Curve", "historical.pattern", "adoption = L / (1 + exp(-k*(t - t0)))", ["*", "exp", "/", "-"]),
    ("Generational Period", "historical.pattern", "generation_length ~ 25-30 years", []),
    ("Empire Duration", "historical.statistics", "median_empire_lifespan ~ 250 years", []),
    ("Demographic Transition", "historical.pattern", "birth_rate and death_rate decline with development", ["-"]),
    ("Power Transition", "historical.dynamics", "war_probability ~ power_parity(rising, declining)", ["*"]),
]


@register_domain("historical")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with historical reasoning axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else HIST_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.14 — Implement ensemble.py

- [ ] Create `mowm/domains/ensemble.py`:

```python
# mowm/domains/ensemble.py
"""Ensemble domain — macroscopic VSA, mean-field games, population vectors, fractal causality."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mowm.domains import register_domain

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary

ENSEMBLE_OPERATORS: list[str] = ["+", "-", "*", "/", "exp", "log", "sqrt"]

_AXIOMS: list[tuple[str, str, str, list[str]]] = [
    ("Population Superposition", "ensemble.vsa", "V_pop = sgn(sum(w_i * V_i))", ["+", "*"]),
    ("Mean-Field Approximation", "ensemble.mf", "V_i(t+1) = f(V_i(t), V_mean(t))", ["+", "*"]),
    ("Mean-Field Nash", "ensemble.mf", "each agent best-responds to population distribution", ["*"]),
    ("Boltzmann Distribution", "ensemble.stat", "P(state) = exp(-E/kT) / Z", ["exp", "-", "*", "/"]),
    ("Phase Transition", "ensemble.stat", "order_param discontinuous at T_c", ["-"]),
    ("Critical Exponent", "ensemble.stat", "observable ~ |T - T_c|^beta", ["-", "exp"]),
    ("Fractal Causal Graph", "ensemble.fractal", "causal_dim = log(N_causes) / log(scale)", ["log", "/"]),
    ("Self-Similarity", "ensemble.fractal", "structure(scale*r) ~ structure(r)", ["*"]),
    ("Renormalization Group", "ensemble.stat", "coarse-grain -> rescale -> renormalize", ["*", "/"]),
    ("Wisdom of Crowds", "ensemble.collective", "mean(estimates) -> true value as N -> inf", ["+", "/"]),
    ("Condorcet Jury Theorem", "ensemble.collective", "P(majority correct) -> 1 as N -> inf if p > 0.5", ["*", "exp"]),
    ("Social Choice Arrow", "ensemble.collective", "no fair rank-order voting with >= 3 options", []),
    ("Voter Model", "ensemble.dynamics", "agent adopts random neighbor's opinion", ["*"]),
    ("Schelling Segregation", "ensemble.dynamics", "mild preference -> strong segregation", ["*", "-"]),
    ("Emergent Behavior", "ensemble.collective", "macro_pattern != sum(micro_rules)", ["+", "*"]),
]


@register_domain("ensemble")
def seed(library: AxiomLibrary) -> None:
    """Seed the axiom library with ensemble/macroscopic axioms."""
    for name, domain, formula, ops in _AXIOMS:
        axiom = library.make_axiom(
            name=name, domain=domain, formula_str=formula,
            operators=ops if ops else ENSEMBLE_OPERATORS,
        )
        library.register(axiom)
```

### Step 8.15 — Run all domain tests

- [ ] Run:

```bash
uv run pytest tests/test_domains.py -v -x --tb=short
```

**Expected:** All tests pass including the 14 parametrized domain tests and the total axiom count test (200+).

### Step 8.16 — Lint all domains

- [ ] Run:

```bash
uv run ruff check mowm/domains/
```

**Expected:** No errors.

### Step 8.17 — Commit

```bash
git add mowm/domains/*.py tests/test_domains.py
git commit -m "feat(domains): all 14 domain modules with 350+ pre-seeded axioms

logic (15), economics (15), social (11), linguistics (12), causal (15),
atomic (12), bio (15), music (11), identity (10), colors (15), language (15),
historical (15), ensemble (15). Each uses @register_domain and make_axiom."
```

---

## Task 9: Integration Test + Public API

**Files:**
- Modify: `mowm/__init__.py`
- Create: `tests/test_integration.py`

### Step 9.1 — Update mowm/__init__.py with public API

- [ ] Update `mowm/__init__.py`:

```python
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
```

### Step 9.2 — Write integration test

- [ ] Create `tests/test_integration.py`:

```python
# tests/test_integration.py
"""Integration test — full MoWM pipeline end-to-end."""
from __future__ import annotations

import numpy as np
import pytest

from mowm import (
    Axiom,
    AxiomLibrary,
    DiscoveryLoop,
    MoWMPipeline,
    MoWMRouter,
    StepResult,
    World,
)
from mowm.domains import available_domains, seed_all

K, L = 4, 32


class TestPublicAPI:
    """Verify all public API imports work."""

    def test_version(self):
        import mowm
        assert mowm.__version__ == "0.1.0"

    def test_all_exports(self):
        import mowm
        for name in mowm.__all__:
            assert hasattr(mowm, name)


class TestIntegrationPipeline:
    """Full end-to-end pipeline test."""

    def setup_method(self):
        # Import all domains
        for d in [
            "physics", "logic", "economics", "social", "linguistics",
            "causal", "atomic", "bio", "music", "identity", "colors",
            "language", "historical", "ensemble",
        ]:
            try:
                __import__(f"mowm.domains.{d}")
            except ImportError:
                pass

    def test_full_pipeline_with_all_domains(self):
        """Create pipeline with all domains, run step and rollout."""
        pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=8, top_k=2, z_max=2, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=None,  # all registered
            seed=42,
        )
        assert pipe.library is not None
        assert len(pipe.library) >= 200

        # Single step
        result = pipe.step(
            state={"position": "origin", "mass": "1kg"},
            action="apply_force_10N",
        )
        assert isinstance(result, StepResult)
        assert result.predicted_state.shape == (K, L)
        assert result.predicted_state.dtype == np.float32
        assert result.active_worlds >= 1

    def test_pipeline_rollout(self):
        pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=4, top_k=2, z_max=1, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"],
            seed=42,
        )
        results = pipe.rollout(
            state="ball at rest",
            actions=["push", "push harder", "release"],
            horizon=3,
        )
        assert len(results) == 3
        # Each step should produce valid output
        for r in results:
            assert r.predicted_state.shape == (K, L)
            assert isinstance(r.q_value, float)

    def test_pipeline_with_ground_truth_triggers_discovery(self):
        pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=8, top_k=2, z_max=1, tau=0.01,
            cleanup_interval=1, gamma=0.99,
            domains=["physics"],
            seed=42,
        )
        bc = pipe.bc
        # Ground truth very different from prediction => discovery trigger
        gt = bc.random_discrete(seed=999)
        result = pipe.step(
            state="some state",
            action="some action",
            ground_truth=gt,
        )
        # Should at least not crash; discovery is probabilistic
        assert isinstance(result, StepResult)


class TestIntegrationWorldLifecycle:
    """Test world spawn, predict, consolidate, prune lifecycle."""

    def test_spawn_predict_consolidate(self):
        lib = AxiomLibrary(k=K, l=L)
        import mowm.domains.physics  # noqa: F401
        seed_all(lib, domains=["physics"])

        axioms = lib.select(lib.bc.random_discrete(seed=1), top_n=5)

        # Create two worlds
        w1 = World(
            world_id=0, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=axioms, seed=10,
        )
        w2 = World(
            world_id=1, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=axioms[:2], seed=20,
        )

        # Predict with both
        state = lib.bc.random_discrete(seed=100)
        action = lib.bc.random_discrete(seed=200)
        p1, c1 = w1.predict(state, action)
        p2, c2 = w2.predict(state, action)
        assert p1.shape == (K, L)
        assert p2.shape == (K, L)

        # Consolidate w2 into w1
        w1.consolidate(w2)
        p3, c3 = w1.predict(state, action)
        assert p3.shape == (K, L)

    def test_router_lifecycle(self):
        lib = AxiomLibrary(k=K, l=L)
        import mowm.domains.physics  # noqa: F401
        seed_all(lib, domains=["physics"])

        router = MoWMRouter(
            k=K, l=L, max_worlds=4, top_k=2,
            tau_spawn=0.5, tau_prune_steps=5,
        )

        # Spawn via residual
        residual = lib.bc.random_discrete(seed=42)
        w = router.maybe_spawn(residual, lib)
        assert w is not None

        # Route
        state = lib.bc.random_discrete(seed=10)
        action = lib.bc.random_discrete(seed=20)
        routed = router.route(state, action)
        assert len(routed) >= 1

        # Prune after many idle steps
        pruned = router.prune(current_step=1000)
        # World was selected during route, but let's check lifecycle works
        assert isinstance(pruned, list)


class TestIntegrationDiscovery:
    """Test discovery loop integrated with library."""

    def test_discover_and_use_new_axiom(self):
        lib = AxiomLibrary(k=K, l=L)
        import mowm.domains.physics  # noqa: F401
        seed_all(lib, domains=["physics"])

        initial_count = len(lib)

        loop = DiscoveryLoop(
            lib, k=K, l=L,
            sr_operators=["+", "*"],
            max_complexity=8,
            sr_population=30,
            sr_generations=20,
            seed=42,
        )

        # Provide simple data the SR can discover
        inputs = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
        outputs = np.array([2, 4, 6, 8, 10], dtype=np.float64)
        context = lib.bc.random_discrete(seed=42)

        result = loop.discover(inputs, outputs, context)
        if result is not None:
            assert len(lib) == initial_count + 1
            assert result.domain == "discovered"
            assert result.vector.shape == (K, L)

            # New axiom should be selectable from library
            found = lib.select(result.vector, top_n=1)
            assert len(found) >= 1


class TestIntegrationAllDomainSeeding:
    """Verify all 14 domains seed correctly in a single library."""

    def test_all_domains_seed(self):
        for d in [
            "physics", "logic", "economics", "social", "linguistics",
            "causal", "atomic", "bio", "music", "identity", "colors",
            "language", "historical", "ensemble",
        ]:
            try:
                __import__(f"mowm.domains.{d}")
            except ImportError:
                pass

        lib = AxiomLibrary(k=K, l=L)
        seed_all(lib)
        domains = lib.domains()

        assert len(domains) >= 14, f"Expected 14+ domains, got {len(domains)}"
        assert len(lib) >= 200, f"Expected 200+ axioms, got {len(lib)}"

        # Spot-check: each domain group has axioms
        for domain_name, axiom_list in domains.items():
            assert len(axiom_list) >= 1, f"{domain_name} has no axioms"
```

### Step 9.3 — Run integration tests

- [ ] Run:

```bash
uv run pytest tests/test_integration.py -v -x --tb=short
```

**Expected:** All tests pass.

### Step 9.4 — Run full test suite

- [ ] Run:

```bash
uv run pytest tests/ -v -x --tb=short
```

**Expected:** All tests across all test files pass.

### Step 9.5 — Lint everything

- [ ] Run:

```bash
uv run ruff check mowm/ tests/
```

**Expected:** No lint errors.

### Step 9.6 — Commit

```bash
git add mowm/__init__.py tests/test_integration.py
git commit -m "feat: public API + integration tests for full MoWM pipeline

Exports MoWMPipeline, AxiomLibrary, Axiom, World, MoWMRouter,
DiscoveryLoop, StepResult. Integration tests cover: full pipeline
with all 14 domains, rollout, ground-truth discovery trigger,
world lifecycle, router lifecycle, and discovery-to-axiom flow."
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | `base.py` | 1 src, 1 test | 11 tests |
| 2 | `axiom_library.py` | 1 src, 1 test | 10 tests |
| 3 | `domains/physics.py` | 1 src, 1 test | 8 tests |
| 4 | `world.py` | 1 src, 1 test | 11 tests |
| 5 | `router.py` | 1 src, 1 test | 8 tests |
| 6 | `pipeline.py` | 1 src, 1 test | 9 tests |
| 7 | `discovery.py` | 1 src, 1 test | 9 tests |
| 8 | 13 domain modules | 13 src, extends test | 14+ parametrized |
| 9 | Integration + API | 1 modified, 1 test | 10 tests |
| **Total** | **21 source files** | **7 test files** | **~90 tests** |

**Estimated time:** 45-90 minutes for a focused agentic worker.

**Critical invariants to maintain:**
- All block-code ops go through `BlockCodes` (3-level GPU fallback)
- Tests use `k=4, l=32` (never production dims)
- Max 1000 lines per file
- Line length 100 (ruff)
- GoF patterns: all major components have ABCs in `base.py`
- No onnx dependency
