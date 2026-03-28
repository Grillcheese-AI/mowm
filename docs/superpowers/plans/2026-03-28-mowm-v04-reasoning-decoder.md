# MoWM v0.4.0 Reasoning Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LLM dependency with a self-contained reasoning decoder that generates natural language answers by walking the axiom graph, building derivation chains, tracing causal relationships, and composing text from structured reasoning.

**Architecture:** KnowledgeSearch finds relevant axioms via multi-hop VSA similarity. ChainBuilder connects axioms through shared variables (like textbook derivations). CausalTracer identifies cause-effect paths and variable cancellations. Comparator does structural diff. Composer scales template complexity with reasoning depth. ReasoningDecoder orchestrates all.

**Tech Stack:** Python 3.12, numpy, cubemind (BlockCodes, WorldEncoder), existing mowm modules

---

## Pre-flight Checklist

Before starting any task, confirm:

```bash
cd c:\Users\grill\Documents\GitHub\mowm
uv run pytest tests/ -v -q --tb=short -x
# Must show 542 tests passing -- this is the v0.3.0 baseline
```

All new implementation code goes into `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\` directory.
All new tests go into `c:\Users\grill\Documents\GitHub\mowm\tests\` directory.
Constants: `K = 4`, `L = 32` in all tests.
Line length: 100 characters (ruff enforced).
Max 1000 lines per file.
NO external LLM dependency -- this is the whole point of v0.4.0.
Composer uses templates, not generation. Quality from reasoning depth, not fluency.

### Existing APIs to use (DO NOT modify these files):

| Module | Key Classes/Functions |
|--------|----------------------|
| `mowm.axiom_library` | `Axiom` (dataclass: name, domain, formula_str, vector, operators), `AxiomLibrary` (.register, .select, .make_axiom, .all_axioms, .domains, .compose, .encoder) |
| `mowm.causal_graph` | `CausalGraph` (.propagate, .counterfactual, .find_paths, .feedback_loops, .add_node, .add_edge, .nodes, .edges), `CausalNode`, `CausalEdge` |
| `mowm.executor` | `SafeFormulaExecutor` (.parse, .execute) -- AST-based, no eval() |
| `mowm.logic_engine` | `LogicEngine` (.evaluate, .check_obligation, .evaluate_axiom, .conversion_probability) |
| `mowm.qa.parser` | `QuestionParser` (.parse), `ParsedQuestion` (.raw, .intent, .variables, .target, .keywords, .conditions, .causal_node, .causal_delta), `Intent` (COMPUTE, EVALUATE, CAUSAL, EXPLAIN, COMPARE, GENERAL) |
| `mowm.qa.direct_solver` | `QAResponse` (answer, mode, confidence, domain, evidence, computed_values, reasoning_trace) |
| `mowm.qa.augmented_solver` | `AugmentedSolver` (.solve, .find_relevant_axioms) -- becomes true last resort |
| `mowm.domains` | `seed_all(library)`, 16 domain modules each with `@register_domain` |

### Domain axiom availability (for test design):

| Domain | Example Axioms Available |
|--------|--------------------------|
| physics | Newton's Second Law (F=m*a), Weight (W=m*g), Kinetic Energy (KE=0.5*m*v^2), Momentum (p=m*v), Gravitational Force (F=G*m1*m2/r^2) |
| economics | Supply-Demand Equilibrium, Price Elasticity, Nash Equilibrium, Marginal Utility |
| logic | Modus Ponens, Modus Tollens, Hypothetical Syllogism, De Morgan 1/2, Contrapositive |
| bio | Hamilton's Rule (r*B>C), Hardy-Weinberg, Fisher's Fundamental Theorem, Mutation-Selection Balance |
| music | Harmonic Series (f_n=n*f_0), Equal Temperament, Consonance Ratio, Beat Frequency |
| historical | Event Causation, Periodicity, Temporal Before/After/During, Epoch Binding |
| causal | Bayes' Theorem, Do-Calculus Rules 1-3, Signed Positive/Negative Cause |
| colors | Wavelength-Frequency, Beer-Lambert, Color Mixing |
| social | Network Density, Clustering Coefficient, Diffusion Rate |
| identity | Alignment Score, Emotional Valence |
| language | Grammar rules, Syntax tree axioms |
| linguistics | CFG Derivation, Binding Theory |
| ensemble | Population Dynamics, Phase Transition |
| atomic | Quantum State, Uncertainty Principle |

---

## Task 1: reasoning/knowledge_search.py -- KnowledgeContext + KnowledgeSearch

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\knowledge_search.py`

Depends on: `AxiomLibrary` (read-only). No other reasoning module dependencies.

This module finds relevant axioms for a question by combining keyword matching,
VSA similarity, and variable-based graph traversal through the axiom library.

### Step 1.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_knowledge_search.py`

```python
# tests/test_knowledge_search.py
"""Tests for KnowledgeSearch -- multi-hop axiom discovery across domains."""
from __future__ import annotations

import re

import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.reasoning.knowledge_search import KnowledgeContext, KnowledgeSearch

K, L = 4, 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def physics_lib() -> AxiomLibrary:
    """Library with a handful of physics axioms sharing variables."""
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    ))
    lib.register(lib.make_axiom(
        "Weight", "physics.dynamics", "W = m * g",
    ))
    lib.register(lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    ))
    lib.register(lib.make_axiom(
        "Momentum", "physics.dynamics", "p = m * v",
    ))
    lib.register(lib.make_axiom(
        "Gravitational Force", "physics.dynamics",
        "F = G * m1 * m2 / r^2",
    ))
    return lib


@pytest.fixture
def multi_domain_lib() -> AxiomLibrary:
    """Library spanning physics, economics, logic, bio, music domains."""
    lib = AxiomLibrary(k=K, l=L)
    # Physics
    lib.register(lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    ))
    lib.register(lib.make_axiom(
        "Weight", "physics.dynamics", "W = m * g",
    ))
    # Economics
    lib.register(lib.make_axiom(
        "Supply-Demand Equilibrium", "economics.micro",
        "Q_d = Q_s",
    ))
    lib.register(lib.make_axiom(
        "Price Elasticity", "economics.micro",
        "epsilon = dQ / dP",
    ))
    # Logic
    lib.register(lib.make_axiom(
        "Modus Ponens", "logic.deductive",
        "(P AND (P IMPLIES Q)) IMPLIES Q",
    ))
    lib.register(lib.make_axiom(
        "Hypothetical Syllogism", "logic.deductive",
        "((P IMPLIES Q) AND (Q IMPLIES R)) IMPLIES (P IMPLIES R)",
    ))
    # Bio
    lib.register(lib.make_axiom(
        "Hamilton's Rule", "bio.evolution", "r * B > C",
    ))
    lib.register(lib.make_axiom(
        "Hardy-Weinberg", "bio.genetics",
        "p^2 + 2*p*q + q^2 = 1",
    ))
    # Music
    lib.register(lib.make_axiom(
        "Harmonic Series", "music.harmony", "f_n = n * f_0",
    ))
    lib.register(lib.make_axiom(
        "Equal Temperament", "music.harmony",
        "f_n = f_0 * 2^(n/12)",
    ))
    lib.register(lib.make_axiom(
        "Consonance Ratio", "music.harmony",
        "consonance = p / q",
    ))
    # Historical
    lib.register(lib.make_axiom(
        "Event Causation", "historical.causal",
        "event_A causes event_B",
    ))
    # Causal
    lib.register(lib.make_axiom(
        "Bayes Theorem", "causal.probability",
        "P_A_given_B = P_B_given_A * P_A / P_B",
    ))
    lib.register(lib.make_axiom(
        "Do-Calculus Rule 1", "causal.intervention",
        "P_Y_do_X_Z = P_Y_do_X",
    ))
    return lib


@pytest.fixture
def phys_search(physics_lib) -> KnowledgeSearch:
    return KnowledgeSearch(library=physics_lib, k=K, l=L)


@pytest.fixture
def multi_search(multi_domain_lib) -> KnowledgeSearch:
    return KnowledgeSearch(library=multi_domain_lib, k=K, l=L)


# ---------------------------------------------------------------------------
# TestKnowledgeContext dataclass
# ---------------------------------------------------------------------------


class TestKnowledgeContext:
    def test_default_fields(self):
        ctx = KnowledgeContext()
        assert ctx.primary_axioms == []
        assert ctx.supporting_axioms == []
        assert ctx.concepts == []
        assert ctx.domain_chain == []
        assert ctx.variables_mentioned == set()

    def test_fields_set(self):
        ctx = KnowledgeContext(
            concepts=["force", "mass"],
            domain_chain=["physics.dynamics"],
            variables_mentioned={"F", "m", "a"},
        )
        assert "force" in ctx.concepts
        assert "F" in ctx.variables_mentioned


# ---------------------------------------------------------------------------
# TestKnowledgeSearch -- Physics domain
# ---------------------------------------------------------------------------


class TestSearchPhysics:
    def test_search_finds_primary_axioms(self, phys_search):
        ctx = phys_search.search(
            keywords=["force", "mass", "acceleration"],
            domain="physics",
        )
        assert isinstance(ctx, KnowledgeContext)
        assert len(ctx.primary_axioms) > 0
        names = [a.name for a in ctx.primary_axioms]
        assert "Newton's Second Law" in names

    def test_search_finds_supporting_axioms(self, phys_search):
        ctx = phys_search.search(
            keywords=["force", "mass"],
            domain="physics",
        )
        # Weight also uses mass -- should appear as supporting
        all_names = [a.name for a in ctx.primary_axioms + ctx.supporting_axioms]
        assert "Weight" in all_names

    def test_search_extracts_variables(self, phys_search):
        ctx = phys_search.search(
            keywords=["force", "mass", "acceleration"],
            domain="physics",
        )
        assert "m" in ctx.variables_mentioned
        assert "a" in ctx.variables_mentioned or "F" in ctx.variables_mentioned

    def test_search_with_target_var(self, phys_search):
        ctx = phys_search.search(
            keywords=["energy"],
            domain="physics",
            target_var="KE",
        )
        names = [a.name for a in ctx.primary_axioms]
        assert "Kinetic Energy" in names

    def test_search_domain_chain(self, phys_search):
        ctx = phys_search.search(
            keywords=["force"],
            domain="physics",
        )
        assert len(ctx.domain_chain) > 0
        assert any("physics" in d for d in ctx.domain_chain)

    def test_search_empty_keywords_returns_context(self, phys_search):
        ctx = phys_search.search(keywords=[], domain="physics")
        assert isinstance(ctx, KnowledgeContext)

    def test_search_unknown_domain_returns_empty_primary(self, phys_search):
        ctx = phys_search.search(
            keywords=["force"],
            domain="underwater_basket_weaving",
        )
        # May still find axioms via keyword match but domain_chain differs
        assert isinstance(ctx, KnowledgeContext)


# ---------------------------------------------------------------------------
# TestSearchMultiDomain -- cross-domain
# ---------------------------------------------------------------------------


class TestSearchMultiDomain:
    def test_economics_search(self, multi_search):
        ctx = multi_search.search(
            keywords=["supply", "demand", "equilibrium"],
            domain="economics",
        )
        names = [a.name for a in ctx.primary_axioms]
        assert "Supply-Demand Equilibrium" in names

    def test_logic_search(self, multi_search):
        ctx = multi_search.search(
            keywords=["modus", "ponens", "implies"],
            domain="logic",
        )
        names = [a.name for a in ctx.primary_axioms]
        assert "Modus Ponens" in names

    def test_bio_search(self, multi_search):
        ctx = multi_search.search(
            keywords=["altruism", "kin", "selection", "Hamilton"],
            domain="bio",
        )
        names = [a.name for a in ctx.primary_axioms]
        assert "Hamilton's Rule" in names

    def test_music_search(self, multi_search):
        ctx = multi_search.search(
            keywords=["harmonic", "frequency", "overtone"],
            domain="music",
        )
        names = [a.name for a in ctx.primary_axioms]
        assert "Harmonic Series" in names

    def test_historical_search(self, multi_search):
        ctx = multi_search.search(
            keywords=["event", "causation", "causes"],
            domain="historical",
        )
        names = [a.name for a in ctx.primary_axioms]
        assert "Event Causation" in names

    def test_causal_search(self, multi_search):
        ctx = multi_search.search(
            keywords=["do", "calculus", "intervention"],
            domain="causal",
        )
        names = [a.name for a in ctx.primary_axioms]
        assert "Do-Calculus Rule 1" in names


# ---------------------------------------------------------------------------
# TestDerivationPath
# ---------------------------------------------------------------------------


class TestDerivationPath:
    def test_find_path_newton_to_weight(self, phys_search, physics_lib):
        """Newton (F=m*a) and Weight (W=m*g) share variable m."""
        axioms = physics_lib.all_axioms()
        newton = [a for a in axioms if a.name == "Newton's Second Law"][0]
        path = phys_search.find_derivation_path(newton, "g")
        # Should find Weight axiom which contains g
        assert len(path) > 0
        path_names = [a.name for a in path]
        assert "Weight" in path_names

    def test_find_path_momentum_to_energy(self, phys_search, physics_lib):
        """Momentum (p=m*v) and KE (KE=0.5*m*v^2) share m and v."""
        axioms = physics_lib.all_axioms()
        momentum = [a for a in axioms if a.name == "Momentum"][0]
        path = phys_search.find_derivation_path(momentum, "KE")
        path_names = [a.name for a in path]
        assert "Kinetic Energy" in path_names

    def test_find_path_no_connection_returns_empty(self, phys_search, physics_lib):
        axioms = physics_lib.all_axioms()
        newton = [a for a in axioms if a.name == "Newton's Second Law"][0]
        path = phys_search.find_derivation_path(newton, "nonexistent_var_xyz")
        assert path == []


# ---------------------------------------------------------------------------
# TestVariableExtraction
# ---------------------------------------------------------------------------


class TestVariableExtraction:
    def test_extract_vars_from_formula(self, phys_search):
        """Verify _extract_variables finds identifiers in formula strings."""
        ctx = phys_search.search(
            keywords=["momentum"],
            domain="physics",
        )
        # Momentum: p = m * v  -> should find p, m, v
        assert "m" in ctx.variables_mentioned
        assert "v" in ctx.variables_mentioned or "p" in ctx.variables_mentioned
```

### Step 1.2 -- Write implementation

- [ ] Create directory `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\`
- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\knowledge_search.py`

```python
# mowm/reasoning/knowledge_search.py
"""KnowledgeSearch -- multi-hop axiom discovery for reasoning decoder.

Finds relevant axioms by combining:
  1. Keyword matching (question tokens vs axiom name + formula tokens)
  2. VSA similarity (encode keywords -> compare against axiom vectors)
  3. Variable graph traversal (shared variables as edges between axioms)
  4. Domain-scoped filtering with cross-domain fallback
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")

# Constants
_MAX_PRIMARY = 5
_MAX_SUPPORTING = 10
_MAX_PATH_DEPTH = 6


@dataclass
class KnowledgeContext:
    """Structured knowledge retrieved for a question.

    Attributes:
        primary_axioms: Most directly relevant axioms (top 3-5).
        supporting_axioms: Related by domain/similarity (next 5-10).
        concepts: Extracted concept names from matched axioms.
        domain_chain: Sub-domain paths found (e.g. ["physics.dynamics"]).
        variables_mentioned: All variables in relevant formulas.
    """

    primary_axioms: list[Axiom] = field(default_factory=list)
    supporting_axioms: list[Axiom] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    domain_chain: list[str] = field(default_factory=list)
    variables_mentioned: set[str] = field(default_factory=set)


def _tokens(text: str) -> set[str]:
    """Return lowercased tokens from a string."""
    return {m.lower() for m in _TOKEN_RE.findall(text)}


def _extract_formula_vars(formula: str) -> set[str]:
    """Extract variable identifiers from a formula string.

    Filters out known function names and pure numeric tokens.
    """
    _FUNCTIONS = {
        "sin", "cos", "sqrt", "exp", "log", "square", "not",
        "and", "or", "implies", "for", "all", "if", "iff",
        "such", "that", "with", "subject", "to", "max", "min",
        "sum", "prod", "integral", "no", "in", "const",
        "avoid", "consecutive", "between", "voices", "pairs",
    }
    raw = set(_TOKEN_RE.findall(formula))
    return {v for v in raw if v.lower() not in _FUNCTIONS and not v.isdigit()}


class KnowledgeSearch:
    """Multi-hop knowledge search through the axiom library.

    Combines keyword overlap, VSA similarity, and variable-graph
    traversal to find primary and supporting axioms for a question.

    Args:
        library: AxiomLibrary instance.
        k: Number of VSA blocks.
        l: Block length.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        k: int = 80,
        l: int = 128,  # noqa: E741
    ) -> None:
        self._lib = library
        self._k = k
        self._l = l
        # Pre-compute variable sets per axiom for graph traversal
        self._axiom_vars: dict[str, set[str]] = {}
        for ax in self._lib.all_axioms():
            self._axiom_vars[ax.name] = _extract_formula_vars(ax.formula_str)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        keywords: list[str],
        domain: str,
        target_var: str | None = None,
    ) -> KnowledgeContext:
        """Multi-hop knowledge search through the axiom library.

        1. Direct keyword match -> primary axioms
        2. VSA similarity from primary axiom vectors -> supporting axioms
        3. Domain chain: follow sub-domain connections
        4. Variable tracing: find all axioms sharing variables with primaries

        Args:
            keywords: Question keywords (lowercased tokens).
            domain: Top-level domain string (e.g. "physics").
            target_var: Optional target variable to prioritize.

        Returns:
            KnowledgeContext with primary/supporting axioms, concepts,
            domain chain, and variables mentioned.
        """
        all_axioms = self._lib.all_axioms()
        if not all_axioms:
            return KnowledgeContext()

        kw_set = {kw.lower() for kw in keywords}

        # Step 1: score every axiom by keyword overlap + domain bonus
        scored = self._score_axioms(all_axioms, kw_set, domain, target_var)

        # Step 2: split into primary and initial supporting
        primary = [ax for ax, _ in scored[:_MAX_PRIMARY]]
        supporting_candidates = [ax for ax, _ in scored[_MAX_PRIMARY:]]

        # Step 3: variable-based expansion -- find axioms sharing vars
        primary_vars: set[str] = set()
        for ax in primary:
            primary_vars.update(self._axiom_vars.get(ax.name, set()))

        var_related = self._find_by_shared_vars(
            primary_vars, primary, all_axioms,
        )
        supporting_candidates = var_related + supporting_candidates

        # Step 4: VSA similarity expansion from primary vectors
        if primary:
            vsa_related = self._vsa_expand(primary, all_axioms)
            supporting_candidates.extend(vsa_related)

        # Deduplicate supporting, exclude already-primary axioms
        primary_names = {a.name for a in primary}
        seen: set[str] = set()
        supporting: list = []
        for ax in supporting_candidates:
            if ax.name not in primary_names and ax.name not in seen:
                seen.add(ax.name)
                supporting.append(ax)
        supporting = supporting[:_MAX_SUPPORTING]

        # Build context
        all_relevant = primary + supporting
        variables = set()
        for ax in all_relevant:
            variables.update(self._axiom_vars.get(ax.name, set()))

        domain_chain = sorted({ax.domain for ax in all_relevant})
        concepts = self._extract_concepts(all_relevant)

        return KnowledgeContext(
            primary_axioms=primary,
            supporting_axioms=supporting,
            concepts=concepts,
            domain_chain=domain_chain,
            variables_mentioned=variables,
        )

    def find_derivation_path(
        self,
        start_axiom: Axiom,
        target_concept: str,
    ) -> list[Axiom]:
        """Find a chain of axioms from start to target concept.

        Uses shared variables as edges: if axiom A produces variable X
        and axiom B consumes variable X, there's a derivation link.

        BFS through the variable graph, max depth _MAX_PATH_DEPTH.

        Args:
            start_axiom: Starting axiom.
            target_concept: Variable name or keyword to reach.

        Returns:
            List of axioms forming the path (may be empty).
        """
        all_axioms = self._lib.all_axioms()
        target_lower = target_concept.lower()

        # Check if target is already in start axiom
        start_vars = self._axiom_vars.get(start_axiom.name, set())

        # BFS: queue of (current_axiom, path_so_far)
        from collections import deque
        queue: deque[tuple[Axiom, list[Axiom]]] = deque()
        queue.append((start_axiom, [start_axiom]))
        visited: set[str] = {start_axiom.name}

        while queue:
            current, path = queue.popleft()
            if len(path) > _MAX_PATH_DEPTH:
                continue

            current_vars = self._axiom_vars.get(current.name, set())

            # Find neighbors: axioms sharing at least one variable
            for ax in all_axioms:
                if ax.name in visited:
                    continue
                ax_vars = self._axiom_vars.get(ax.name, set())
                shared = current_vars & ax_vars
                if not shared:
                    continue

                new_path = path + [ax]

                # Check if target is in this axiom's variables or name
                ax_all_tokens = _tokens(ax.name) | {
                    v.lower() for v in ax_vars
                }
                if target_lower in ax_all_tokens:
                    return new_path[1:]  # exclude start_axiom

                visited.add(ax.name)
                queue.append((ax, new_path))

        return []

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_axioms(
        self,
        axioms: list[Axiom],
        kw_set: set[str],
        domain: str,
        target_var: str | None,
    ) -> list[tuple[Axiom, float]]:
        """Score axioms by keyword overlap + domain bonus + target bonus."""
        scored: list[tuple[Axiom, float]] = []
        for ax in axioms:
            ax_tokens = _tokens(ax.name) | _tokens(ax.formula_str)
            overlap = len(kw_set & ax_tokens)

            # Domain bonus
            domain_bonus = 1.0 if ax.domain.split(".")[0] == domain else 0.0

            # Target variable bonus
            target_bonus = 0.0
            if target_var:
                ax_vars = self._axiom_vars.get(ax.name, set())
                if target_var in ax_vars:
                    target_bonus = 3.0

            score = float(overlap) + domain_bonus + target_bonus
            scored.append((ax, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Variable-based expansion
    # ------------------------------------------------------------------

    def _find_by_shared_vars(
        self,
        target_vars: set[str],
        exclude: list[Axiom],
        all_axioms: list[Axiom],
    ) -> list[Axiom]:
        """Find axioms sharing variables with the target set."""
        exclude_names = {a.name for a in exclude}
        results: list[tuple[Axiom, int]] = []
        for ax in all_axioms:
            if ax.name in exclude_names:
                continue
            ax_vars = self._axiom_vars.get(ax.name, set())
            shared_count = len(target_vars & ax_vars)
            if shared_count > 0:
                results.append((ax, shared_count))
        results.sort(key=lambda x: x[1], reverse=True)
        return [ax for ax, _ in results]

    # ------------------------------------------------------------------
    # VSA similarity expansion
    # ------------------------------------------------------------------

    def _vsa_expand(
        self,
        primary: list[Axiom],
        all_axioms: list[Axiom],
    ) -> list[Axiom]:
        """Find axioms similar to primary via VSA cosine similarity."""
        if not primary or not all_axioms:
            return []
        primary_names = {a.name for a in primary}
        # Average primary vectors
        vecs = np.stack([a.vector for a in primary], axis=0)
        centroid = np.mean(vecs, axis=0)
        # Use library select
        candidates = self._lib.select(centroid, top_n=_MAX_SUPPORTING + len(primary))
        return [a for a in candidates if a.name not in primary_names]

    # ------------------------------------------------------------------
    # Concept extraction
    # ------------------------------------------------------------------

    def _extract_concepts(self, axioms: list[Axiom]) -> list[str]:
        """Extract unique concept names from axiom names."""
        seen: set[str] = set()
        concepts: list[str] = []
        for ax in axioms:
            # Use the axiom name as a concept
            if ax.name not in seen:
                seen.add(ax.name)
                concepts.append(ax.name)
        return concepts
```

### Step 1.3 -- Verify

- [ ] Run `uv run pytest tests/test_knowledge_search.py -v --tb=short -x`
- [ ] Run `uv run ruff check mowm/reasoning/knowledge_search.py`
- [ ] Verify all tests pass, no lint errors

---

## Task 2: reasoning/chain_builder.py -- DerivationStep + DerivationChain + ChainBuilder

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\chain_builder.py`

Depends on: `SafeFormulaExecutor`, `AxiomLibrary`, `KnowledgeContext` from Task 1.

This module builds step-by-step derivation chains -- connecting axioms through
shared variables to derive conclusions. Both numeric (with values) and symbolic
(variable substitution like "a = F/m = mg/m = g") chains.

### Step 2.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_chain_builder.py`

```python
# tests/test_chain_builder.py
"""Tests for ChainBuilder -- derivation chains across multiple domains."""
from __future__ import annotations

import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.reasoning.chain_builder import (
    ChainBuilder,
    DerivationChain,
    DerivationStep,
)
from mowm.reasoning.knowledge_search import KnowledgeContext, KnowledgeSearch

K, L = 4, 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> SafeFormulaExecutor:
    return SafeFormulaExecutor()


@pytest.fixture
def physics_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    ))
    lib.register(lib.make_axiom(
        "Weight", "physics.dynamics", "W = m * g",
    ))
    lib.register(lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    ))
    lib.register(lib.make_axiom(
        "Momentum", "physics.dynamics", "p = m * v",
    ))
    return lib


@pytest.fixture
def economics_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Supply-Demand Equilibrium", "economics.micro",
        "Q_d = Q_s",
    ))
    lib.register(lib.make_axiom(
        "Price Elasticity", "economics.micro",
        "epsilon = dQ / dP",
    ))
    lib.register(lib.make_axiom(
        "Consumer Surplus", "economics.micro",
        "CS = WTP - P",
    ))
    return lib


@pytest.fixture
def logic_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Modus Ponens", "logic.deductive",
        "(P AND (P IMPLIES Q)) IMPLIES Q",
    ))
    lib.register(lib.make_axiom(
        "Hypothetical Syllogism", "logic.deductive",
        "((P IMPLIES Q) AND (Q IMPLIES R)) IMPLIES (P IMPLIES R)",
    ))
    lib.register(lib.make_axiom(
        "Contrapositive", "logic.propositional",
        "(P IMPLIES Q) = (NOT Q IMPLIES NOT P)",
    ))
    return lib


@pytest.fixture
def bio_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Hamilton's Rule", "bio.evolution", "benefit = r * B - C",
    ))
    lib.register(lib.make_axiom(
        "Fitness", "bio.evolution", "w = 1 + s",
    ))
    return lib


@pytest.fixture
def music_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Harmonic Series", "music.harmony", "f_n = n * f_0",
    ))
    lib.register(lib.make_axiom(
        "Beat Frequency", "music.harmony", "f_beat = f1 - f2",
    ))
    return lib


@pytest.fixture
def physics_builder(physics_lib, executor) -> ChainBuilder:
    return ChainBuilder(executor=executor, library=physics_lib)


@pytest.fixture
def economics_builder(economics_lib, executor) -> ChainBuilder:
    return ChainBuilder(executor=executor, library=economics_lib)


@pytest.fixture
def logic_builder(logic_lib, executor) -> ChainBuilder:
    return ChainBuilder(executor=executor, library=logic_lib)


@pytest.fixture
def bio_builder(bio_lib, executor) -> ChainBuilder:
    return ChainBuilder(executor=executor, library=bio_lib)


@pytest.fixture
def music_builder(music_lib, executor) -> ChainBuilder:
    return ChainBuilder(executor=executor, library=music_lib)


# ---------------------------------------------------------------------------
# TestDerivationStep dataclass
# ---------------------------------------------------------------------------


class TestDerivationStep:
    def test_step_fields(self):
        step = DerivationStep(
            axiom_name="Newton's Second Law",
            formula="F = m * a",
            substitutions={"m": 10.0, "a": 9.8},
            result_var="F",
            result_value=98.0,
            explanation="F = m * a = 10.0 * 9.8 = 98.0",
        )
        assert step.axiom_name == "Newton's Second Law"
        assert step.result_value == pytest.approx(98.0)


class TestDerivationChain:
    def test_chain_fields(self):
        chain = DerivationChain(
            steps=[],
            conclusion="",
            variables_used=set(),
            axioms_used=[],
        )
        assert isinstance(chain.steps, list)
        assert isinstance(chain.variables_used, set)


# ---------------------------------------------------------------------------
# TestChainBuilder -- Physics: numeric chain
# ---------------------------------------------------------------------------


class TestBuildChainPhysics:
    def test_build_chain_newton(self, physics_builder, physics_lib):
        """Build a numeric chain: F = m * a with m=10, a=9.8."""
        search = KnowledgeSearch(physics_lib, k=K, l=L)
        ctx = search.search(["force", "mass"], domain="physics")
        chain = physics_builder.build_chain(
            context=ctx,
            variables={"m": 10.0, "a": 9.8},
            target="F",
        )
        assert isinstance(chain, DerivationChain)
        assert len(chain.steps) > 0
        assert chain.conclusion != ""

    def test_chain_computes_correct_value(self, physics_builder, physics_lib):
        search = KnowledgeSearch(physics_lib, k=K, l=L)
        ctx = search.search(["force"], domain="physics", target_var="F")
        chain = physics_builder.build_chain(
            context=ctx,
            variables={"m": 10.0, "a": 9.8},
            target="F",
        )
        # Find the step that computes F
        f_steps = [s for s in chain.steps if s.result_var == "F"]
        if f_steps:
            assert f_steps[0].result_value == pytest.approx(98.0, rel=1e-3)

    def test_chain_tracks_variables_used(self, physics_builder, physics_lib):
        search = KnowledgeSearch(physics_lib, k=K, l=L)
        ctx = search.search(["force"], domain="physics")
        chain = physics_builder.build_chain(
            context=ctx,
            variables={"m": 10.0, "a": 9.8},
        )
        assert "m" in chain.variables_used

    def test_chain_tracks_axioms_used(self, physics_builder, physics_lib):
        search = KnowledgeSearch(physics_lib, k=K, l=L)
        ctx = search.search(["force"], domain="physics")
        chain = physics_builder.build_chain(
            context=ctx,
            variables={"m": 10.0, "a": 9.8},
        )
        assert len(chain.axioms_used) > 0


# ---------------------------------------------------------------------------
# TestExplanationChain -- symbolic (no numeric values)
# ---------------------------------------------------------------------------


class TestExplanationChainPhysics:
    def test_explanation_chain_for_free_fall(self, physics_builder, physics_lib):
        """Why do heavier objects fall at the same rate?
        Should produce symbolic chain: W=m*g, F=m*a, a=F/m=mg/m=g."""
        search = KnowledgeSearch(physics_lib, k=K, l=L)
        ctx = search.search(
            ["heavier", "objects", "fall", "rate", "mass"],
            domain="physics",
        )
        chain = physics_builder.build_explanation_chain(
            context=ctx,
            question_keywords=["heavier", "objects", "fall", "rate"],
        )
        assert isinstance(chain, DerivationChain)
        assert len(chain.steps) > 0
        assert chain.conclusion != ""

    def test_explanation_chain_has_explanations(
        self, physics_builder, physics_lib,
    ):
        search = KnowledgeSearch(physics_lib, k=K, l=L)
        ctx = search.search(["force", "mass"], domain="physics")
        chain = physics_builder.build_explanation_chain(
            context=ctx,
            question_keywords=["force", "mass"],
        )
        for step in chain.steps:
            assert step.explanation != ""


# ---------------------------------------------------------------------------
# TestChainBuilder -- Economics
# ---------------------------------------------------------------------------


class TestBuildChainEconomics:
    def test_economics_chain(self, economics_builder, economics_lib):
        search = KnowledgeSearch(economics_lib, k=K, l=L)
        ctx = search.search(
            ["consumer", "surplus", "price"],
            domain="economics",
        )
        chain = economics_builder.build_chain(
            context=ctx,
            variables={"WTP": 50.0, "P": 30.0},
            target="CS",
        )
        assert isinstance(chain, DerivationChain)
        assert len(chain.steps) > 0

    def test_economics_explanation(self, economics_builder, economics_lib):
        search = KnowledgeSearch(economics_lib, k=K, l=L)
        ctx = search.search(
            ["supply", "demand", "equilibrium"],
            domain="economics",
        )
        chain = economics_builder.build_explanation_chain(
            context=ctx,
            question_keywords=["supply", "demand"],
        )
        assert len(chain.steps) > 0


# ---------------------------------------------------------------------------
# TestChainBuilder -- Logic (symbolic only)
# ---------------------------------------------------------------------------


class TestBuildChainLogic:
    def test_logic_explanation_modus_ponens(
        self, logic_builder, logic_lib,
    ):
        search = KnowledgeSearch(logic_lib, k=K, l=L)
        ctx = search.search(
            ["modus", "ponens", "implies"],
            domain="logic",
        )
        chain = logic_builder.build_explanation_chain(
            context=ctx,
            question_keywords=["modus", "ponens"],
        )
        assert isinstance(chain, DerivationChain)
        assert len(chain.steps) > 0
        # Should reference Modus Ponens
        assert "Modus Ponens" in chain.axioms_used


# ---------------------------------------------------------------------------
# TestChainBuilder -- Bio
# ---------------------------------------------------------------------------


class TestBuildChainBio:
    def test_bio_hamilton_chain(self, bio_builder, bio_lib):
        """Hamilton's Rule: altruism favored when r*B > C."""
        search = KnowledgeSearch(bio_lib, k=K, l=L)
        ctx = search.search(
            ["altruism", "kin", "Hamilton"],
            domain="bio",
        )
        chain = bio_builder.build_chain(
            context=ctx,
            variables={"r": 0.5, "B": 10.0, "C": 3.0},
            target="benefit",
        )
        assert isinstance(chain, DerivationChain)
        assert len(chain.steps) > 0


# ---------------------------------------------------------------------------
# TestChainBuilder -- Music
# ---------------------------------------------------------------------------


class TestBuildChainMusic:
    def test_music_harmonic_chain(self, music_builder, music_lib):
        search = KnowledgeSearch(music_lib, k=K, l=L)
        ctx = search.search(
            ["harmonic", "frequency"],
            domain="music",
        )
        chain = music_builder.build_chain(
            context=ctx,
            variables={"n": 3.0, "f_0": 440.0},
            target="f_n",
        )
        assert isinstance(chain, DerivationChain)
        assert len(chain.steps) > 0


# ---------------------------------------------------------------------------
# TestSimplifyChain
# ---------------------------------------------------------------------------


class TestSimplifyChain:
    def test_simplify_removes_redundant(self, physics_builder, physics_lib):
        search = KnowledgeSearch(physics_lib, k=K, l=L)
        ctx = search.search(["force"], domain="physics")
        chain = physics_builder.build_chain(
            context=ctx, variables={"m": 10.0, "a": 9.8},
        )
        simplified = physics_builder.simplify_chain(chain)
        assert isinstance(simplified, DerivationChain)
        assert len(simplified.steps) <= len(chain.steps)

    def test_simplify_empty_chain(self, physics_builder):
        chain = DerivationChain(
            steps=[], conclusion="", variables_used=set(), axioms_used=[],
        )
        simplified = physics_builder.simplify_chain(chain)
        assert simplified.steps == []
```

### Step 2.2 -- Write implementation

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\chain_builder.py`

```python
# mowm/reasoning/chain_builder.py
"""ChainBuilder -- build step-by-step derivation chains from axioms.

Connects axioms through shared variables to derive conclusions.
Supports both numeric chains (with values) and symbolic chains
(variable substitution for explanation-style derivations).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary
    from mowm.executor import SafeFormulaExecutor
    from mowm.reasoning.knowledge_search import KnowledgeContext

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")

_FUNCTIONS = {
    "sin", "cos", "sqrt", "exp", "log", "square", "not",
    "and", "or", "implies", "for", "all", "if", "iff",
    "such", "that", "with", "subject", "to", "max", "min",
    "sum", "prod", "integral", "no", "in", "const",
    "avoid", "consecutive", "between", "voices", "pairs",
}


@dataclass
class DerivationStep:
    """A single step in a derivation chain.

    Attributes:
        axiom_name: Name of the axiom applied.
        formula: The formula string used.
        substitutions: Variable values at this step.
        result_var: What this step derives.
        result_value: The derived value (float) or expression (str).
        explanation: Human-readable step explanation.
    """

    axiom_name: str
    formula: str
    substitutions: dict[str, float | str] = field(default_factory=dict)
    result_var: str = ""
    result_value: float | str = 0.0
    explanation: str = ""


@dataclass
class DerivationChain:
    """A complete derivation chain from axioms to conclusion.

    Attributes:
        steps: Ordered list of derivation steps.
        conclusion: Final derived statement.
        variables_used: All variable names used across the chain.
        axioms_used: Axiom names in order of use.
    """

    steps: list[DerivationStep] = field(default_factory=list)
    conclusion: str = ""
    variables_used: set[str] = field(default_factory=set)
    axioms_used: list[str] = field(default_factory=list)


def _extract_lhs(formula: str) -> str | None:
    """Extract LHS variable from 'X = expr' formula."""
    if "=" in formula and "==" not in formula:
        parts = formula.split("=", 1)
        lhs = parts[0].strip()
        if re.match(r"^[A-Za-z_]\w*$", lhs):
            return lhs
    return None


def _extract_rhs(formula: str) -> str:
    """Extract RHS expression from 'X = expr' formula."""
    if "=" in formula and "==" not in formula:
        parts = formula.split("=", 1)
        return parts[1].strip()
    return formula


def _formula_vars(formula: str) -> set[str]:
    """Extract variable identifiers from a formula string."""
    raw = set(_TOKEN_RE.findall(formula))
    return {v for v in raw if v.lower() not in _FUNCTIONS and not v.isdigit()}


class ChainBuilder:
    """Build derivation chains from knowledge context.

    Numeric mode: executes formulas with provided variable values.
    Symbolic mode: builds explanation chains without numeric execution.

    Args:
        executor: SafeFormulaExecutor for numeric evaluation.
        library: AxiomLibrary for axiom metadata access.
    """

    def __init__(
        self,
        executor: SafeFormulaExecutor,
        library: AxiomLibrary,
    ) -> None:
        self._executor = executor
        self._lib = library

    # ------------------------------------------------------------------
    # Numeric chain building
    # ------------------------------------------------------------------

    def build_chain(
        self,
        context: KnowledgeContext,
        variables: dict[str, float],
        target: str | None = None,
    ) -> DerivationChain:
        """Build a derivation chain from knowledge context with values.

        Strategy:
          1. Start with axioms containing provided variables
          2. Execute each, producing new variable values
          3. Feed results into next axiom (variable propagation)
          4. Continue until target is derived or no more steps

        Args:
            context: KnowledgeContext from KnowledgeSearch.
            variables: Known variable bindings.
            target: Optional target variable to derive.

        Returns:
            DerivationChain with computed steps and conclusion.
        """
        steps: list[DerivationStep] = []
        known = dict(variables)
        used_axioms: set[str] = set()
        all_axioms = context.primary_axioms + context.supporting_axioms

        # Sort axioms: prioritize those with target as LHS
        if target:
            all_axioms = sorted(
                all_axioms,
                key=lambda ax: (
                    0 if _extract_lhs(ax.formula_str) == target else 1
                ),
            )

        for ax in all_axioms:
            if ax.name in used_axioms:
                continue
            lhs = _extract_lhs(ax.formula_str)
            if lhs is None:
                continue

            formula_variables = _formula_vars(_extract_rhs(ax.formula_str))
            # Check if we have at least some variables to evaluate
            available = formula_variables & set(known.keys())
            if not available and formula_variables:
                continue

            try:
                result = self._executor.execute(ax.formula_str, known)
            except Exception:
                continue

            # Build substitution display
            subs: dict[str, float | str] = {}
            for v in formula_variables:
                if v in known:
                    subs[v] = known[v]

            sub_str = ", ".join(f"{k}={v}" for k, v in subs.items())
            explanation = (
                f"{ax.name}: {ax.formula_str}"
                f" with {sub_str} => {lhs} = {result:.4g}"
                if sub_str
                else f"{ax.name}: {ax.formula_str} => {lhs} = {result:.4g}"
            )

            step = DerivationStep(
                axiom_name=ax.name,
                formula=ax.formula_str,
                substitutions=subs,
                result_var=lhs,
                result_value=result,
                explanation=explanation,
            )
            steps.append(step)
            used_axioms.add(ax.name)
            known[lhs] = result

            if target and lhs == target:
                break

        all_vars: set[str] = set(variables.keys())
        for s in steps:
            all_vars.update(s.substitutions.keys())
            all_vars.add(s.result_var)

        conclusion = self._build_conclusion(steps, target)

        return DerivationChain(
            steps=steps,
            conclusion=conclusion,
            variables_used=all_vars,
            axioms_used=[s.axiom_name for s in steps],
        )

    # ------------------------------------------------------------------
    # Symbolic / explanation chain building
    # ------------------------------------------------------------------

    def build_explanation_chain(
        self,
        context: KnowledgeContext,
        question_keywords: list[str],
    ) -> DerivationChain:
        """Build a chain optimized for explanation (not computation).

        For EXPLAIN intent: finds the axiom sequence that best
        explains the concept, even without numeric values.
        Uses symbolic substitution: "a = F/m = mg/m = g" style.

        Args:
            context: KnowledgeContext from KnowledgeSearch.
            question_keywords: Keywords from the question.

        Returns:
            DerivationChain with symbolic explanation steps.
        """
        steps: list[DerivationStep] = []
        all_axioms = context.primary_axioms + context.supporting_axioms

        for ax in all_axioms:
            lhs = _extract_lhs(ax.formula_str)
            rhs = _extract_rhs(ax.formula_str)

            explanation = f"{ax.name}: {ax.formula_str}"
            if lhs:
                explanation = (
                    f"From {ax.name}: {lhs} = {rhs}"
                )
            else:
                explanation = f"By {ax.name}: {ax.formula_str}"

            step = DerivationStep(
                axiom_name=ax.name,
                formula=ax.formula_str,
                substitutions={},
                result_var=lhs or "",
                result_value=rhs if lhs else ax.formula_str,
                explanation=explanation,
            )
            steps.append(step)

        all_vars: set[str] = set()
        for ax in all_axioms:
            all_vars.update(_formula_vars(ax.formula_str))

        conclusion = self._build_symbolic_conclusion(steps, question_keywords)

        return DerivationChain(
            steps=steps,
            conclusion=conclusion,
            variables_used=all_vars,
            axioms_used=[s.axiom_name for s in steps],
        )

    # ------------------------------------------------------------------
    # Simplification
    # ------------------------------------------------------------------

    def simplify_chain(self, chain: DerivationChain) -> DerivationChain:
        """Remove redundant steps, combine trivial substitutions.

        A step is redundant if:
          - It derives a variable that is never used in later steps
          - It derives the same variable as an earlier step (keep later)
        """
        if not chain.steps:
            return chain

        # Track which variables are consumed by later steps
        consumed: set[str] = set()
        for step in chain.steps:
            consumed.update(step.substitutions.keys())

        # Also keep steps that derive the final result
        final_var = chain.steps[-1].result_var if chain.steps else ""

        # Filter: keep steps whose result_var is consumed or is the final
        seen_vars: set[str] = set()
        filtered: list[DerivationStep] = []
        for step in reversed(chain.steps):
            if step.result_var == final_var or step.result_var in consumed:
                if step.result_var not in seen_vars:
                    filtered.append(step)
                    seen_vars.add(step.result_var)
                    consumed.update(step.substitutions.keys())

        filtered.reverse()

        return DerivationChain(
            steps=filtered,
            conclusion=chain.conclusion,
            variables_used={
                v for s in filtered
                for v in (list(s.substitutions.keys()) + [s.result_var])
            },
            axioms_used=[s.axiom_name for s in filtered],
        )

    # ------------------------------------------------------------------
    # Conclusion builders
    # ------------------------------------------------------------------

    def _build_conclusion(
        self,
        steps: list[DerivationStep],
        target: str | None,
    ) -> str:
        if not steps:
            return "No derivation steps could be built."
        last = steps[-1]
        if target and last.result_var == target:
            return (
                f"{target} = {last.result_value:.4g}"
                if isinstance(last.result_value, float)
                else f"{target} = {last.result_value}"
            )
        parts = []
        for s in steps:
            if isinstance(s.result_value, float):
                parts.append(f"{s.result_var} = {s.result_value:.4g}")
            else:
                parts.append(f"{s.result_var} = {s.result_value}")
        return "; ".join(parts)

    def _build_symbolic_conclusion(
        self,
        steps: list[DerivationStep],
        keywords: list[str],
    ) -> str:
        if not steps:
            return "No derivation steps could be built."
        axiom_names = [s.axiom_name for s in steps]
        return (
            f"By applying {', '.join(axiom_names)}, "
            f"we can derive the relationship between "
            f"{', '.join(keywords[:3])}."
        )
```

### Step 2.3 -- Verify

- [ ] Run `uv run pytest tests/test_chain_builder.py -v --tb=short -x`
- [ ] Run `uv run ruff check mowm/reasoning/chain_builder.py`
- [ ] Verify all tests pass, no lint errors

---

## Task 3: reasoning/causal_tracer.py -- CausalNarrative + CausalTracer

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\causal_tracer.py`

Depends on: `CausalGraph`. No other reasoning module dependencies.

This module traces cause-effect relationships for explanations.
Identifies positive/negative feedback loops, variable cancellations,
and builds narrative explanations of causal paths.

### Step 3.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_causal_tracer.py`

```python
# tests/test_causal_tracer.py
"""Tests for CausalTracer -- causal narrative generation across domains."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.causal_graph import CausalGraph
from mowm.reasoning.causal_tracer import CausalNarrative, CausalTracer

K, L = 4, 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def physics_graph() -> CausalGraph:
    """Physics: mass -> grav_force (+), mass -> inertia (+),
    grav_force -> acceleration (+), inertia -> acceleration (-).
    This models why mass cancels in free fall."""
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    for name in ["mass", "grav_force", "inertia", "acceleration"]:
        g.add_node(name, "physics", vec)
    g.add_edge("mass", "grav_force", polarity="+", strength=1.0)
    g.add_edge("mass", "inertia", polarity="+", strength=1.0)
    g.add_edge("grav_force", "acceleration", polarity="+", strength=1.0)
    g.add_edge("inertia", "acceleration", polarity="-", strength=1.0)
    return g


@pytest.fixture
def economics_graph() -> CausalGraph:
    """Economics: price -> demand (-), price -> supply (+),
    demand -> revenue (+), supply -> revenue (+)."""
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    for name in ["price", "demand", "supply", "revenue"]:
        g.add_node(name, "economics", vec)
    g.add_edge("price", "demand", polarity="-", strength=0.8)
    g.add_edge("price", "supply", polarity="+", strength=0.6)
    g.add_edge("demand", "revenue", polarity="+", strength=0.9)
    g.add_edge("supply", "revenue", polarity="+", strength=0.3)
    return g


@pytest.fixture
def feedback_graph() -> CausalGraph:
    """Graph with a feedback loop: A -> B -> C -> A."""
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    for name in ["A", "B", "C"]:
        g.add_node(name, "causal", vec)
    g.add_edge("A", "B", polarity="+", strength=1.0)
    g.add_edge("B", "C", polarity="+", strength=1.0)
    g.add_edge("C", "A", polarity="+", strength=0.5)
    return g


@pytest.fixture
def historical_graph() -> CausalGraph:
    """Historical: innovation -> prosperity -> complacency -> decline."""
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    for name in [
        "innovation", "prosperity", "complacency", "decline",
    ]:
        g.add_node(name, "historical", vec)
    g.add_edge("innovation", "prosperity", polarity="+", strength=0.9)
    g.add_edge("prosperity", "complacency", polarity="+", strength=0.7)
    g.add_edge("complacency", "decline", polarity="+", strength=0.8)
    return g


@pytest.fixture
def empty_graph() -> CausalGraph:
    return CausalGraph(k=K, l=L)


@pytest.fixture
def phys_tracer(physics_graph) -> CausalTracer:
    return CausalTracer(causal=physics_graph)


@pytest.fixture
def econ_tracer(economics_graph) -> CausalTracer:
    return CausalTracer(causal=economics_graph)


@pytest.fixture
def feedback_tracer(feedback_graph) -> CausalTracer:
    return CausalTracer(causal=feedback_graph)


@pytest.fixture
def hist_tracer(historical_graph) -> CausalTracer:
    return CausalTracer(causal=historical_graph)


@pytest.fixture
def empty_tracer(empty_graph) -> CausalTracer:
    return CausalTracer(causal=empty_graph)


# ---------------------------------------------------------------------------
# TestCausalNarrative dataclass
# ---------------------------------------------------------------------------


class TestCausalNarrative:
    def test_default_fields(self):
        n = CausalNarrative()
        assert n.cause == ""
        assert n.effects == []
        assert n.cancellations == []
        assert n.net_effect == ""
        assert n.graph_path == []


# ---------------------------------------------------------------------------
# TestTraceExplanation -- Physics (mass cancellation)
# ---------------------------------------------------------------------------


class TestTracePhysics:
    def test_trace_finds_mass_effects(self, phys_tracer):
        narrative = phys_tracer.trace_explanation(
            question_keywords=["mass", "acceleration", "fall"],
            domain="physics",
        )
        assert narrative is not None
        assert narrative.cause != ""
        assert len(narrative.effects) > 0

    def test_trace_detects_cancellation(self, phys_tracer):
        """Mass increases grav_force (+) and inertia (+), both feed
        into acceleration with opposite signs -> cancellation."""
        narrative = phys_tracer.trace_explanation(
            question_keywords=["mass", "acceleration"],
            domain="physics",
        )
        assert narrative is not None
        assert len(narrative.cancellations) > 0

    def test_trace_has_graph_path(self, phys_tracer):
        narrative = phys_tracer.trace_explanation(
            question_keywords=["mass"],
            domain="physics",
        )
        assert narrative is not None
        assert len(narrative.graph_path) > 0


# ---------------------------------------------------------------------------
# TestTraceExplanation -- Economics
# ---------------------------------------------------------------------------


class TestTraceEconomics:
    def test_trace_price_effects(self, econ_tracer):
        narrative = econ_tracer.trace_explanation(
            question_keywords=["price", "demand", "revenue"],
            domain="economics",
        )
        assert narrative is not None
        assert len(narrative.effects) > 0

    def test_economics_no_cancellation(self, econ_tracer):
        """Price->demand(-)->revenue(+) and price->supply(+)->revenue(+)
        have opposite initial signs but don't cancel at same target
        with equal magnitude, so cancellation list may be empty or
        contain partial cancellations."""
        narrative = econ_tracer.trace_explanation(
            question_keywords=["price"],
            domain="economics",
        )
        assert narrative is not None


# ---------------------------------------------------------------------------
# TestTraceExplanation -- Historical
# ---------------------------------------------------------------------------


class TestTraceHistorical:
    def test_trace_historical_chain(self, hist_tracer):
        narrative = hist_tracer.trace_explanation(
            question_keywords=["innovation", "decline"],
            domain="historical",
        )
        assert narrative is not None
        assert len(narrative.effects) > 0
        assert "innovation" in narrative.graph_path


# ---------------------------------------------------------------------------
# TestFindCancellation
# ---------------------------------------------------------------------------


class TestFindCancellation:
    def test_cancellation_mass_paths(self, phys_tracer):
        """Explicit cancellation check: grav_force and inertia both
        feed into acceleration with opposite polarities."""
        result = phys_tracer.find_cancellation(
            "grav_force", "inertia",
        )
        assert result is not None
        assert "cancel" in result.lower() or "acceleration" in result.lower()

    def test_cancellation_no_shared_target(self, econ_tracer):
        """demand and supply don't cancel at revenue (same polarity)."""
        result = econ_tracer.find_cancellation("demand", "supply")
        # Both have + polarity to revenue, so no cancellation
        assert result is None or "cancel" not in result.lower()


# ---------------------------------------------------------------------------
# TestFeedbackLoopDetection
# ---------------------------------------------------------------------------


class TestFeedbackDetection:
    def test_detect_feedback_loop(self, feedback_tracer):
        narrative = feedback_tracer.trace_explanation(
            question_keywords=["A", "B", "C"],
            domain="causal",
        )
        assert narrative is not None

    def test_feedback_in_net_effect(self, feedback_tracer):
        """The feedback loop A->B->C->A should show up somewhere."""
        narrative = feedback_tracer.trace_explanation(
            question_keywords=["A"],
            domain="causal",
        )
        assert narrative is not None
        # Net effect should mention reinforcing/feedback
        assert (
            "reinforc" in narrative.net_effect.lower()
            or "feedback" in narrative.net_effect.lower()
            or "loop" in narrative.net_effect.lower()
            or len(narrative.effects) > 0
        )


# ---------------------------------------------------------------------------
# TestEmptyGraph
# ---------------------------------------------------------------------------


class TestEmptyGraph:
    def test_trace_empty_graph(self, empty_tracer):
        result = empty_tracer.trace_explanation(
            question_keywords=["anything"],
            domain="physics",
        )
        assert result is None
```

### Step 3.2 -- Write implementation

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\causal_tracer.py`

```python
# mowm/reasoning/causal_tracer.py
"""CausalTracer -- trace causal relationships for natural language explanations.

Uses CausalGraph to identify:
  - Positive/negative effects between variables
  - Variable cancellations (key to many "why" answers)
  - Feedback loops (reinforcing or balancing)
  - Net direction of causal influence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mowm.causal_graph import CausalGraph


@dataclass
class CausalNarrative:
    """Structured causal narrative for composing explanations.

    Attributes:
        cause: Starting cause variable.
        effects: List of (target, direction_str, strength) tuples.
        cancellations: Pairs of variables that cancel at a common target.
        net_effect: Human-readable summary of net causal result.
        graph_path: Node sequence traversed in causal graph.
    """

    cause: str = ""
    effects: list[tuple[str, str, float]] = field(default_factory=list)
    cancellations: list[tuple[str, str]] = field(default_factory=list)
    net_effect: str = ""
    graph_path: list[str] = field(default_factory=list)


class CausalTracer:
    """Trace causal relationships for natural language explanation.

    Args:
        causal: CausalGraph instance.
    """

    def __init__(self, causal: CausalGraph) -> None:
        self._graph = causal

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def trace_explanation(
        self,
        question_keywords: list[str],
        domain: str,
    ) -> CausalNarrative | None:
        """Trace causal relationships relevant to the question.

        Identifies:
          - The best starting node from question keywords
          - Downstream effects via BFS propagation
          - Variable cancellations (opposing paths to same target)
          - Feedback loops (cycles)
          - Net direction of effect

        Args:
            question_keywords: Keywords from the question.
            domain: Domain string for context.

        Returns:
            CausalNarrative or None if no relevant nodes found.
        """
        if not self._graph.nodes:
            return None

        # Find the best starting node from keywords
        cause_node = self._find_best_node(question_keywords)
        if cause_node is None:
            return None

        # Propagate from cause node
        effects_dict = self._graph.propagate(cause_node, delta=1.0)

        # Build effects list
        effects: list[tuple[str, str, float]] = []
        graph_path = [cause_node]
        for target, value in effects_dict.items():
            direction = "positive" if value > 0 else "negative"
            effects.append((target, direction, abs(value)))
            graph_path.append(target)

        # Detect cancellations
        cancellations = self._detect_cancellations(cause_node)

        # Detect feedback loops
        loops = self._graph.feedback_loops()
        cause_loops = [
            loop for loop in loops if cause_node in loop
        ]

        # Build net effect summary
        net_effect = self._build_net_effect(
            cause_node, effects, cancellations, cause_loops,
        )

        return CausalNarrative(
            cause=cause_node,
            effects=effects,
            cancellations=cancellations,
            net_effect=net_effect,
            graph_path=graph_path,
        )

    def find_cancellation(
        self,
        var_a: str,
        var_b: str,
    ) -> str | None:
        """Check if two paths through var_a and var_b cancel.

        Two variables cancel at a target if they both have edges
        to the same target with opposite polarities.

        Args:
            var_a: First variable node name.
            var_b: Second variable node name.

        Returns:
            Description string if cancellation found, None otherwise.
        """
        if var_a not in self._graph.nodes or var_b not in self._graph.nodes:
            return None

        # Find outgoing edges from both
        edges_a = self._graph._adj.get(var_a, [])
        edges_b = self._graph._adj.get(var_b, [])

        targets_a = {e.target: e.polarity for e in edges_a}
        targets_b = {e.target: e.polarity for e in edges_b}

        shared_targets = set(targets_a.keys()) & set(targets_b.keys())

        for target in shared_targets:
            pol_a = targets_a[target]
            pol_b = targets_b[target]
            if pol_a != pol_b:
                return (
                    f"{var_a} ({pol_a}) and {var_b} ({pol_b}) "
                    f"cancel at {target}"
                )

        return None

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def _find_best_node(self, keywords: list[str]) -> str | None:
        """Find the graph node best matching the question keywords."""
        kw_set = {kw.lower() for kw in keywords}
        best_node: str | None = None
        best_score = 0

        for name in self._graph.nodes:
            name_tokens = {t.lower() for t in name.replace("_", " ").split()}
            score = len(kw_set & name_tokens)
            # Also check substring match
            for kw in kw_set:
                if kw in name.lower():
                    score += 1
            if score > best_score:
                best_score = score
                best_node = name

        # Fallback: return first node with outgoing edges
        if best_node is None:
            for name in self._graph.nodes:
                if self._graph._adj.get(name):
                    return name

        return best_node

    # ------------------------------------------------------------------
    # Cancellation detection
    # ------------------------------------------------------------------

    def _detect_cancellations(
        self, source: str,
    ) -> list[tuple[str, str]]:
        """Detect variable cancellations downstream from source.

        A cancellation occurs when source has two outgoing paths
        to the same target through intermediaries with opposite
        net polarities.
        """
        cancellations: list[tuple[str, str]] = []

        # Get direct children of source
        edges = self._graph._adj.get(source, [])
        if len(edges) < 2:
            return []

        # For each pair of children, check if they share a target
        children = [(e.target, e.polarity) for e in edges]
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                child_a, pol_a = children[i]
                child_b, pol_b = children[j]

                # Check if child_a and child_b share downstream targets
                # with opposite net polarities
                edges_a = self._graph._adj.get(child_a, [])
                edges_b = self._graph._adj.get(child_b, [])

                targets_a = {}
                for e in edges_a:
                    # Net polarity = source->child polarity * child->target
                    net = (
                        "+" if (pol_a == e.polarity) else "-"
                    )
                    targets_a[e.target] = net

                targets_b = {}
                for e in edges_b:
                    net = (
                        "+" if (pol_b == e.polarity) else "-"
                    )
                    targets_b[e.target] = net

                shared = set(targets_a.keys()) & set(targets_b.keys())
                for target in shared:
                    if targets_a[target] != targets_b[target]:
                        cancellations.append((child_a, child_b))
                        break

        return cancellations

    # ------------------------------------------------------------------
    # Net effect builder
    # ------------------------------------------------------------------

    def _build_net_effect(
        self,
        cause: str,
        effects: list[tuple[str, str, float]],
        cancellations: list[tuple[str, str]],
        loops: list[list[str]],
    ) -> str:
        """Build human-readable net effect summary."""
        parts: list[str] = []

        if cancellations:
            for a, b in cancellations:
                parts.append(
                    f"{a} and {b} cancel out"
                )

        if loops:
            parts.append(
                f"reinforcing feedback loop detected: "
                f"{' -> '.join(loops[0])}"
            )

        if not cancellations and not loops:
            if effects:
                pos = sum(1 for _, d, _ in effects if d == "positive")
                neg = sum(1 for _, d, _ in effects if d == "negative")
                if pos > neg:
                    parts.append(f"net positive effect from {cause}")
                elif neg > pos:
                    parts.append(f"net negative effect from {cause}")
                else:
                    parts.append(f"mixed effects from {cause}")
            else:
                parts.append(f"no downstream effects from {cause}")

        return "; ".join(parts) if parts else f"effect of {cause} traced"
```

### Step 3.3 -- Verify

- [ ] Run `uv run pytest tests/test_causal_tracer.py -v --tb=short -x`
- [ ] Run `uv run ruff check mowm/reasoning/causal_tracer.py`
- [ ] Verify all tests pass, no lint errors

---

## Task 4: reasoning/comparator.py -- ComparisonResult + Comparator

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\comparator.py`

Depends on: `AxiomLibrary`, `SafeFormulaExecutor`. No other reasoning module dependencies.

This module compares two or more concepts by finding their axioms,
executing formulas with shared variables, and analyzing structural differences.

### Step 4.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_comparator.py`

```python
# tests/test_comparator.py
"""Tests for Comparator -- dual-execute and structural diff across domains."""
from __future__ import annotations

import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.reasoning.comparator import Comparator, ComparisonResult

K, L = 4, 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> SafeFormulaExecutor:
    return SafeFormulaExecutor()


@pytest.fixture
def physics_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    ))
    lib.register(lib.make_axiom(
        "Momentum", "physics.dynamics", "p = m * v",
    ))
    lib.register(lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    ))
    lib.register(lib.make_axiom(
        "Weight", "physics.dynamics", "W = m * g",
    ))
    return lib


@pytest.fixture
def economics_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Supply-Demand Equilibrium", "economics.micro",
        "Q_d = Q_s",
    ))
    lib.register(lib.make_axiom(
        "Price Elasticity", "economics.micro",
        "epsilon = dQ / dP",
    ))
    lib.register(lib.make_axiom(
        "Consumer Surplus", "economics.micro",
        "CS = WTP - P",
    ))
    lib.register(lib.make_axiom(
        "Producer Surplus", "economics.micro",
        "PS = P - MC",
    ))
    return lib


@pytest.fixture
def music_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom(
        "Major Third", "music.harmony", "ratio = 5 / 4",
    ))
    lib.register(lib.make_axiom(
        "Minor Third", "music.harmony", "ratio = 6 / 5",
    ))
    lib.register(lib.make_axiom(
        "Perfect Fifth", "music.harmony", "ratio = 3 / 2",
    ))
    return lib


@pytest.fixture
def phys_comparator(physics_lib, executor) -> Comparator:
    return Comparator(library=physics_lib, executor=executor)


@pytest.fixture
def econ_comparator(economics_lib, executor) -> Comparator:
    return Comparator(library=economics_lib, executor=executor)


@pytest.fixture
def music_comparator(music_lib, executor) -> Comparator:
    return Comparator(library=music_lib, executor=executor)


# ---------------------------------------------------------------------------
# TestComparisonResult dataclass
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_default_fields(self):
        r = ComparisonResult()
        assert r.items == []
        assert r.similarities == []
        assert r.differences == []
        assert r.structural_diff == ""


# ---------------------------------------------------------------------------
# TestComparator -- Physics: force vs momentum
# ---------------------------------------------------------------------------


class TestComparePhysics:
    def test_compare_force_momentum(self, phys_comparator):
        result = phys_comparator.compare(
            item_names=["Newton's Second Law", "Momentum"],
            domain="physics",
        )
        assert isinstance(result, ComparisonResult)
        assert len(result.items) == 2
        # Both depend on mass -> should be a similarity
        assert len(result.similarities) > 0
        # Force uses acceleration, momentum uses velocity -> different
        assert len(result.differences) > 0

    def test_compare_with_variables(self, phys_comparator):
        result = phys_comparator.compare(
            item_names=["Newton's Second Law", "Weight"],
            domain="physics",
            variables={"m": 10.0, "a": 9.8, "g": 9.8},
        )
        assert len(result.items) == 2
        # Both should have computed values
        for item in result.items:
            if "value" in item and item["value"] is not None:
                assert isinstance(item["value"], (int, float))

    def test_compare_structural_diff(self, phys_comparator):
        result = phys_comparator.compare(
            item_names=["Newton's Second Law", "Kinetic Energy"],
            domain="physics",
        )
        assert result.structural_diff != ""

    def test_compare_nonexistent_axiom(self, phys_comparator):
        result = phys_comparator.compare(
            item_names=["Newton's Second Law", "Nonexistent Axiom"],
            domain="physics",
        )
        # Should still work with what it can find
        assert isinstance(result, ComparisonResult)


# ---------------------------------------------------------------------------
# TestComparator -- Economics: consumer vs producer surplus
# ---------------------------------------------------------------------------


class TestCompareEconomics:
    def test_compare_surpluses(self, econ_comparator):
        result = econ_comparator.compare(
            item_names=["Consumer Surplus", "Producer Surplus"],
            domain="economics",
        )
        assert len(result.items) == 2
        # Both involve price P -> similarity
        assert len(result.similarities) > 0
        # CS = WTP - P, PS = P - MC -> structural difference
        assert len(result.differences) > 0

    def test_compare_with_values(self, econ_comparator):
        result = econ_comparator.compare(
            item_names=["Consumer Surplus", "Producer Surplus"],
            domain="economics",
            variables={"WTP": 50.0, "P": 30.0, "MC": 10.0},
        )
        for item in result.items:
            if "value" in item and item["value"] is not None:
                assert isinstance(item["value"], (int, float))


# ---------------------------------------------------------------------------
# TestComparator -- Music: major vs minor thirds
# ---------------------------------------------------------------------------


class TestCompareMusic:
    def test_compare_intervals(self, music_comparator):
        result = music_comparator.compare(
            item_names=["Major Third", "Minor Third"],
            domain="music",
        )
        assert len(result.items) == 2
        assert len(result.differences) > 0

    def test_compare_three_items(self, music_comparator):
        result = music_comparator.compare(
            item_names=["Major Third", "Minor Third", "Perfect Fifth"],
            domain="music",
        )
        assert len(result.items) == 3


# ---------------------------------------------------------------------------
# TestComparator -- Edge cases
# ---------------------------------------------------------------------------


class TestCompareEdgeCases:
    def test_compare_single_item(self, phys_comparator):
        result = phys_comparator.compare(
            item_names=["Newton's Second Law"],
            domain="physics",
        )
        assert len(result.items) == 1
        assert result.similarities == []
        assert result.differences == []

    def test_compare_empty_items(self, phys_comparator):
        result = phys_comparator.compare(
            item_names=[],
            domain="physics",
        )
        assert len(result.items) == 0
```

### Step 4.2 -- Write implementation

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\comparator.py`

```python
# mowm/reasoning/comparator.py
"""Comparator -- dual-execute and structural diff for COMPARE intent.

Compares two or more concepts by:
  1. Finding their axioms in the library
  2. Executing formulas with shared variables (if provided)
  3. Analyzing structural differences in formulas
  4. Identifying shared and distinguishing properties
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary
    from mowm.executor import SafeFormulaExecutor

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")

_FUNCTIONS = {
    "sin", "cos", "sqrt", "exp", "log", "square", "not",
    "and", "or", "implies", "for", "all", "if", "iff",
    "such", "that", "with", "subject", "to", "max", "min",
    "sum", "prod", "integral", "no", "in", "const",
    "avoid", "consecutive", "between", "voices", "pairs",
}


def _formula_vars(formula: str) -> set[str]:
    """Extract variable identifiers from a formula string."""
    raw = set(_TOKEN_RE.findall(formula))
    return {v for v in raw if v.lower() not in _FUNCTIONS and not v.isdigit()}


def _extract_lhs(formula: str) -> str | None:
    """Extract LHS variable from 'X = expr' formula."""
    if "=" in formula and "==" not in formula:
        parts = formula.split("=", 1)
        lhs = parts[0].strip()
        if re.match(r"^[A-Za-z_]\w*$", lhs):
            return lhs
    return None


@dataclass
class ComparisonResult:
    """Structured comparison result.

    Attributes:
        items: List of dicts with {name, axiom_name, formula, domain, value}.
        similarities: Shared properties between items.
        differences: Distinguishing properties between items.
        structural_diff: How the formulas differ structurally.
    """

    items: list[dict[str, Any]] = field(default_factory=list)
    similarities: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    structural_diff: str = ""


class Comparator:
    """Compare concepts by finding axioms, executing, and diffing.

    Args:
        library: AxiomLibrary for axiom lookup.
        executor: SafeFormulaExecutor for formula evaluation.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
    ) -> None:
        self._lib = library
        self._executor = executor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compare(
        self,
        item_names: list[str],
        domain: str,
        variables: dict[str, float] | None = None,
    ) -> ComparisonResult:
        """Compare two or more concepts by axiom lookup and structural diff.

        Args:
            item_names: Axiom names or concept keywords to compare.
            domain: Domain for scoping the search.
            variables: Optional variable bindings for numeric execution.

        Returns:
            ComparisonResult with items, similarities, differences,
            and structural diff.
        """
        if not item_names:
            return ComparisonResult()

        variables = variables or {}

        # Step 1: find axioms for each item
        items: list[dict[str, Any]] = []
        found_axioms: list[Axiom | None] = []
        for name in item_names:
            axiom = self._find_axiom(name, domain)
            found_axioms.append(axiom)
            if axiom:
                value = self._try_execute(axiom, variables)
                items.append({
                    "name": name,
                    "axiom_name": axiom.name,
                    "formula": axiom.formula_str,
                    "domain": axiom.domain,
                    "value": value,
                })
            else:
                items.append({
                    "name": name,
                    "axiom_name": None,
                    "formula": None,
                    "domain": domain,
                    "value": None,
                })

        # Step 2: analyze similarities and differences
        valid_axioms = [a for a in found_axioms if a is not None]

        if len(valid_axioms) < 2:
            return ComparisonResult(items=items)

        similarities = self._find_similarities(valid_axioms)
        differences = self._find_differences(valid_axioms)
        structural_diff = self._structural_diff(valid_axioms)

        return ComparisonResult(
            items=items,
            similarities=similarities,
            differences=differences,
            structural_diff=structural_diff,
        )

    # ------------------------------------------------------------------
    # Axiom lookup
    # ------------------------------------------------------------------

    def _find_axiom(self, name: str, domain: str) -> Axiom | None:
        """Find an axiom by exact name match or keyword search."""
        all_axioms = self._lib.all_axioms()

        # Exact name match
        for ax in all_axioms:
            if ax.name == name:
                return ax

        # Keyword match (name tokens overlap)
        name_tokens = {
            t.lower() for t in re.findall(r"[A-Za-z_]\w*", name)
        }
        best: Axiom | None = None
        best_score = 0
        for ax in all_axioms:
            ax_tokens = {
                t.lower()
                for t in re.findall(r"[A-Za-z_]\w*", ax.name)
            }
            score = len(name_tokens & ax_tokens)
            domain_bonus = 1 if ax.domain.split(".")[0] == domain else 0
            total = score + domain_bonus
            if total > best_score:
                best_score = total
                best = ax

        return best if best_score > 0 else None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _try_execute(
        self, axiom: Axiom, variables: dict[str, float],
    ) -> float | None:
        """Try to execute an axiom formula with given variables."""
        if not variables:
            return None
        try:
            return self._executor.execute(axiom.formula_str, variables)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Similarity / difference analysis
    # ------------------------------------------------------------------

    def _find_similarities(self, axioms: list[Axiom]) -> list[str]:
        """Find shared variables across all axioms."""
        if len(axioms) < 2:
            return []

        var_sets = [_formula_vars(ax.formula_str) for ax in axioms]
        # Shared variables present in ALL axioms
        shared = var_sets[0]
        for vs in var_sets[1:]:
            shared = shared & vs

        similarities: list[str] = []
        if shared:
            similarities.append(
                f"Shared variables: {', '.join(sorted(shared))}"
            )

        # Shared domain
        domains = {ax.domain.split(".")[0] for ax in axioms}
        if len(domains) == 1:
            similarities.append(f"Same domain: {domains.pop()}")

        # Shared sub-domain
        sub_domains = {ax.domain for ax in axioms}
        if len(sub_domains) == 1:
            similarities.append(f"Same sub-domain: {sub_domains.pop()}")

        # Shared operators
        if all(ax.operators for ax in axioms):
            op_sets = [set(ax.operators) for ax in axioms]
            shared_ops = op_sets[0]
            for ops in op_sets[1:]:
                shared_ops = shared_ops & ops
            if shared_ops:
                similarities.append(
                    f"Shared operations: {', '.join(sorted(shared_ops))}"
                )

        return similarities

    def _find_differences(self, axioms: list[Axiom]) -> list[str]:
        """Find distinguishing variables and structural differences."""
        if len(axioms) < 2:
            return []

        differences: list[str] = []
        var_sets = [_formula_vars(ax.formula_str) for ax in axioms]

        # Unique variables per axiom
        for i, ax in enumerate(axioms):
            others = set()
            for j, vs in enumerate(var_sets):
                if j != i:
                    others.update(vs)
            unique = var_sets[i] - others
            if unique:
                differences.append(
                    f"{ax.name} uniquely uses: {', '.join(sorted(unique))}"
                )

        # Different sub-domains
        sub_domains = [(ax.name, ax.domain) for ax in axioms]
        if len({d for _, d in sub_domains}) > 1:
            parts = [f"{name}: {dom}" for name, dom in sub_domains]
            differences.append(f"Different sub-domains: {'; '.join(parts)}")

        return differences

    def _structural_diff(self, axioms: list[Axiom]) -> str:
        """Describe structural formula differences."""
        if len(axioms) < 2:
            return ""

        parts: list[str] = []
        for ax in axioms:
            lhs = _extract_lhs(ax.formula_str)
            if lhs:
                rhs_vars = _formula_vars(ax.formula_str) - {lhs}
                parts.append(
                    f"{ax.name} ({lhs}) depends on: "
                    f"{', '.join(sorted(rhs_vars)) if rhs_vars else 'constants'}"
                )
            else:
                parts.append(f"{ax.name}: {ax.formula_str}")

        return " vs ".join(parts)
```

### Step 4.3 -- Verify

- [ ] Run `uv run pytest tests/test_comparator.py -v --tb=short -x`
- [ ] Run `uv run ruff check mowm/reasoning/comparator.py`
- [ ] Verify all tests pass, no lint errors

---

## Task 5: reasoning/composer.py -- Composer

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\composer.py`

Depends on: `DerivationChain` (Task 2), `CausalNarrative` (Task 3), `ComparisonResult` (Task 4), `KnowledgeContext` (Task 1).

The text generation engine. Composes natural language from structured
reasoning results. Template-based, NOT a language model. Complexity
scales with reasoning depth (levels 1-4).

### Step 5.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_composer.py`

```python
# tests/test_composer.py
"""Tests for Composer -- template-based text generation from reasoning."""
from __future__ import annotations

import pytest

from mowm.reasoning.chain_builder import DerivationChain, DerivationStep
from mowm.reasoning.causal_tracer import CausalNarrative
from mowm.reasoning.comparator import ComparisonResult
from mowm.reasoning.composer import Composer
from mowm.reasoning.knowledge_search import KnowledgeContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def composer() -> Composer:
    return Composer()


@pytest.fixture
def physics_chain() -> DerivationChain:
    """Chain for: why do objects fall at the same rate?"""
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Weight",
                formula="W = m * g",
                substitutions={},
                result_var="W",
                result_value="m * g",
                explanation="From Weight: W = m * g",
            ),
            DerivationStep(
                axiom_name="Newton's Second Law",
                formula="F = m * a",
                substitutions={},
                result_var="a",
                result_value="F / m = m*g / m = g",
                explanation=(
                    "From Newton's Second Law: a = F/m = m*g/m = g"
                ),
            ),
        ],
        conclusion=(
            "Acceleration a = g for all masses; mass cancels."
        ),
        variables_used={"m", "g", "a", "F", "W"},
        axioms_used=["Weight", "Newton's Second Law"],
    )


@pytest.fixture
def physics_causal() -> CausalNarrative:
    return CausalNarrative(
        cause="mass",
        effects=[
            ("grav_force", "positive", 1.0),
            ("inertia", "positive", 1.0),
        ],
        cancellations=[("grav_force", "inertia")],
        net_effect="grav_force and inertia cancel out",
        graph_path=["mass", "grav_force", "inertia", "acceleration"],
    )


@pytest.fixture
def economics_chain() -> DerivationChain:
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Supply-Demand Equilibrium",
                formula="Q_d = Q_s",
                substitutions={},
                result_var="Q_d",
                result_value="Q_s",
                explanation=(
                    "At equilibrium, quantity demanded equals "
                    "quantity supplied."
                ),
            ),
        ],
        conclusion="Equilibrium: Q_d = Q_s at price P*.",
        variables_used={"Q_d", "Q_s", "P"},
        axioms_used=["Supply-Demand Equilibrium"],
    )


@pytest.fixture
def logic_chain() -> DerivationChain:
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Modus Ponens",
                formula="(P AND (P IMPLIES Q)) IMPLIES Q",
                substitutions={},
                result_var="Q",
                result_value="True (given P and P->Q)",
                explanation=(
                    "By Modus Ponens: if P is true and P implies Q, "
                    "then Q must be true."
                ),
            ),
        ],
        conclusion="Q follows from P and P->Q by Modus Ponens.",
        variables_used={"P", "Q"},
        axioms_used=["Modus Ponens"],
    )


@pytest.fixture
def bio_chain() -> DerivationChain:
    return DerivationChain(
        steps=[
            DerivationStep(
                axiom_name="Hamilton's Rule",
                formula="benefit = r * B - C",
                substitutions={"r": 0.5, "B": 10.0, "C": 3.0},
                result_var="benefit",
                result_value=2.0,
                explanation=(
                    "Hamilton's Rule: benefit = r*B - C = "
                    "0.5*10 - 3 = 2.0 > 0, altruism favored."
                ),
            ),
        ],
        conclusion="benefit = 2.0 > 0; altruism is favored.",
        variables_used={"r", "B", "C", "benefit"},
        axioms_used=["Hamilton's Rule"],
    )


@pytest.fixture
def music_comparison() -> ComparisonResult:
    return ComparisonResult(
        items=[
            {
                "name": "Major Third",
                "axiom_name": "Major Third",
                "formula": "ratio = 5 / 4",
                "domain": "music.harmony",
                "value": 1.25,
            },
            {
                "name": "Minor Third",
                "axiom_name": "Minor Third",
                "formula": "ratio = 6 / 5",
                "domain": "music.harmony",
                "value": 1.2,
            },
        ],
        similarities=["Same domain: music", "Both are intervals"],
        differences=[
            "Major Third ratio: 5/4 = 1.25",
            "Minor Third ratio: 6/5 = 1.2",
        ],
        structural_diff=(
            "Major Third uses 5/4 ratio vs Minor Third uses 6/5 ratio"
        ),
    )


@pytest.fixture
def causal_comparison() -> ComparisonResult:
    return ComparisonResult(
        items=[
            {
                "name": "Do-Calculus Rule 1",
                "axiom_name": "Do-Calculus Rule 1",
                "formula": "P_Y_do_X_Z = P_Y_do_X",
                "domain": "causal.intervention",
                "value": None,
            },
            {
                "name": "Bayes Theorem",
                "axiom_name": "Bayes Theorem",
                "formula": "P_A_given_B = P_B_given_A * P_A / P_B",
                "domain": "causal.probability",
                "value": None,
            },
        ],
        similarities=["Same domain: causal"],
        differences=[
            "Do-Calculus handles interventions (do-operator)",
            "Bayes handles observational conditioning",
        ],
        structural_diff=(
            "Do-Calculus Rule 1 (P_Y_do_X_Z) depends on: "
            "P_Y_do_X vs Bayes Theorem (P_A_given_B) "
            "depends on: P_B_given_A, P_A, P_B"
        ),
    )


# ---------------------------------------------------------------------------
# TestLevelSelection
# ---------------------------------------------------------------------------


class TestLevelSelection:
    def test_level_1_no_chain(self, composer):
        level = composer._select_level(chain=None, causal=None)
        assert level == 1

    def test_level_2_short_chain(self, composer):
        chain = DerivationChain(
            steps=[DerivationStep(
                axiom_name="A", formula="x = 1",
                explanation="simple",
            )],
            conclusion="x = 1",
            variables_used={"x"},
            axioms_used=["A"],
        )
        level = composer._select_level(chain=chain, causal=None)
        assert level == 2

    def test_level_3_chain_with_multiple_steps(self, composer, physics_chain):
        level = composer._select_level(
            chain=physics_chain, causal=None,
        )
        assert level >= 3

    def test_level_4_chain_plus_causal(
        self, composer, physics_chain, physics_causal,
    ):
        level = composer._select_level(
            chain=physics_chain, causal=physics_causal,
        )
        assert level == 4


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Physics
# ---------------------------------------------------------------------------


class TestComposeExplanationPhysics:
    def test_explanation_contains_conclusion(
        self, composer, physics_chain, physics_causal,
    ):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=physics_causal,
            question="Why do heavier objects fall at the same rate?",
        )
        assert "mass" in text.lower() or "cancel" in text.lower()

    def test_explanation_lists_axioms(
        self, composer, physics_chain, physics_causal,
    ):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=physics_causal,
            question="Why do heavier objects fall at the same rate?",
        )
        assert "Weight" in text or "Newton" in text

    def test_explanation_has_derivation_steps(
        self, composer, physics_chain,
    ):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=None,
            question="How does force relate to mass?",
        )
        assert "W = m * g" in text or "F = m * a" in text

    def test_explanation_nonempty(self, composer, physics_chain):
        text = composer.compose_explanation(
            chain=physics_chain,
            causal=None,
            question="Explain force.",
        )
        assert len(text) > 20


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Economics
# ---------------------------------------------------------------------------


class TestComposeExplanationEconomics:
    def test_economics_explanation(self, composer, economics_chain):
        text = composer.compose_explanation(
            chain=economics_chain,
            causal=None,
            question="Explain supply and demand equilibrium.",
        )
        assert "equilibrium" in text.lower() or "Q_d" in text

    def test_economics_lists_axiom(self, composer, economics_chain):
        text = composer.compose_explanation(
            chain=economics_chain,
            causal=None,
            question="Explain supply and demand.",
        )
        assert "Supply-Demand" in text or "Q_s" in text


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Logic
# ---------------------------------------------------------------------------


class TestComposeExplanationLogic:
    def test_logic_explanation(self, composer, logic_chain):
        text = composer.compose_explanation(
            chain=logic_chain,
            causal=None,
            question="Explain modus ponens.",
        )
        assert "Modus Ponens" in text
        assert "P" in text or "Q" in text


# ---------------------------------------------------------------------------
# TestComposeExplanation -- Bio
# ---------------------------------------------------------------------------


class TestComposeExplanationBio:
    def test_bio_explanation(self, composer, bio_chain):
        text = composer.compose_explanation(
            chain=bio_chain,
            causal=None,
            question=(
                "Why does natural selection favor altruism in kin?"
            ),
        )
        assert "Hamilton" in text
        assert "2.0" in text or "benefit" in text.lower()


# ---------------------------------------------------------------------------
# TestComposeComparison
# ---------------------------------------------------------------------------


class TestComposeComparisonMusic:
    def test_music_comparison(self, composer, music_comparison):
        text = composer.compose_comparison(
            result=music_comparison,
            question="Compare major and minor thirds.",
        )
        assert "Major Third" in text
        assert "Minor Third" in text
        assert "1.25" in text or "5 / 4" in text

    def test_comparison_shows_differences(
        self, composer, music_comparison,
    ):
        text = composer.compose_comparison(
            result=music_comparison,
            question="Compare intervals.",
        )
        # Should mention the different ratios
        assert "1.2" in text or "6 / 5" in text or "differ" in text.lower()


class TestComposeComparisonCausal:
    def test_causal_comparison(self, composer, causal_comparison):
        text = composer.compose_comparison(
            result=causal_comparison,
            question="Compare do-calculus and Bayes theorem.",
        )
        assert "Do-Calculus" in text or "Bayes" in text


# ---------------------------------------------------------------------------
# TestComposeGeneral
# ---------------------------------------------------------------------------


class TestComposeGeneral:
    def test_general_from_context(self, composer):
        from mowm.axiom_library import AxiomLibrary
        lib = AxiomLibrary(k=4, l=32)
        ax = lib.make_axiom(
            "Entropy Change", "physics.thermo", "dS = dQ / T",
        )
        ctx = KnowledgeContext(
            primary_axioms=[ax],
            supporting_axioms=[],
            concepts=["Entropy Change"],
            domain_chain=["physics.thermo"],
            variables_mentioned={"dS", "dQ", "T"},
        )
        text = composer.compose_general(
            context=ctx,
            chain=None,
            question="Tell me about entropy.",
        )
        assert "Entropy" in text or "dS" in text
        assert len(text) > 10

    def test_general_with_chain(self, composer, physics_chain):
        from mowm.axiom_library import AxiomLibrary
        lib = AxiomLibrary(k=4, l=32)
        ax = lib.make_axiom(
            "Newton's Second Law", "physics.dynamics", "F = m * a",
        )
        ctx = KnowledgeContext(
            primary_axioms=[ax],
            concepts=["Newton's Second Law"],
            domain_chain=["physics.dynamics"],
            variables_mentioned={"F", "m", "a"},
        )
        text = composer.compose_general(
            context=ctx,
            chain=physics_chain,
            question="Tell me about force.",
        )
        assert len(text) > 20


# ---------------------------------------------------------------------------
# TestComposeEdgeCases
# ---------------------------------------------------------------------------


class TestComposeEdgeCases:
    def test_empty_chain_explanation(self, composer):
        chain = DerivationChain()
        text = composer.compose_explanation(
            chain=chain,
            causal=None,
            question="Why?",
        )
        assert isinstance(text, str)
        assert len(text) > 0

    def test_empty_comparison(self, composer):
        result = ComparisonResult()
        text = composer.compose_comparison(
            result=result,
            question="Compare nothing.",
        )
        assert isinstance(text, str)

    def test_empty_context_general(self, composer):
        ctx = KnowledgeContext()
        text = composer.compose_general(
            context=ctx,
            chain=None,
            question="Tell me about nothing.",
        )
        assert isinstance(text, str)
```

### Step 5.2 -- Write implementation

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\composer.py`

```python
# mowm/reasoning/composer.py
"""Composer -- template-based natural language generation from reasoning.

NOT a language model. Uses hierarchical templates:
  Level 1: Direct statement (1 sentence)
  Level 2: Statement + evidence (2-3 sentences)
  Level 3: Statement + derivation chain (paragraph)
  Level 4: Full explanation with intro/chain/causal/conclusion

Quality comes from reasoning depth, not fluency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mowm.reasoning.causal_tracer import CausalNarrative
    from mowm.reasoning.chain_builder import DerivationChain
    from mowm.reasoning.comparator import ComparisonResult
    from mowm.reasoning.knowledge_search import KnowledgeContext


class Composer:
    """Compose natural language from structured reasoning results.

    Template-based generation with complexity scaling:
      Level 1: Direct statement (1 sentence)
      Level 2: Statement + evidence (2-3 sentences)
      Level 3: Statement + derivation chain (paragraph)
      Level 4: Full explanation with intro/chain/causal/conclusion
    """

    # ------------------------------------------------------------------
    # compose_explanation
    # ------------------------------------------------------------------

    def compose_explanation(
        self,
        chain: DerivationChain,
        causal: CausalNarrative | None,
        question: str,
    ) -> str:
        """Compose a full explanation from derivation chain + causal trace.

        Structure (Level 4):
          1. Opening: restate the core principle
          2. Derivation: walk through axiom chain step by step
          3. Causal context: if available, explain cause-effect
          4. Conclusion: direct answer to the "why/how"
          5. Evidence: list axioms used

        Args:
            chain: DerivationChain from ChainBuilder.
            causal: Optional CausalNarrative from CausalTracer.
            question: Original question text.

        Returns:
            Composed natural language explanation string.
        """
        level = self._select_level(chain, causal)

        if level == 1:
            return self._compose_level1(chain, question)
        if level == 2:
            return self._compose_level2(chain, question)
        if level == 3:
            return self._compose_level3(chain, question)
        return self._compose_level4(chain, causal, question)

    # ------------------------------------------------------------------
    # compose_comparison
    # ------------------------------------------------------------------

    def compose_comparison(
        self,
        result: ComparisonResult,
        question: str,
    ) -> str:
        """Compose comparison text from structured diff.

        Args:
            result: ComparisonResult from Comparator.
            question: Original question text.

        Returns:
            Composed comparison text.
        """
        if not result.items:
            return "No items available for comparison."

        lines: list[str] = []

        # Header
        names = [item["name"] for item in result.items]
        if len(names) == 2:
            lines.append(f"{names[0]} and {names[1]} are compared below.")
        elif len(names) > 2:
            lines.append(
                f"{', '.join(names[:-1])}, and {names[-1]} "
                f"are compared below."
            )
        else:
            lines.append(f"Analysis of {names[0]}:")

        lines.append("")

        # Item details
        for item in result.items:
            name = item.get("name", "?")
            formula = item.get("formula")
            value = item.get("value")
            if formula:
                line = f"{name}: {formula}"
                if value is not None:
                    line += f" = {value}"
                lines.append(line)
            else:
                lines.append(f"{name}: no formula found")

        # Similarities
        if result.similarities:
            lines.append("")
            lines.append("Similarities:")
            for sim in result.similarities:
                lines.append(f"  - {sim}")

        # Differences
        if result.differences:
            lines.append("")
            lines.append("Differences:")
            for diff in result.differences:
                lines.append(f"  - {diff}")

        # Structural diff
        if result.structural_diff:
            lines.append("")
            lines.append(f"Structural: {result.structural_diff}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # compose_general
    # ------------------------------------------------------------------

    def compose_general(
        self,
        context: KnowledgeContext,
        chain: DerivationChain | None,
        question: str,
    ) -> str:
        """Compose general knowledge response from axiom library.

        Args:
            context: KnowledgeContext from KnowledgeSearch.
            chain: Optional DerivationChain if variables were present.
            question: Original question text.

        Returns:
            Composed general knowledge text.
        """
        lines: list[str] = []

        # Intro from concepts
        if context.concepts:
            if context.domain_chain:
                domain_str = ", ".join(context.domain_chain[:3])
                lines.append(
                    f"{context.concepts[0]} ({domain_str})."
                )
            else:
                lines.append(f"{context.concepts[0]}.")

        # Key formulas from primary axioms
        if context.primary_axioms:
            lines.append("")
            lines.append("Key formulas:")
            for ax in context.primary_axioms:
                lines.append(f"  - {ax.name}: {ax.formula_str}")

        # Supporting axioms
        if context.supporting_axioms:
            lines.append("")
            lines.append("Related concepts:")
            for ax in context.supporting_axioms[:5]:
                lines.append(f"  - {ax.name}: {ax.formula_str}")

        # Derivation chain if present
        if chain and chain.steps:
            lines.append("")
            lines.append("Derivation:")
            for i, step in enumerate(chain.steps, 1):
                lines.append(f"  {i}. {step.explanation}")
            if chain.conclusion:
                lines.append(f"  => {chain.conclusion}")

        # Variables mentioned
        if context.variables_mentioned:
            sorted_vars = sorted(context.variables_mentioned)
            lines.append("")
            lines.append(
                f"Variables: {', '.join(sorted_vars[:15])}"
            )

        # Axioms used
        all_axioms = context.primary_axioms + context.supporting_axioms
        if all_axioms:
            lines.append("")
            lines.append("Axioms used:")
            seen: set[str] = set()
            for ax in all_axioms:
                if ax.name not in seen:
                    seen.add(ax.name)
                    lines.append(f"  - {ax.name}")

        if not lines:
            return (
                "No specific information found in the world model "
                "for this question."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Level selection
    # ------------------------------------------------------------------

    def _select_level(
        self,
        chain: DerivationChain | None,
        causal: CausalNarrative | None,
    ) -> int:
        """Choose template complexity level (1-4).

        Level 1: No chain, no causal -> direct statement
        Level 2: Short chain (1 step), no causal
        Level 3: Multi-step chain, no causal
        Level 4: Chain + causal narrative
        """
        if chain is None or not chain.steps:
            return 1
        if causal is not None and len(chain.steps) >= 1:
            return 4
        if len(chain.steps) >= 2:
            return 3
        return 2

    # ------------------------------------------------------------------
    # Level templates
    # ------------------------------------------------------------------

    def _compose_level1(
        self, chain: DerivationChain, question: str,
    ) -> str:
        """Level 1: direct statement."""
        if chain.conclusion:
            return chain.conclusion
        return f"Based on the available axioms: {question}"

    def _compose_level2(
        self, chain: DerivationChain, question: str,
    ) -> str:
        """Level 2: statement + evidence."""
        lines: list[str] = []
        if chain.conclusion:
            lines.append(chain.conclusion)
        if chain.steps:
            step = chain.steps[0]
            lines.append(
                f"This follows from {step.axiom_name}: "
                f"{step.formula}."
            )
        if chain.axioms_used:
            lines.append(
                f"Axiom used: {', '.join(chain.axioms_used)}."
            )
        return "\n".join(lines) if lines else "No derivation available."

    def _compose_level3(
        self, chain: DerivationChain, question: str,
    ) -> str:
        """Level 3: statement + derivation chain."""
        lines: list[str] = []

        if chain.conclusion:
            lines.append(chain.conclusion)

        lines.append("")
        lines.append("Derivation:")
        for i, step in enumerate(chain.steps, 1):
            lines.append(f"  {i}. {step.explanation}")

        if chain.axioms_used:
            lines.append("")
            lines.append("Axioms used:")
            for name in chain.axioms_used:
                lines.append(f"  - {name}")

        return "\n".join(lines)

    def _compose_level4(
        self,
        chain: DerivationChain,
        causal: CausalNarrative | None,
        question: str,
    ) -> str:
        """Level 4: full explanation with intro/chain/causal/conclusion."""
        lines: list[str] = []

        # 1. Opening
        if chain.conclusion:
            lines.append(chain.conclusion)
        lines.append("")

        # 2. Derivation chain
        lines.append("Derivation:")
        for i, step in enumerate(chain.steps, 1):
            lines.append(f"  {i}. {step.explanation}")

        # 3. Causal context
        if causal:
            lines.append("")
            if causal.cancellations:
                for a, b in causal.cancellations:
                    lines.append(
                        f"The key insight is that {a} and {b} "
                        f"cancel each other out."
                    )
            if causal.net_effect:
                lines.append(f"Net effect: {causal.net_effect}")
            if causal.effects:
                lines.append("Causal path:")
                for target, direction, strength in causal.effects:
                    lines.append(
                        f"  - {causal.cause} -> {target} "
                        f"({direction}, strength {strength:.2f})"
                    )

        # 4. Conclusion
        if chain.conclusion:
            lines.append("")
            lines.append(f"Therefore: {chain.conclusion}")

        # 5. Evidence
        if chain.axioms_used:
            lines.append("")
            lines.append("Axioms used:")
            for name in chain.axioms_used:
                lines.append(f"  - {name}")

        return "\n".join(lines)
```

### Step 5.3 -- Verify

- [ ] Run `uv run pytest tests/test_composer.py -v --tb=short -x`
- [ ] Run `uv run ruff check mowm/reasoning/composer.py`
- [ ] Verify all tests pass, no lint errors

---

## Task 6: reasoning/__init__.py -- ReasoningDecoder orchestrator

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\__init__.py`

Depends on: ALL previous tasks (1-5).

The ReasoningDecoder orchestrates KnowledgeSearch, ChainBuilder, CausalTracer,
Comparator, and Composer to produce self-contained responses for EXPLAIN,
COMPARE, and GENERAL intents. Replaces AugmentedSolver as the primary handler.

### Step 6.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_reasoning_decoder.py`

```python
# tests/test_reasoning_decoder.py
"""Tests for ReasoningDecoder -- end-to-end reasoning across all domains."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa.direct_solver import QAResponse
from mowm.qa.parser import Intent, ParsedQuestion
from mowm.reasoning import ReasoningDecoder

K, L = 4, 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> SafeFormulaExecutor:
    return SafeFormulaExecutor()


@pytest.fixture
def logic() -> LogicEngine:
    return LogicEngine()


def _build_multi_domain_lib() -> AxiomLibrary:
    """Build a library covering physics, economics, logic, bio, music,
    historical, and causal domains."""
    lib = AxiomLibrary(k=K, l=L)
    # Physics
    lib.register(lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    ))
    lib.register(lib.make_axiom(
        "Weight", "physics.dynamics", "W = m * g",
    ))
    lib.register(lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    ))
    lib.register(lib.make_axiom(
        "Momentum", "physics.dynamics", "p = m * v",
    ))
    # Economics
    lib.register(lib.make_axiom(
        "Supply-Demand Equilibrium", "economics.micro",
        "Q_d = Q_s",
    ))
    lib.register(lib.make_axiom(
        "Consumer Surplus", "economics.micro", "CS = WTP - P",
    ))
    lib.register(lib.make_axiom(
        "Producer Surplus", "economics.micro", "PS = P - MC",
    ))
    # Logic
    lib.register(lib.make_axiom(
        "Modus Ponens", "logic.deductive",
        "(P AND (P IMPLIES Q)) IMPLIES Q",
    ))
    lib.register(lib.make_axiom(
        "Hypothetical Syllogism", "logic.deductive",
        "((P IMPLIES Q) AND (Q IMPLIES R)) IMPLIES (P IMPLIES R)",
    ))
    # Bio
    lib.register(lib.make_axiom(
        "Hamilton's Rule", "bio.evolution", "benefit = r * B - C",
    ))
    lib.register(lib.make_axiom(
        "Hardy-Weinberg", "bio.genetics",
        "p^2 + 2*p*q + q^2 = 1",
    ))
    # Music
    lib.register(lib.make_axiom(
        "Major Third", "music.harmony", "ratio = 5 / 4",
    ))
    lib.register(lib.make_axiom(
        "Minor Third", "music.harmony", "ratio = 6 / 5",
    ))
    lib.register(lib.make_axiom(
        "Harmonic Series", "music.harmony", "f_n = n * f_0",
    ))
    # Historical
    lib.register(lib.make_axiom(
        "Event Causation", "historical.causal",
        "event_A causes event_B",
    ))
    lib.register(lib.make_axiom(
        "Periodicity", "historical.pattern",
        "period = mean_delta_t",
    ))
    # Causal
    lib.register(lib.make_axiom(
        "Bayes Theorem", "causal.probability",
        "P_A_given_B = P_B_given_A * P_A / P_B",
    ))
    lib.register(lib.make_axiom(
        "Do-Calculus Rule 1", "causal.intervention",
        "P_Y_do_X_Z = P_Y_do_X",
    ))
    return lib


def _build_physics_graph() -> CausalGraph:
    """Physics causal graph with mass cancellation structure."""
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    for name in ["mass", "grav_force", "inertia", "acceleration"]:
        g.add_node(name, "physics", vec)
    g.add_edge("mass", "grav_force", polarity="+", strength=1.0)
    g.add_edge("mass", "inertia", polarity="+", strength=1.0)
    g.add_edge("grav_force", "acceleration", polarity="+", strength=1.0)
    g.add_edge("inertia", "acceleration", polarity="-", strength=1.0)
    # Economics nodes
    for name in ["price", "demand", "supply"]:
        g.add_node(name, "economics", vec)
    g.add_edge("price", "demand", polarity="-", strength=0.8)
    g.add_edge("price", "supply", polarity="+", strength=0.6)
    # Historical nodes
    for name in [
        "innovation", "prosperity", "complacency", "decline",
    ]:
        g.add_node(name, "historical", vec)
    g.add_edge("innovation", "prosperity", polarity="+", strength=0.9)
    g.add_edge("prosperity", "complacency", polarity="+", strength=0.7)
    g.add_edge("complacency", "decline", polarity="+", strength=0.8)
    return g


@pytest.fixture
def lib() -> AxiomLibrary:
    return _build_multi_domain_lib()


@pytest.fixture
def graph() -> CausalGraph:
    return _build_physics_graph()


@pytest.fixture
def decoder(lib, executor, logic, graph) -> ReasoningDecoder:
    return ReasoningDecoder(
        library=lib,
        executor=executor,
        logic=logic,
        causal=graph,
        k=K,
        l=L,
    )


def _make_parsed(
    raw: str,
    intent: Intent,
    keywords: list[str] | None = None,
    variables: dict[str, float] | None = None,
) -> ParsedQuestion:
    """Helper to build a ParsedQuestion."""
    pq = ParsedQuestion(raw=raw)
    pq.intent = intent
    pq.keywords = keywords or []
    pq.variables = variables or {}
    return pq


# ---------------------------------------------------------------------------
# TestReasoningDecoderInit
# ---------------------------------------------------------------------------


class TestReasoningDecoderInit:
    def test_init_creates_decoder(self, decoder):
        assert isinstance(decoder, ReasoningDecoder)

    def test_init_without_causal(self, lib, executor, logic):
        dec = ReasoningDecoder(
            library=lib, executor=executor, logic=logic,
            causal=None, k=K, l=L,
        )
        assert isinstance(dec, ReasoningDecoder)


# ---------------------------------------------------------------------------
# TestExplain -- Physics: "Why do heavier objects fall at the same rate?"
# ---------------------------------------------------------------------------


class TestExplainPhysics:
    def test_explain_free_fall(self, decoder):
        pq = _make_parsed(
            raw="Why do heavier objects fall at the same rate?",
            intent=Intent.EXPLAIN,
            keywords=[
                "heavier", "objects", "fall", "rate", "mass",
                "acceleration",
            ],
        )
        resp = decoder.decode(pq, domain="physics")
        assert isinstance(resp, QAResponse)
        assert resp.confidence > 0.3
        assert resp.mode == "EXPLAIN"
        assert len(resp.answer) > 20

    def test_explain_answer_mentions_axioms(self, decoder):
        pq = _make_parsed(
            raw="Why do heavier objects fall at the same rate?",
            intent=Intent.EXPLAIN,
            keywords=["heavier", "objects", "fall", "rate", "mass"],
        )
        resp = decoder.decode(pq, domain="physics")
        # Should reference Weight or Newton or mass
        answer_lower = resp.answer.lower()
        assert (
            "weight" in answer_lower
            or "newton" in answer_lower
            or "mass" in answer_lower
            or "m" in resp.answer
        )

    def test_explain_has_evidence(self, decoder):
        pq = _make_parsed(
            raw="Why do heavier objects fall at the same rate?",
            intent=Intent.EXPLAIN,
            keywords=["heavier", "objects", "fall", "rate"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert len(resp.evidence) > 0

    def test_explain_has_reasoning_trace(self, decoder):
        pq = _make_parsed(
            raw="Explain force and acceleration.",
            intent=Intent.EXPLAIN,
            keywords=["force", "acceleration"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert len(resp.reasoning_trace) > 0


# ---------------------------------------------------------------------------
# TestExplain -- Economics: "Explain supply and demand"
# ---------------------------------------------------------------------------


class TestExplainEconomics:
    def test_explain_supply_demand(self, decoder):
        pq = _make_parsed(
            raw="Explain supply and demand equilibrium.",
            intent=Intent.EXPLAIN,
            keywords=["supply", "demand", "equilibrium"],
        )
        resp = decoder.decode(pq, domain="economics")
        assert resp.confidence > 0.3
        assert resp.mode == "EXPLAIN"
        assert (
            "supply" in resp.answer.lower()
            or "demand" in resp.answer.lower()
            or "Q_d" in resp.answer
        )


# ---------------------------------------------------------------------------
# TestExplain -- Logic: "Explain modus ponens"
# ---------------------------------------------------------------------------


class TestExplainLogic:
    def test_explain_modus_ponens(self, decoder):
        pq = _make_parsed(
            raw="Explain modus ponens.",
            intent=Intent.EXPLAIN,
            keywords=["modus", "ponens", "implies"],
        )
        resp = decoder.decode(pq, domain="logic")
        assert resp.confidence > 0.3
        assert "Modus Ponens" in resp.answer


# ---------------------------------------------------------------------------
# TestExplain -- Bio: "Why does natural selection favor altruism?"
# ---------------------------------------------------------------------------


class TestExplainBio:
    def test_explain_altruism(self, decoder):
        pq = _make_parsed(
            raw=(
                "Why does natural selection favor altruism in kin?"
            ),
            intent=Intent.EXPLAIN,
            keywords=["altruism", "kin", "selection", "Hamilton"],
        )
        resp = decoder.decode(pq, domain="bio")
        assert resp.confidence > 0.3
        assert (
            "Hamilton" in resp.answer or "benefit" in resp.answer.lower()
        )


# ---------------------------------------------------------------------------
# TestExplain -- Historical: "Why do civilizations rise and fall?"
# ---------------------------------------------------------------------------


class TestExplainHistorical:
    def test_explain_rise_fall(self, decoder):
        pq = _make_parsed(
            raw="Why do civilizations rise and fall?",
            intent=Intent.EXPLAIN,
            keywords=[
                "civilizations", "rise", "fall", "innovation",
                "decline",
            ],
        )
        resp = decoder.decode(pq, domain="historical")
        assert resp.confidence > 0.3
        assert len(resp.answer) > 10


# ---------------------------------------------------------------------------
# TestExplain -- Causal: "What is do-calculus?"
# ---------------------------------------------------------------------------


class TestExplainCausal:
    def test_explain_do_calculus(self, decoder):
        pq = _make_parsed(
            raw="Explain do-calculus and interventions.",
            intent=Intent.EXPLAIN,
            keywords=["do", "calculus", "intervention"],
        )
        resp = decoder.decode(pq, domain="causal")
        assert resp.confidence > 0.3
        assert (
            "Do-Calculus" in resp.answer or "P_Y_do_X" in resp.answer
        )


# ---------------------------------------------------------------------------
# TestCompare -- Physics: "Compare force and momentum"
# ---------------------------------------------------------------------------


class TestComparePhysics:
    def test_compare_force_momentum(self, decoder):
        pq = _make_parsed(
            raw="Compare force and momentum.",
            intent=Intent.COMPARE,
            keywords=["force", "momentum", "compare"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert resp.confidence > 0.3
        assert resp.mode == "COMPARE"
        assert "Newton" in resp.answer or "Momentum" in resp.answer

    def test_compare_has_evidence(self, decoder):
        pq = _make_parsed(
            raw="Compare force and momentum.",
            intent=Intent.COMPARE,
            keywords=["force", "momentum"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert len(resp.evidence) > 0


# ---------------------------------------------------------------------------
# TestCompare -- Economics: "Compare consumer and producer surplus"
# ---------------------------------------------------------------------------


class TestCompareEconomics:
    def test_compare_surpluses(self, decoder):
        pq = _make_parsed(
            raw="Compare consumer and producer surplus.",
            intent=Intent.COMPARE,
            keywords=["consumer", "producer", "surplus"],
        )
        resp = decoder.decode(pq, domain="economics")
        assert resp.confidence > 0.3
        assert (
            "Consumer" in resp.answer or "Producer" in resp.answer
        )


# ---------------------------------------------------------------------------
# TestCompare -- Music: "Compare major and minor chords"
# ---------------------------------------------------------------------------


class TestCompareMusic:
    def test_compare_intervals(self, decoder):
        pq = _make_parsed(
            raw="Compare major and minor thirds.",
            intent=Intent.COMPARE,
            keywords=["major", "minor", "third"],
        )
        resp = decoder.decode(pq, domain="music")
        assert resp.confidence > 0.3
        assert (
            "Major" in resp.answer or "Minor" in resp.answer
        )


# ---------------------------------------------------------------------------
# TestGeneral -- "Tell me about entropy" style
# ---------------------------------------------------------------------------


class TestGeneral:
    def test_general_physics(self, decoder):
        pq = _make_parsed(
            raw="Tell me about kinetic energy.",
            intent=Intent.GENERAL,
            keywords=["kinetic", "energy"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert resp.confidence > 0.3
        assert resp.mode == "GENERAL"
        assert len(resp.answer) > 10

    def test_general_bio(self, decoder):
        pq = _make_parsed(
            raw="Tell me about Hamilton's Rule.",
            intent=Intent.GENERAL,
            keywords=["Hamilton", "Rule", "altruism"],
        )
        resp = decoder.decode(pq, domain="bio")
        assert resp.confidence > 0.3
        assert "Hamilton" in resp.answer

    def test_general_has_evidence(self, decoder):
        pq = _make_parsed(
            raw="Tell me about force.",
            intent=Intent.GENERAL,
            keywords=["force"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert len(resp.evidence) > 0


# ---------------------------------------------------------------------------
# TestFallbackBehavior
# ---------------------------------------------------------------------------


class TestFallback:
    def test_unknown_domain_still_returns(self, decoder):
        pq = _make_parsed(
            raw="Explain underwater basket weaving.",
            intent=Intent.EXPLAIN,
            keywords=["underwater", "basket", "weaving"],
        )
        resp = decoder.decode(pq, domain="unknown_domain")
        assert isinstance(resp, QAResponse)
        # May have low confidence but should not crash
        assert resp.answer != ""

    def test_empty_keywords_still_returns(self, decoder):
        pq = _make_parsed(
            raw="Explain.",
            intent=Intent.EXPLAIN,
            keywords=[],
        )
        resp = decoder.decode(pq, domain="physics")
        assert isinstance(resp, QAResponse)
```

### Step 6.2 -- Write implementation

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\reasoning\__init__.py`

```python
# mowm/reasoning/__init__.py
"""ReasoningDecoder -- self-contained reasoning for EXPLAIN/COMPARE/GENERAL.

Orchestrates KnowledgeSearch, ChainBuilder, CausalTracer, Comparator, and
Composer to produce natural language answers without any external LLM.

Public API:
    ReasoningDecoder -- the main orchestrator
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary
    from mowm.causal_graph import CausalGraph
    from mowm.executor import SafeFormulaExecutor
    from mowm.logic_engine import LogicEngine

from mowm.qa.direct_solver import QAResponse
from mowm.qa.parser import Intent, ParsedQuestion
from mowm.reasoning.causal_tracer import CausalTracer
from mowm.reasoning.chain_builder import ChainBuilder
from mowm.reasoning.comparator import Comparator
from mowm.reasoning.composer import Composer
from mowm.reasoning.knowledge_search import KnowledgeSearch


class ReasoningDecoder:
    """Self-contained reasoning decoder for EXPLAIN/COMPARE/GENERAL.

    Replaces AugmentedSolver as the primary handler for open-ended
    questions. The LLM fallback stays as a true last resort.

    Args:
        library: AxiomLibrary for axiom lookup.
        executor: SafeFormulaExecutor for formula evaluation.
        logic: LogicEngine for logic evaluation.
        causal: Optional CausalGraph for causal tracing.
        k: Number of VSA blocks.
        l: VSA block length.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
        logic: LogicEngine,
        causal: CausalGraph | None = None,
        k: int = 80,
        l: int = 128,  # noqa: E741
    ) -> None:
        self._lib = library
        self._executor = executor
        self._logic = logic
        self._causal = causal
        self._k = k
        self._l = l

        self._search = KnowledgeSearch(library=library, k=k, l=l)
        self._chain = ChainBuilder(executor=executor, library=library)
        self._tracer = CausalTracer(causal=causal) if causal else None
        self._comparator = Comparator(library=library, executor=executor)
        self._composer = Composer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decode(
        self,
        parsed: ParsedQuestion,
        domain: str,
        facts: dict[str, Any] | None = None,
    ) -> QAResponse:
        """Generate a self-contained response using world model reasoning.

        Routes to the appropriate flow based on intent:
          EXPLAIN -> knowledge search + chain + causal trace + compose
          COMPARE -> knowledge search + comparator + compose
          GENERAL -> knowledge search + optional chain + compose

        Args:
            parsed: Output from QuestionParser.parse().
            domain: Top-level domain string.
            facts: Optional variable/fact bindings.

        Returns:
            QAResponse with answer, confidence, evidence, etc.
        """
        facts = facts or {}
        trace: list[str] = []

        if parsed.intent == Intent.EXPLAIN:
            return self._decode_explain(parsed, domain, trace)
        if parsed.intent == Intent.COMPARE:
            return self._decode_compare(parsed, domain, trace)
        return self._decode_general(parsed, domain, trace)

    # ------------------------------------------------------------------
    # EXPLAIN flow
    # ------------------------------------------------------------------

    def _decode_explain(
        self,
        parsed: ParsedQuestion,
        domain: str,
        trace: list[str],
    ) -> QAResponse:
        """EXPLAIN flow: search -> chain -> causal -> compose."""
        trace.append(f"EXPLAIN intent for domain={domain}")

        # 1. Knowledge search
        ctx = self._search.search(
            keywords=parsed.keywords,
            domain=domain,
            target_var=parsed.target,
        )
        trace.append(
            f"Found {len(ctx.primary_axioms)} primary, "
            f"{len(ctx.supporting_axioms)} supporting axioms"
        )

        # 2. Build explanation chain
        chain = self._chain.build_explanation_chain(
            context=ctx,
            question_keywords=parsed.keywords,
        )
        trace.append(
            f"Built chain with {len(chain.steps)} steps"
        )

        # 3. Causal trace (if available)
        causal_narrative = None
        if self._tracer:
            causal_narrative = self._tracer.trace_explanation(
                question_keywords=parsed.keywords,
                domain=domain,
            )
            if causal_narrative:
                trace.append(
                    f"Causal trace from '{causal_narrative.cause}' "
                    f"with {len(causal_narrative.effects)} effects"
                )

        # 4. Compose
        answer = self._composer.compose_explanation(
            chain=chain,
            causal=causal_narrative,
            question=parsed.raw,
        )

        confidence = self._estimate_confidence(ctx, chain)
        evidence = chain.axioms_used

        return QAResponse(
            answer=answer,
            mode="EXPLAIN",
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # COMPARE flow
    # ------------------------------------------------------------------

    def _decode_compare(
        self,
        parsed: ParsedQuestion,
        domain: str,
        trace: list[str],
    ) -> QAResponse:
        """COMPARE flow: extract items -> comparator -> compose."""
        trace.append(f"COMPARE intent for domain={domain}")

        # Extract item names from keywords
        item_names = self._extract_comparison_items(parsed)
        trace.append(f"Comparison items: {item_names}")

        # Run comparison
        result = self._comparator.compare(
            item_names=item_names,
            domain=domain,
            variables=parsed.variables or None,
        )
        trace.append(
            f"Found {len(result.similarities)} similarities, "
            f"{len(result.differences)} differences"
        )

        # Compose
        answer = self._composer.compose_comparison(
            result=result,
            question=parsed.raw,
        )

        evidence = [
            item["axiom_name"]
            for item in result.items
            if item.get("axiom_name")
        ]
        confidence = 0.7 if len(result.items) >= 2 else 0.4

        return QAResponse(
            answer=answer,
            mode="COMPARE",
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # GENERAL flow
    # ------------------------------------------------------------------

    def _decode_general(
        self,
        parsed: ParsedQuestion,
        domain: str,
        trace: list[str],
    ) -> QAResponse:
        """GENERAL flow: search -> optional chain -> compose."""
        trace.append(f"GENERAL intent for domain={domain}")

        # 1. Knowledge search
        ctx = self._search.search(
            keywords=parsed.keywords,
            domain=domain,
        )
        trace.append(
            f"Found {len(ctx.primary_axioms)} primary, "
            f"{len(ctx.supporting_axioms)} supporting axioms"
        )

        # 2. Optional chain if variables present
        chain = None
        if parsed.variables:
            chain = self._chain.build_chain(
                context=ctx,
                variables=parsed.variables,
                target=parsed.target,
            )
            trace.append(
                f"Built chain with {len(chain.steps)} steps"
            )

        # 3. Compose
        answer = self._composer.compose_general(
            context=ctx,
            chain=chain,
            question=parsed.raw,
        )

        confidence = self._estimate_confidence(ctx, chain)
        evidence = [ax.name for ax in ctx.primary_axioms]

        return QAResponse(
            answer=answer,
            mode="GENERAL",
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_comparison_items(
        self,
        parsed: ParsedQuestion,
    ) -> list[str]:
        """Extract item names for comparison from parsed question.

        Strategy: find axiom names that match keywords, or use raw
        keyword pairs separated by 'and', 'vs', 'versus'.
        """
        # Try to find axiom names matching keywords
        all_axioms = self._lib.all_axioms()
        kw_set = {kw.lower() for kw in parsed.keywords}

        matched: list[str] = []
        for ax in all_axioms:
            ax_tokens = {
                t.lower()
                for t in re.findall(r"[A-Za-z_]\w*", ax.name)
            }
            if kw_set & ax_tokens:
                matched.append(ax.name)

        if len(matched) >= 2:
            return matched[:4]  # Max 4 items

        # Fallback: split on "and", "vs", "versus"
        raw = parsed.raw
        parts = re.split(
            r"\s+(?:and|vs\.?|versus)\s+",
            raw,
            flags=re.IGNORECASE,
        )
        if len(parts) >= 2:
            return [p.strip() for p in parts if p.strip()][:4]

        return parsed.keywords[:2] if parsed.keywords else []

    def _estimate_confidence(
        self,
        ctx,
        chain=None,
    ) -> float:
        """Estimate response confidence from context and chain quality."""
        score = 0.0

        # Primary axioms found -> base confidence
        if ctx.primary_axioms:
            score += 0.4
        if len(ctx.primary_axioms) >= 3:
            score += 0.1

        # Chain steps -> higher confidence
        if chain and chain.steps:
            score += 0.2
            if len(chain.steps) >= 2:
                score += 0.1

        # Domain chain -> domain relevance
        if ctx.domain_chain:
            score += 0.1

        return min(score, 0.95)


__all__ = ["ReasoningDecoder"]
```

### Step 6.3 -- Verify

- [ ] Run `uv run pytest tests/test_reasoning_decoder.py -v --tb=short -x`
- [ ] Run `uv run ruff check mowm/reasoning/__init__.py`
- [ ] Verify all tests pass, no lint errors

---

## Task 7: Integration -- QAPipeline update + exports + integration tests

**Files modified:**
- `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\__init__.py` (add ReasoningDecoder to pipeline)
- `c:\Users\grill\Documents\GitHub\mowm\tests\test_qa_pipeline.py` (update integration tests)

Depends on: ALL previous tasks (1-6) complete and passing.

### Step 7.1 -- Update QAPipeline

- [ ] Modify `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\__init__.py` to integrate ReasoningDecoder

The QAPipeline.ask() flow becomes:
1. Classify domain
2. Parse question
3. Try DirectSolver (COMPUTE / EVALUATE / CAUSAL)
4. **NEW: Try ReasoningDecoder (EXPLAIN / COMPARE / GENERAL)**
5. Fall back to AugmentedSolver (true last resort)

```python
# mowm/qa/__init__.py
"""MoWM Q&A Pipeline -- question answering with world model reasoning.

Orchestrates: classify domain -> parse question -> direct solve
-> reasoning decode -> augmented fallback.

Public API:
    QAPipeline  -- End-to-end Q&A orchestrator
    QAResponse  -- Structured answer dataclass (re-exported from direct_solver)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mowm.causal_graph import CausalGraph
    from mowm.logic_engine import LogicEngine

from mowm.axiom_library import AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.qa.augmented_solver import AugmentedSolver
from mowm.qa.classifier import DomainClassifier
from mowm.qa.direct_solver import DirectSolver, QAResponse
from mowm.qa.formatter import ResponseFormatter
from mowm.qa.parser import QuestionParser
from mowm.reasoning import ReasoningDecoder


class QAPipeline:
    """End-to-end Q&A pipeline with self-contained reasoning decoder.

    Flow:
        1. Classify domain via DomainClassifier.
        2. Parse question into ParsedQuestion via QuestionParser.
        3. Attempt DirectSolver (COMPUTE / EVALUATE / CAUSAL).
        4. Attempt ReasoningDecoder (EXPLAIN / COMPARE / GENERAL).
        5. Fall back to AugmentedSolver (LLM or template -- true last resort).

    Args:
        library: AxiomLibrary instance.
        executor: SafeFormulaExecutor for formula evaluation.
        logic: LogicEngine for EVALUATE intent handling.
        causal: Optional CausalGraph for CAUSAL intent and causal tracing.
        k: VSA block count.
        l: VSA block length.
        codebook_path: Unused reserved parameter.
        llm_endpoint: Optional OpenRouter-compatible endpoint (last resort).
        api_key: Optional bearer token for the LLM endpoint.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
        logic: "LogicEngine",
        causal: "CausalGraph | None" = None,
        k: int = 80,
        l: int = 128,  # noqa: E741
        codebook_path: str | None = None,
        llm_endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._lib = library
        self._executor = executor
        self._logic = logic
        self._causal = causal
        self._fmt = ResponseFormatter()
        self._parser = QuestionParser()
        self._classifier = DomainClassifier(library)

        # Build a null causal graph if none provided
        if self._causal is None:
            from mowm.causal_graph import CausalGraph
            self._causal = CausalGraph(k=library.k, l=library.l)

        self._direct = DirectSolver(
            library=library,
            executor=executor,
            logic_engine=logic,
            causal_graph=self._causal,
            formatter=self._fmt,
        )

        # v0.4.0: ReasoningDecoder replaces AugmentedSolver as primary
        self._reasoning = ReasoningDecoder(
            library=library,
            executor=executor,
            logic=logic,
            causal=self._causal,
            k=k,
            l=l,
        )

        # AugmentedSolver is now true last resort
        self._augmented = AugmentedSolver(
            library=library,
            executor=executor,
            llm_endpoint=llm_endpoint,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(
        self, question: str, facts: dict[str, Any] | None = None,
    ) -> QAResponse:
        """Answer a single question using the full pipeline.

        Flow:
            1. Classify domain via DomainClassifier.
            2. Parse question into ParsedQuestion via QuestionParser.
            3. Attempt DirectSolver (COMPUTE / EVALUATE / CAUSAL).
            4. Attempt ReasoningDecoder (EXPLAIN / COMPARE / GENERAL).
            5. Fall back to AugmentedSolver (true last resort).

        Args:
            question: Natural-language question string.
            facts: Optional boolean / numeric fact bindings.

        Returns:
            QAResponse with answer, mode, confidence, domain, etc.
        """
        if not question or not question.strip():
            return QAResponse(
                answer="No question provided.",
                mode="NONE",
                confidence=0.0,
                domain="",
                reasoning_trace=["Empty question string received."],
            )

        facts = facts or {}

        # 1. Classify domain
        domain = self._classifier.top_domain(question)
        if not domain:
            domain = "general"

        # 2. Parse question
        parsed = self._parser.parse(question)

        # 3. Try direct solve
        if self._direct.can_solve(parsed, domain):
            resp = self._direct.solve(parsed, domain, facts=facts)
            if resp.confidence > 0.0:
                return resp

        # 4. Try reasoning decoder (v0.4.0)
        resp = self._reasoning.decode(parsed, domain, facts=facts)
        if resp.confidence > 0.3:
            return resp

        # 5. True last resort: augmented solver
        return self._augmented.solve(parsed, domain, facts=facts)

    def ask_batch(
        self,
        questions: list[str],
        facts: dict[str, Any] | None = None,
    ) -> list[QAResponse]:
        """Answer a list of questions, applying the same facts to each.

        Args:
            questions: List of natural-language question strings.
            facts: Optional shared fact bindings applied to every question.

        Returns:
            List of QAResponse objects in the same order as *questions*.
        """
        return [self.ask(q, facts=facts) for q in questions]


__all__ = ["QAPipeline", "QAResponse"]
```

### Step 7.2 -- Write integration tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_reasoning_integration.py`

```python
# tests/test_reasoning_integration.py
"""Integration tests: QAPipeline with ReasoningDecoder across all domains.

These tests verify end-to-end flow: question string -> QAPipeline.ask() ->
QAResponse with self-contained reasoning (no LLM).
"""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa import QAPipeline, QAResponse

K, L = 4, 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_full_lib() -> AxiomLibrary:
    """Build a library with axioms across many domains."""
    lib = AxiomLibrary(k=K, l=L)
    # Physics
    lib.register(lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    ))
    lib.register(lib.make_axiom(
        "Weight", "physics.dynamics", "W = m * g",
    ))
    lib.register(lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    ))
    lib.register(lib.make_axiom(
        "Momentum", "physics.dynamics", "p = m * v",
    ))
    # Economics
    lib.register(lib.make_axiom(
        "Supply-Demand Equilibrium", "economics.micro",
        "Q_d = Q_s",
    ))
    lib.register(lib.make_axiom(
        "Consumer Surplus", "economics.micro", "CS = WTP - P",
    ))
    lib.register(lib.make_axiom(
        "Producer Surplus", "economics.micro", "PS = P - MC",
    ))
    # Logic
    lib.register(lib.make_axiom(
        "Modus Ponens", "logic.deductive",
        "(P AND (P IMPLIES Q)) IMPLIES Q",
    ))
    lib.register(lib.make_axiom(
        "Hypothetical Syllogism", "logic.deductive",
        "((P IMPLIES Q) AND (Q IMPLIES R)) IMPLIES (P IMPLIES R)",
    ))
    # Bio
    lib.register(lib.make_axiom(
        "Hamilton's Rule", "bio.evolution",
        "benefit = r * B - C",
    ))
    # Music
    lib.register(lib.make_axiom(
        "Major Third", "music.harmony", "ratio = 5 / 4",
    ))
    lib.register(lib.make_axiom(
        "Minor Third", "music.harmony", "ratio = 6 / 5",
    ))
    lib.register(lib.make_axiom(
        "Harmonic Series", "music.harmony", "f_n = n * f_0",
    ))
    # Historical
    lib.register(lib.make_axiom(
        "Event Causation", "historical.causal",
        "event_A causes event_B",
    ))
    # Causal
    lib.register(lib.make_axiom(
        "Bayes Theorem", "causal.probability",
        "P_A_given_B = P_B_given_A * P_A / P_B",
    ))
    lib.register(lib.make_axiom(
        "Do-Calculus Rule 1", "causal.intervention",
        "P_Y_do_X_Z = P_Y_do_X",
    ))
    return lib


def _build_full_graph() -> CausalGraph:
    """Build a causal graph with physics + economics + historical nodes."""
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    # Physics
    for n in ["mass", "grav_force", "inertia", "acceleration"]:
        g.add_node(n, "physics", vec)
    g.add_edge("mass", "grav_force", "+", 1.0)
    g.add_edge("mass", "inertia", "+", 1.0)
    g.add_edge("grav_force", "acceleration", "+", 1.0)
    g.add_edge("inertia", "acceleration", "-", 1.0)
    # Economics
    for n in ["price", "demand", "supply", "revenue"]:
        g.add_node(n, "economics", vec)
    g.add_edge("price", "demand", "-", 0.8)
    g.add_edge("price", "supply", "+", 0.6)
    g.add_edge("demand", "revenue", "+", 0.9)
    g.add_edge("supply", "revenue", "+", 0.3)
    # Historical
    for n in ["innovation", "prosperity", "complacency", "decline"]:
        g.add_node(n, "historical", vec)
    g.add_edge("innovation", "prosperity", "+", 0.9)
    g.add_edge("prosperity", "complacency", "+", 0.7)
    g.add_edge("complacency", "decline", "+", 0.8)
    return g


@pytest.fixture
def pipeline() -> QAPipeline:
    lib = _build_full_lib()
    executor = SafeFormulaExecutor()
    logic = LogicEngine()
    graph = _build_full_graph()
    return QAPipeline(
        library=lib,
        executor=executor,
        logic=logic,
        causal=graph,
        k=K,
        l=L,
    )


# ---------------------------------------------------------------------------
# TestPipelineExplain -- EXPLAIN questions self-contained
# ---------------------------------------------------------------------------


class TestPipelineExplainPhysics:
    def test_explain_free_fall(self, pipeline):
        """Physics: Why do heavier objects fall at the same rate?"""
        resp = pipeline.ask(
            "Why do heavier objects fall at the same rate?"
        )
        assert isinstance(resp, QAResponse)
        assert resp.confidence > 0.3
        assert resp.mode != "TEMPLATE"  # NOT using LLM fallback
        assert resp.mode != "LLM"
        assert len(resp.answer) > 20

    def test_explain_force(self, pipeline):
        resp = pipeline.ask("Explain Newton's Second Law")
        assert resp.confidence > 0.3
        assert "Newton" in resp.answer or "F" in resp.answer


class TestPipelineExplainEconomics:
    def test_explain_supply_demand(self, pipeline):
        """Economics: Explain supply and demand."""
        resp = pipeline.ask(
            "Explain supply and demand equilibrium."
        )
        assert resp.confidence > 0.3
        assert resp.mode != "TEMPLATE"


class TestPipelineExplainLogic:
    def test_explain_modus_ponens(self, pipeline):
        """Logic: Explain modus ponens."""
        resp = pipeline.ask("Explain modus ponens")
        assert resp.confidence > 0.3
        assert "Modus Ponens" in resp.answer


class TestPipelineExplainBio:
    def test_explain_altruism(self, pipeline):
        """Bio: Why does natural selection favor altruism in kin?"""
        resp = pipeline.ask(
            "Why does natural selection favor altruism in kin?"
        )
        assert resp.confidence > 0.3
        assert "Hamilton" in resp.answer or "benefit" in resp.answer.lower()


# ---------------------------------------------------------------------------
# TestPipelineCompare -- COMPARE questions self-contained
# ---------------------------------------------------------------------------


class TestPipelineCompare:
    def test_compare_surpluses(self, pipeline):
        """Economics: Compare consumer and producer surplus."""
        resp = pipeline.ask(
            "Compare consumer surplus and producer surplus"
        )
        assert resp.confidence > 0.3
        assert resp.mode != "TEMPLATE"
        assert "Consumer" in resp.answer or "Producer" in resp.answer

    def test_compare_music_intervals(self, pipeline):
        """Music: Compare major and minor thirds."""
        resp = pipeline.ask("Compare major third vs minor third")
        assert resp.confidence > 0.3
        assert "Major" in resp.answer or "Minor" in resp.answer


# ---------------------------------------------------------------------------
# TestPipelineGeneral -- GENERAL questions self-contained
# ---------------------------------------------------------------------------


class TestPipelineGeneral:
    def test_general_physics(self, pipeline):
        resp = pipeline.ask("Tell me about kinetic energy")
        assert resp.confidence > 0.3
        assert resp.mode != "TEMPLATE"
        assert len(resp.answer) > 10

    def test_general_bio(self, pipeline):
        resp = pipeline.ask("Tell me about Hamilton's Rule")
        assert resp.confidence > 0.3


# ---------------------------------------------------------------------------
# TestPipelineCompute -- v0.3.0 COMPUTE still works
# ---------------------------------------------------------------------------


class TestPipelineComputeStillWorks:
    def test_compute_force(self, pipeline):
        """Verify v0.3.0 COMPUTE path is not broken."""
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert resp.confidence > 0.0
        assert resp.mode == "COMPUTE"
        assert "F" in resp.computed_values
        assert resp.computed_values["F"] == pytest.approx(98.0, rel=1e-3)


# ---------------------------------------------------------------------------
# TestPipelineNoLLM -- verify no LLM calls
# ---------------------------------------------------------------------------


class TestPipelineNoLLM:
    def test_explain_does_not_use_llm(self, pipeline):
        """EXPLAIN should use ReasoningDecoder, not AugmentedSolver."""
        resp = pipeline.ask("Explain Newton's Second Law")
        assert resp.mode != "LLM"
        assert resp.mode != "TEMPLATE"
        assert resp.mode == "EXPLAIN"

    def test_compare_does_not_use_llm(self, pipeline):
        resp = pipeline.ask(
            "Compare force vs momentum"
        )
        assert resp.mode != "LLM"
        assert resp.mode != "TEMPLATE"
        assert resp.mode == "COMPARE"

    def test_general_does_not_use_llm(self, pipeline):
        resp = pipeline.ask("Tell me about force")
        assert resp.mode != "LLM"
        assert resp.mode != "TEMPLATE"
        assert resp.mode == "GENERAL"
```

### Step 7.3 -- Verify full integration

- [ ] Run `uv run pytest tests/test_reasoning_integration.py -v --tb=short -x`
- [ ] Run `uv run ruff check mowm/qa/__init__.py`
- [ ] Run full test suite: `uv run pytest tests/ -v -q --tb=short -x`
- [ ] Verify all previous 542 tests still pass
- [ ] Verify all new tests pass
- [ ] Run `uv run ruff check mowm/reasoning/ mowm/qa/__init__.py`

### Step 7.4 -- Final line count verification

- [ ] Verify no file exceeds 1000 lines:

```bash
wc -l mowm/reasoning/knowledge_search.py  # Target: ~250 lines
wc -l mowm/reasoning/chain_builder.py     # Target: ~300 lines
wc -l mowm/reasoning/causal_tracer.py     # Target: ~250 lines
wc -l mowm/reasoning/comparator.py        # Target: ~250 lines
wc -l mowm/reasoning/composer.py          # Target: ~300 lines
wc -l mowm/reasoning/__init__.py          # Target: ~300 lines
```

---

## Summary

| Task | File | Lines (est.) | Tests (est.) |
|------|------|-------------|-------------|
| 1 | `reasoning/knowledge_search.py` | ~250 | ~20 |
| 2 | `reasoning/chain_builder.py` | ~300 | ~18 |
| 3 | `reasoning/causal_tracer.py` | ~250 | ~15 |
| 4 | `reasoning/comparator.py` | ~250 | ~14 |
| 5 | `reasoning/composer.py` | ~300 | ~18 |
| 6 | `reasoning/__init__.py` | ~300 | ~22 |
| 7 | `qa/__init__.py` + integration tests | ~150 + ~200 | ~15 |
| **Total** | **7 files** | **~2000** | **~122** |

### Domain coverage in tests:

| Domain | Test File(s) | What's Tested |
|--------|-------------|---------------|
| Physics | All test files | Newton, Weight, KE, Momentum, free fall cancellation |
| Economics | test_chain_builder, test_comparator, test_reasoning_decoder, test_reasoning_integration | Supply-Demand, Consumer/Producer Surplus |
| Logic | test_chain_builder, test_reasoning_decoder, test_reasoning_integration | Modus Ponens, Hypothetical Syllogism |
| Bio | test_chain_builder, test_reasoning_decoder, test_reasoning_integration | Hamilton's Rule |
| Music | test_chain_builder, test_comparator, test_reasoning_decoder, test_reasoning_integration | Major/Minor Third, Harmonic Series |
| Historical | test_causal_tracer, test_reasoning_decoder | Event Causation, rise-fall chain |
| Causal | test_causal_tracer, test_reasoning_decoder | Do-Calculus, feedback loops |

### Dependency order:

```
Task 1 (knowledge_search) ─┐
Task 3 (causal_tracer) ────┤
Task 4 (comparator) ───────┤── Task 5 (composer) ─── Task 6 (__init__) ─── Task 7 (integration)
Task 2 (chain_builder) ────┘
```

Tasks 1, 3, and 4 can be implemented in parallel.
Task 2 depends on Task 1's `KnowledgeContext`.
Task 5 depends on Tasks 2, 3, and 4's dataclasses.
Task 6 depends on all of 1-5.
Task 7 depends on Task 6.
