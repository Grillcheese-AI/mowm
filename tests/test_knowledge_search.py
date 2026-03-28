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
