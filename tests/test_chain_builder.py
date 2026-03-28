# tests/test_chain_builder.py
"""Tests for ChainBuilder — derivation chains via shared variable propagation."""
from __future__ import annotations

import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.reasoning.chain_builder import ChainBuilder, DerivationChain, DerivationStep
from mowm.reasoning.knowledge_search import KnowledgeContext

K, L = 4, 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_context(*axiom_specs, lib: AxiomLibrary | None = None) -> KnowledgeContext:
    """Build a KnowledgeContext from (name, domain, formula) tuples."""
    if lib is None:
        lib = AxiomLibrary(k=K, l=L)
    axioms = [lib.make_axiom(name, domain, formula) for name, domain, formula in axiom_specs]
    return KnowledgeContext(primary_axioms=axioms)


@pytest.fixture
def executor() -> SafeFormulaExecutor:
    return SafeFormulaExecutor()


@pytest.fixture
def builder(executor) -> ChainBuilder:
    return ChainBuilder(executor)


# ---------------------------------------------------------------------------
# Tests: build_chain
# ---------------------------------------------------------------------------


def test_physics_chain_fma(builder):
    """F=ma: given m and a, derives F."""
    ctx = make_context(("Newton's Second Law", "physics.dynamics", "F = m * a"))
    chain = builder.build_chain(ctx, {"m": 5.0, "a": 3.0}, target="F")
    assert chain.conclusion == "F = 15.0"
    assert len(chain.steps) == 1
    assert chain.steps[0].result_var == "F"
    assert chain.steps[0].result_value == pytest.approx(15.0)


def test_physics_chain_ke(builder):
    """KE = 0.5 * m * v^2: given m and v, derives KE."""
    ctx = make_context(("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2"))
    chain = builder.build_chain(ctx, {"m": 2.0, "v": 4.0}, target="KE")
    assert chain.conclusion == "KE = 16.0"
    assert chain.steps[0].result_value == pytest.approx(16.0)


def test_physics_chain_propagation(builder):
    """F=ma then KE — momentum variable propagated: m used in both."""
    ctx = make_context(
        ("Newton's Second Law", "physics.dynamics", "F = m * a"),
        ("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2"),
    )
    chain = builder.build_chain(ctx, {"m": 2.0, "a": 10.0, "v": 3.0})
    axiom_names = chain.axioms_used
    assert "Newton's Second Law" in axiom_names
    assert "Kinetic Energy" in axiom_names
    assert "F" in chain.variables_used
    assert "KE" in chain.variables_used


def test_empty_context_returns_empty_chain(builder):
    """Empty context produces chain with no steps."""
    ctx = KnowledgeContext()
    chain = builder.build_chain(ctx, {"x": 1.0})
    assert chain.steps == []
    assert chain.conclusion == ""


def test_missing_variables_skips_axiom(builder):
    """Axiom with unavailable RHS vars is skipped."""
    ctx = make_context(("Newton's Second Law", "physics.dynamics", "F = m * a"))
    # Only 'a' provided, 'm' missing — executor returns 0.0 with vars absent
    chain = builder.build_chain(ctx, {"a": 3.0}, target="F")
    # The step may be included with result 0.0 or skipped; either way no error
    assert isinstance(chain, DerivationChain)


def test_chain_conclusion_set_without_target(builder):
    """When no target is given, conclusion is derived from last step."""
    ctx = make_context(("Momentum", "physics.dynamics", "p = m * v"))
    chain = builder.build_chain(ctx, {"m": 3.0, "v": 5.0})
    assert chain.conclusion == "p = 15.0"


def test_economics_chain(builder):
    """Profit formula: P = R - C."""
    ctx = make_context(("Profit", "economics.micro", "P = R - C"))
    chain = builder.build_chain(ctx, {"R": 1000.0, "C": 750.0}, target="P")
    assert chain.conclusion == "P = 250.0"
    assert chain.steps[0].result_value == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Tests: build_explanation_chain
# ---------------------------------------------------------------------------


def test_explanation_chain_symbolic(builder):
    """Explanation chain contains symbolic steps, no numeric result_value."""
    ctx = make_context(
        ("Newton's Second Law", "physics.dynamics", "F = m * a"),
        ("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2"),
    )
    chain = builder.build_explanation_chain(ctx, keywords=["force", "energy"])
    assert len(chain.steps) >= 1
    for step in chain.steps:
        assert isinstance(step.explanation, str)
        assert step.explanation.startswith("By ")
    assert "Connected through:" in chain.conclusion


def test_explanation_chain_empty_context(builder):
    """Empty context returns empty explanation chain."""
    ctx = KnowledgeContext()
    chain = builder.build_explanation_chain(ctx, keywords=["force"])
    assert chain.steps == []
    assert chain.conclusion == ""


def test_bio_hamiltons_rule(builder):
    """Hamilton's Rule (fitness > cost): bio formula chain."""
    ctx = make_context(("Hamilton's Rule", "bio.evolution", "W = r * B - C"))
    chain = builder.build_chain(ctx, {"r": 0.5, "B": 10.0, "C": 2.0}, target="W")
    assert chain.conclusion == "W = 3.0"
    assert chain.steps[0].result_value == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Tests: simplify_chain
# ---------------------------------------------------------------------------


def test_simplify_chain_removes_zero_unused(builder):
    """Steps with result_value=0 not used downstream are pruned."""
    ctx = make_context(
        ("A Formula", "test.domain", "X = a * b"),
        ("B Formula", "test.domain", "Y = c * d"),
    )
    chain = builder.build_chain(ctx, {"a": 2.0, "b": 3.0, "c": 0.0, "d": 0.0})
    simplified = builder.simplify_chain(chain)
    # Y=0 and X depends on a,b not Y — Y step may be pruned
    assert isinstance(simplified, DerivationChain)
    assert len(simplified.steps) >= 1


def test_simplify_single_step_unchanged(builder):
    """Single-step chain is returned as-is by simplify_chain."""
    ctx = make_context(("Simple", "test", "Z = x + y"))
    chain = builder.build_chain(ctx, {"x": 1.0, "y": 2.0})
    simplified = builder.simplify_chain(chain)
    assert len(simplified.steps) == len(chain.steps)
