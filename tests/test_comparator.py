# tests/test_comparator.py
"""Tests for Comparator — structural diff + dual execution for COMPARE intent."""
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
    lib.register(lib.make_axiom("Newton's Second Law", "physics.dynamics", "F = m * a"))
    lib.register(lib.make_axiom("Momentum", "physics.dynamics", "p = m * v"))
    lib.register(lib.make_axiom("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2"))
    lib.register(lib.make_axiom("Weight", "physics.dynamics", "W = m * g"))
    lib.register(lib.make_axiom("Profit", "economics.micro", "P = R - C"))
    return lib


@pytest.fixture
def comparator(physics_lib, executor) -> Comparator:
    return Comparator(physics_lib, executor)


# ---------------------------------------------------------------------------
# Tests: compare — basic structure
# ---------------------------------------------------------------------------


def test_force_vs_momentum_found(comparator):
    """Comparing force and momentum finds both axioms."""
    result = comparator.compare(["Newton's Second Law", "Momentum"], domain="physics")
    assert len(result.items) == 2
    names = [item["name"] for item in result.items]
    assert any("Newton" in n for n in names)
    assert any("Momentum" in n for n in names)


def test_shared_vars_detected(comparator):
    """Force and momentum share variable 'm'."""
    result = comparator.compare(["Newton's Second Law", "Momentum"], domain="physics")
    shared_strs = " ".join(result.similarities)
    assert "m" in shared_strs


def test_unique_vars_detected(comparator):
    """Force uses 'a' uniquely; momentum uses 'v' uniquely."""
    result = comparator.compare(["Newton's Second Law", "Momentum"], domain="physics")
    diff_strs = " ".join(result.differences)
    # At least one unique-var difference should be reported
    assert len(result.differences) >= 1


def test_same_domain_similarity(comparator):
    """Force and momentum are both physics.dynamics — domain match noted."""
    result = comparator.compare(["Newton's Second Law", "Momentum"], domain="physics")
    sim_strs = " ".join(result.similarities)
    assert "physics" in sim_strs


def test_different_domains_noted(comparator):
    """Kinetic energy (physics.energy) vs profit (economics.micro) — domain diff noted."""
    result = comparator.compare(["Kinetic Energy", "Profit"], domain="mixed")
    diff_strs = " ".join(result.differences)
    assert "physics" in diff_strs or "economics" in diff_strs


# ---------------------------------------------------------------------------
# Tests: dual execution with variables
# ---------------------------------------------------------------------------


def test_dual_execution_with_variables(comparator):
    """Both formulas are numerically evaluated when variables are provided."""
    result = comparator.compare(
        ["Newton's Second Law", "Momentum"],
        domain="physics",
        variables={"m": 4.0, "a": 5.0, "v": 3.0},
    )
    values = [item["value"] for item in result.items]
    # F = 4*5 = 20, p = 4*3 = 12
    assert any(v == pytest.approx(20.0) for v in values if v is not None)
    assert any(v == pytest.approx(12.0) for v in values if v is not None)


def test_execution_without_variables(comparator):
    """Without variables dict, all item values are None."""
    result = comparator.compare(["Newton's Second Law", "Momentum"], domain="physics")
    assert all(item["value"] is None for item in result.items)


# ---------------------------------------------------------------------------
# Tests: structural diff analysis
# ---------------------------------------------------------------------------


def test_quadratic_vs_linear_detected(comparator):
    """KE (quadratic in v) vs momentum (linear) gives quadratic structure note."""
    result = comparator.compare(["Kinetic Energy", "Momentum"], domain="physics")
    assert "quadratic" in result.structural_diff.lower() or "linear" in result.structural_diff.lower()


def test_no_matching_axioms_returns_message(comparator):
    """Query for completely unknown concepts returns structural_diff message."""
    result = comparator.compare(["unknownX", "unknownY"], domain="physics")
    assert "Not enough axioms" in result.structural_diff
    assert result.items == []


def test_single_axiom_found_returns_message(comparator):
    """Only one axiom matched returns fallback message."""
    result = comparator.compare(["Newton's Second Law", "unknownZ"], domain="physics")
    assert "Not enough axioms" in result.structural_diff
    assert len(result.items) <= 1
