# tests/test_direct_solver.py
"""Tests for DirectSolver — deterministic COMPUTE/EVALUATE/CAUSAL Q&A."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa.direct_solver import DirectSolver, QAResponse
from mowm.qa.formatter import ResponseFormatter
from mowm.qa.parser import Intent, QuestionParser

K, L = 4, 32  # small dims for fast testing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom("Newton's Second Law", "physics", "F = m * a"))
    lib.register(lib.make_axiom("Kinetic Energy", "physics", "KE = 0.5 * m * v^2"))
    lib.register(lib.make_axiom("De Morgan", "logic", "A AND B"))
    return lib


@pytest.fixture
def executor() -> SafeFormulaExecutor:
    return SafeFormulaExecutor()


@pytest.fixture
def logic() -> LogicEngine:
    return LogicEngine()


@pytest.fixture
def graph() -> CausalGraph:
    """CausalGraph with force → acceleration → velocity chain."""
    g = CausalGraph(k=K, l=L)
    # Add nodes (vectors don't matter for propagation tests)
    vec = np.zeros((K, L), dtype=np.float32)
    g.add_node("force", "physics", vec)
    g.add_node("acceleration", "physics", vec)
    g.add_node("velocity", "physics", vec)
    g.add_edge("force", "acceleration", polarity="+", strength=1.0)
    g.add_edge("acceleration", "velocity", polarity="+", strength=0.5)
    return g


@pytest.fixture
def empty_graph() -> CausalGraph:
    return CausalGraph(k=K, l=L)


@pytest.fixture
def fmt() -> ResponseFormatter:
    return ResponseFormatter()


@pytest.fixture
def solver(lib, executor, logic, graph, fmt) -> DirectSolver:
    return DirectSolver(lib, executor, logic, graph, fmt)


@pytest.fixture
def parser() -> QuestionParser:
    return QuestionParser()


# ---------------------------------------------------------------------------
# TestQAResponseDataclass
# ---------------------------------------------------------------------------


class TestQAResponseDataclass:
    def test_default_fields(self):
        r = QAResponse()
        assert r.answer == ""
        assert r.mode == ""
        assert r.confidence == 0.0
        assert r.domain == ""
        assert r.evidence == []
        assert r.computed_values == {}
        assert r.reasoning_trace == []

    def test_fields_set_correctly(self):
        r = QAResponse(
            answer="42",
            mode="COMPUTE",
            confidence=0.9,
            domain="physics",
            evidence=["Newton's Second Law"],
            computed_values={"F": 98.0},
            reasoning_trace=["step1"],
        )
        assert r.answer == "42"
        assert r.confidence == 0.9
        assert "Newton's Second Law" in r.evidence
        assert r.computed_values["F"] == pytest.approx(98.0)


# ---------------------------------------------------------------------------
# TestCompute — Newton's Second Law (F = m * a)
# ---------------------------------------------------------------------------


class TestComputeNewton:
    def test_can_solve_newton(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        assert solver.can_solve(pq, "physics") is True

    def test_solve_newton_result(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        resp = solver.solve(pq, "physics")
        assert resp.confidence > 0
        assert "F" in resp.computed_values
        assert resp.computed_values["F"] == pytest.approx(98.0, rel=1e-3)

    def test_solve_newton_mode(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        resp = solver.solve(pq, "physics")
        assert resp.mode == Intent.COMPUTE.value

    def test_solve_newton_answer_contains_axiom(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        resp = solver.solve(pq, "physics")
        assert "Newton" in resp.answer or "F" in resp.answer

    def test_solve_newton_evidence(self, solver, parser):
        pq = parser.parse("What is F given m=10 and a=9.8?")
        resp = solver.solve(pq, "physics")
        assert len(resp.evidence) > 0


class TestComputeKineticEnergy:
    def test_can_solve_ke(self, solver, parser):
        pq = parser.parse("What is KE given m=2 and v=3?")
        assert solver.can_solve(pq, "physics") is True

    def test_solve_ke_result(self, solver, parser):
        pq = parser.parse("What is KE given m=2 and v=3?")
        resp = solver.solve(pq, "physics")
        # KE = 0.5 * 2 * 9 = 9.0
        assert resp.confidence > 0
        assert resp.computed_values.get("KE") == pytest.approx(9.0, rel=1e-3)


class TestComputeMissingVars:
    def test_missing_variable_still_executes(self, solver, parser):
        # Only m provided; a defaults to 0.0 inside the executor
        pq = parser.parse("What is F given m=10?")
        resp = solver.solve(pq, "physics")
        # Should complete without raising; result is 0 (a defaults to 0)
        assert isinstance(resp, QAResponse)

    def test_no_target_cannot_solve(self, solver, parser):
        pq = parser.parse("Calculate force m=10 a=9.8")
        assert solver.can_solve(pq, "physics") is False

    def test_no_target_solve_returns_low_confidence(self, solver, parser):
        pq = parser.parse("Calculate force m=10 a=9.8")
        resp = solver.solve(pq, "physics")
        assert resp.confidence == 0.0


class TestComputeNoMatch:
    def test_unknown_target_cannot_solve(self, solver, parser):
        pq = parser.parse("What is FLUX given x=1?")
        assert solver.can_solve(pq, "physics") is False

    def test_unknown_target_solve_returns_low_confidence(self, solver, parser):
        pq = parser.parse("What is FLUX given x=1?")
        resp = solver.solve(pq, "physics")
        assert resp.confidence == 0.0


# ---------------------------------------------------------------------------
# TestEvaluate — obligation checks
# ---------------------------------------------------------------------------


class TestEvaluateObligation:
    def test_obligation_holds(self, solver, parser):
        pq = parser.parse("Check if signed_contract and paid_deposit")
        facts = {"signed_contract": True, "paid_deposit": True}
        assert solver.can_solve(pq, "logic") is True
        resp = solver.solve(pq, "logic", facts=facts)
        assert resp.confidence > 0
        assert resp.computed_values.get("holds") is True

    def test_obligation_fails(self, solver, parser):
        pq = parser.parse("Check if signed_contract and paid_deposit")
        facts = {"signed_contract": True, "paid_deposit": False}
        resp = solver.solve(pq, "logic", facts=facts)
        assert resp.computed_values.get("holds") is False

    def test_evaluate_mode_set(self, solver, parser):
        pq = parser.parse("Check if signed_contract and paid_deposit")
        facts = {"signed_contract": True, "paid_deposit": True}
        resp = solver.solve(pq, "logic", facts=facts)
        assert resp.mode == Intent.EVALUATE.value

    def test_evaluate_answer_contains_holds(self, solver, parser):
        pq = parser.parse("Check if signed_contract and paid_deposit")
        facts = {"signed_contract": True, "paid_deposit": True}
        resp = solver.solve(pq, "logic", facts=facts)
        assert "holds" in resp.answer


class TestEvaluateLogicFormula:
    def test_logic_formula_via_facts(self, solver, parser, lib):
        # No conditions extracted → falls back to logic formula evaluation
        pq = parser.parse("Is this valid?")
        pq.intent = Intent.EVALUATE  # force intent
        # logic domain has "A AND B" axiom
        facts = {"A": True, "B": True}
        resp = solver.solve(pq, "logic", facts=facts)
        assert isinstance(resp, QAResponse)

    def test_evaluate_no_conditions_no_facts_low_confidence(self, solver, parser):
        pq = parser.parse("Does it hold?")
        pq.intent = Intent.EVALUATE
        resp = solver.solve(pq, "logic", facts={})
        assert resp.confidence <= 0.5


# ---------------------------------------------------------------------------
# TestCausal — propagation
# ---------------------------------------------------------------------------


class TestCausalPropagation:
    def test_can_solve_causal(self, solver, parser):
        pq = parser.parse("force increases by 10")
        assert solver.can_solve(pq, "physics") is True

    def test_causal_effects_downstream(self, solver, parser):
        pq = parser.parse("force increases by 10")
        resp = solver.solve(pq, "physics")
        assert resp.confidence > 0
        assert "acceleration" in resp.computed_values

    def test_causal_mode_set(self, solver, parser):
        pq = parser.parse("force increases by 10")
        resp = solver.solve(pq, "physics")
        assert resp.mode == Intent.CAUSAL.value

    def test_causal_velocity_propagated(self, solver, parser):
        pq = parser.parse("force increases by 10")
        resp = solver.solve(pq, "physics")
        # acceleration→velocity with strength 0.5 → velocity gets 10*1.0*0.5 = 5.0
        assert "velocity" in resp.computed_values
        assert resp.computed_values["velocity"] == pytest.approx(5.0, rel=1e-3)


class TestCausalNoGraph:
    def test_unknown_node_returns_low_confidence(self, solver, parser):
        pq = parser.parse("entropy increases by 5")
        resp = solver.solve(pq, "physics")
        assert resp.confidence == 0.0

    def test_unknown_node_answer_mentions_node(self, solver, parser):
        pq = parser.parse("entropy increases by 5")
        resp = solver.solve(pq, "physics")
        assert "entropy" in resp.answer


# ---------------------------------------------------------------------------
# TestCannotSolveIntents
# ---------------------------------------------------------------------------


class TestExplainCannotSolve:
    def test_can_solve_false_for_explain(self, solver, parser):
        pq = parser.parse("Explain Newton's Second Law")
        assert solver.can_solve(pq, "physics") is False

    def test_solve_explain_returns_zero_confidence(self, solver, parser):
        pq = parser.parse("Explain Newton's Second Law")
        resp = solver.solve(pq, "physics")
        assert resp.confidence == 0.0


class TestGeneralCannotSolve:
    def test_can_solve_false_for_general(self, solver, parser):
        pq = parser.parse("Tell me about physics")
        assert solver.can_solve(pq, "physics") is False

    def test_solve_general_returns_zero_confidence(self, solver, parser):
        pq = parser.parse("Tell me about physics")
        resp = solver.solve(pq, "physics")
        assert resp.confidence == 0.0


class TestCompareCannotSolve:
    def test_can_solve_false_for_compare(self, solver, parser):
        pq = parser.parse("Compare car_A vs car_B")
        assert solver.can_solve(pq, "physics") is False

    def test_solve_compare_returns_zero_confidence(self, solver, parser):
        pq = parser.parse("Compare car_A vs car_B")
        resp = solver.solve(pq, "physics")
        assert resp.confidence == 0.0
