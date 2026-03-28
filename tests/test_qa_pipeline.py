# tests/test_qa_pipeline.py
"""Tests for QAPipeline — orchestrated Q&A with direct-first, augmented-fallback flow."""
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


@pytest.fixture
def lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    lib.register(lib.make_axiom("Newton's Second Law", "physics", "F = m * a"))
    lib.register(lib.make_axiom("Kinetic Energy", "physics", "KE = 0.5 * m * v^2"))
    lib.register(lib.make_axiom("Obligation Rule", "logic", "A AND B"))
    return lib


@pytest.fixture
def executor() -> SafeFormulaExecutor:
    return SafeFormulaExecutor()


@pytest.fixture
def logic() -> LogicEngine:
    return LogicEngine()


@pytest.fixture
def causal() -> CausalGraph:
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    g.add_node("force", "physics", vec)
    g.add_node("acceleration", "physics", vec)
    g.add_node("velocity", "physics", vec)
    g.add_edge("force", "acceleration", polarity="+", strength=1.0)
    g.add_edge("acceleration", "velocity", polarity="+", strength=0.5)
    return g


@pytest.fixture
def pipeline(lib, executor, logic, causal) -> QAPipeline:
    return QAPipeline(lib, executor, logic, causal=causal)


# ---------------------------------------------------------------------------
# TestDirectMode
# ---------------------------------------------------------------------------


class TestDirectModeNewton:
    def test_newton_returns_qa_response(self, pipeline):
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert isinstance(resp, QAResponse)

    def test_newton_high_confidence(self, pipeline):
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert resp.confidence > 0.5

    def test_newton_mode_is_compute(self, pipeline):
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert resp.mode == "COMPUTE"

    def test_newton_computed_value(self, pipeline):
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert resp.computed_values.get("F") == pytest.approx(98.0, rel=1e-3)


class TestDirectModeKE:
    def test_ke_returns_result(self, pipeline):
        resp = pipeline.ask("What is KE given m=2 and v=3?")
        assert resp.confidence > 0
        assert resp.computed_values.get("KE") == pytest.approx(9.0, rel=1e-3)


class TestDirectModeObligation:
    def test_obligation_holds(self, pipeline):
        resp = pipeline.ask(
            "Check if signed_contract and paid_deposit",
            facts={"signed_contract": True, "paid_deposit": True},
        )
        assert resp.mode == "EVALUATE"
        assert resp.computed_values.get("holds") is True

    def test_obligation_fails(self, pipeline):
        resp = pipeline.ask(
            "Check if signed_contract and paid_deposit",
            facts={"signed_contract": True, "paid_deposit": False},
        )
        assert resp.computed_values.get("holds") is False


class TestDirectModeCausal:
    def test_causal_propagates(self, pipeline):
        resp = pipeline.ask("force increases by 10")
        assert resp.mode == "CAUSAL"
        assert resp.confidence > 0
        assert "acceleration" in resp.computed_values

    def test_causal_velocity_downstream(self, pipeline):
        resp = pipeline.ask("force increases by 10")
        assert "velocity" in resp.computed_values
        assert resp.computed_values["velocity"] == pytest.approx(5.0, rel=1e-3)


class TestDirectModeEvidence:
    def test_evidence_list_populated(self, pipeline):
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert len(resp.evidence) > 0

    def test_reasoning_trace_populated(self, pipeline):
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert len(resp.reasoning_trace) > 0


# ---------------------------------------------------------------------------
# TestAugmentedMode
# ---------------------------------------------------------------------------


class TestAugmentedModeExplain:
    def test_explain_falls_back_to_augmented(self, pipeline):
        resp = pipeline.ask("Explain Newton's Second Law")
        assert isinstance(resp, QAResponse)
        # EXPLAIN cannot be solved by DirectSolver → augmented
        assert resp.mode in ("TEMPLATE", "LLM")

    def test_explain_has_answer(self, pipeline):
        resp = pipeline.ask("Explain Newton's Second Law")
        assert len(resp.answer) > 0


class TestAugmentedModeCompare:
    def test_compare_falls_back_to_augmented(self, pipeline):
        resp = pipeline.ask("Compare Newton's Second Law vs Kinetic Energy")
        assert resp.mode in ("TEMPLATE", "LLM")

    def test_compare_answer_non_empty(self, pipeline):
        resp = pipeline.ask("Compare Newton's Second Law vs Kinetic Energy")
        assert len(resp.answer) > 0


class TestAugmentedModeGeneral:
    def test_general_question_gets_answer(self, pipeline):
        resp = pipeline.ask("Tell me about the world model")
        assert isinstance(resp, QAResponse)
        assert len(resp.answer) > 0


# ---------------------------------------------------------------------------
# TestBatch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_multiple_questions(self, pipeline):
        questions = [
            "What is F given m=5 and a=2?",
            "Explain kinetic energy",
        ]
        results = pipeline.ask_batch(questions)
        assert len(results) == 2
        assert all(isinstance(r, QAResponse) for r in results)

    def test_batch_empty_returns_empty(self, pipeline):
        results = pipeline.ask_batch([])
        assert results == []

    def test_batch_single_question(self, pipeline):
        results = pipeline.ask_batch(["What is KE given m=2 and v=3?"])
        assert len(results) == 1
        assert results[0].computed_values.get("KE") == pytest.approx(9.0, rel=1e-3)

    def test_batch_shared_facts(self, pipeline):
        questions = [
            "Check if signed_contract and paid_deposit",
            "Evaluate if signed_contract and paid_deposit",
        ]
        facts = {"signed_contract": True, "paid_deposit": True}
        results = pipeline.ask_batch(questions, facts=facts)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_question_returns_response(self, pipeline):
        resp = pipeline.ask("")
        assert isinstance(resp, QAResponse)
        assert resp.confidence == 0.0

    def test_whitespace_only_question(self, pipeline):
        resp = pipeline.ask("   ")
        assert isinstance(resp, QAResponse)
        assert resp.confidence == 0.0

    def test_nonsense_question_gets_answer(self, pipeline):
        resp = pipeline.ask("zzzzxxx aaa bbb ccc")
        assert isinstance(resp, QAResponse)
        assert len(resp.answer) > 0

    def test_unknown_domain_defaults_gracefully(self, pipeline):
        # Question has no domain-matching keywords
        resp = pipeline.ask("quasar magnetosphere plasma resonance frequency?")
        assert isinstance(resp, QAResponse)

    def test_confidence_between_0_and_1(self, pipeline):
        resp = pipeline.ask("What is F given m=10 and a=9.8?")
        assert 0.0 <= resp.confidence <= 1.0


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

    def test_qa_response_importable_from_qa(self):
        from mowm.qa import QAResponse as QR
        assert QR is QAResponse

    def test_fields_customisable(self):
        r = QAResponse(
            answer="test",
            mode="COMPUTE",
            confidence=0.9,
            domain="physics",
            evidence=["Newton's Second Law"],
            computed_values={"F": 98.0},
            reasoning_trace=["step1"],
        )
        assert r.confidence == pytest.approx(0.9)
        assert "Newton's Second Law" in r.evidence
