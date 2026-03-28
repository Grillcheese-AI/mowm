# tests/test_reasoning_decoder.py
"""Tests for ReasoningDecoder — end-to-end reasoning across all domains."""
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
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _build_lib() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    # Physics
    lib.register(lib.make_axiom("Newton's Second Law", "physics.dynamics", "F = m * a"))
    lib.register(lib.make_axiom("Weight", "physics.dynamics", "W = m * g"))
    lib.register(lib.make_axiom("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2"))
    lib.register(lib.make_axiom("Momentum", "physics.dynamics", "p = m * v"))
    # Economics
    lib.register(lib.make_axiom("Supply-Demand Equilibrium", "economics.micro", "Q_d = Q_s"))
    lib.register(lib.make_axiom("Consumer Surplus", "economics.micro", "CS = WTP - P"))
    lib.register(lib.make_axiom("Producer Surplus", "economics.micro", "PS = P - MC"))
    # Logic
    lib.register(lib.make_axiom(
        "Modus Ponens", "logic.deductive", "(P AND (P IMPLIES Q)) IMPLIES Q",
    ))
    # Bio
    lib.register(lib.make_axiom("Hamilton's Rule", "bio.evolution", "benefit = r * B - C"))
    # Music
    lib.register(lib.make_axiom("Major Third", "music.harmony", "ratio = 5 / 4"))
    lib.register(lib.make_axiom("Minor Third", "music.harmony", "ratio = 6 / 5"))
    lib.register(lib.make_axiom("Harmonic Series", "music.harmony", "f_n = n * f_0"))
    # Causal / historical
    lib.register(lib.make_axiom(
        "Do-Calculus Rule 1", "causal.intervention", "P_Y_do_X_Z = P_Y_do_X",
    ))
    lib.register(lib.make_axiom(
        "Bayes Theorem", "causal.probability",
        "P_A_given_B = P_B_given_A * P_A / P_B",
    ))
    return lib


def _build_graph() -> CausalGraph:
    g = CausalGraph(k=K, l=L)
    vec = np.zeros((K, L), dtype=np.float32)
    for name in ["mass", "grav_force", "inertia", "acceleration"]:
        g.add_node(name, "physics", vec)
    g.add_edge("mass", "grav_force", polarity="+", strength=1.0)
    g.add_edge("mass", "inertia", polarity="+", strength=1.0)
    g.add_edge("grav_force", "acceleration", polarity="+", strength=1.0)
    g.add_edge("inertia", "acceleration", polarity="-", strength=1.0)
    for name in ["price", "demand", "supply"]:
        g.add_node(name, "economics", vec)
    g.add_edge("price", "demand", polarity="-", strength=0.8)
    g.add_edge("price", "supply", polarity="+", strength=0.6)
    return g


def _make_parsed(
    raw: str,
    intent: Intent,
    keywords: list[str] | None = None,
    variables: dict[str, float] | None = None,
) -> ParsedQuestion:
    pq = ParsedQuestion(raw=raw)
    pq.intent = intent
    pq.keywords = keywords or []
    pq.variables = variables or {}
    return pq


@pytest.fixture
def lib() -> AxiomLibrary:
    return _build_lib()


@pytest.fixture
def decoder(lib) -> ReasoningDecoder:
    executor = SafeFormulaExecutor()
    logic = LogicEngine()
    graph = _build_graph()
    return ReasoningDecoder(
        library=lib,
        executor=executor,
        logic=logic,
        causal=graph,
        k=K,
        l=L,
    )


@pytest.fixture
def decoder_no_causal(lib) -> ReasoningDecoder:
    return ReasoningDecoder(
        library=lib,
        executor=SafeFormulaExecutor(),
        logic=LogicEngine(),
        causal=None,
        k=K,
        l=L,
    )


# ---------------------------------------------------------------------------
# TestReasoningDecoderInit
# ---------------------------------------------------------------------------


class TestReasoningDecoderInit:
    def test_init_creates_decoder(self, decoder):
        assert isinstance(decoder, ReasoningDecoder)

    def test_init_without_causal(self, decoder_no_causal):
        assert isinstance(decoder_no_causal, ReasoningDecoder)


# ---------------------------------------------------------------------------
# TestExplain -- Physics
# ---------------------------------------------------------------------------


class TestExplainPhysics:
    def test_explain_free_fall_returns_response(self, decoder):
        pq = _make_parsed(
            raw="Why do heavier objects fall at the same rate?",
            intent=Intent.EXPLAIN,
            keywords=["heavier", "objects", "fall", "rate", "mass", "acceleration"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert isinstance(resp, QAResponse)
        assert resp.confidence > 0.3
        assert resp.mode == "EXPLAIN"
        assert len(resp.answer) > 20

    def test_explain_answer_mentions_physics(self, decoder):
        pq = _make_parsed(
            raw="Why do heavier objects fall at the same rate?",
            intent=Intent.EXPLAIN,
            keywords=["mass", "fall", "acceleration"],
        )
        resp = decoder.decode(pq, domain="physics")
        answer_lower = resp.answer.lower()
        assert (
            "weight" in answer_lower
            or "newton" in answer_lower
            or "mass" in answer_lower
            or "m" in resp.answer
        )

    def test_explain_has_evidence(self, decoder):
        pq = _make_parsed(
            raw="Explain force and acceleration.",
            intent=Intent.EXPLAIN,
            keywords=["force", "acceleration"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert len(resp.evidence) > 0

    def test_explain_has_reasoning_trace(self, decoder):
        pq = _make_parsed(
            raw="Explain Newton's second law.",
            intent=Intent.EXPLAIN,
            keywords=["newton", "force", "mass"],
        )
        resp = decoder.decode(pq, domain="physics")
        assert len(resp.reasoning_trace) > 0


# ---------------------------------------------------------------------------
# TestExplain -- Economics
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
# TestExplain -- Logic
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
# TestExplain -- Bio
# ---------------------------------------------------------------------------


class TestExplainBio:
    def test_explain_altruism(self, decoder):
        pq = _make_parsed(
            raw="Why does natural selection favor altruism in kin?",
            intent=Intent.EXPLAIN,
            keywords=["altruism", "kin", "Hamilton"],
        )
        resp = decoder.decode(pq, domain="bio")
        assert resp.confidence > 0.3
        assert (
            "Hamilton" in resp.answer
            or "benefit" in resp.answer.lower()
        )


# ---------------------------------------------------------------------------
# TestExplain -- Causal
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
            "Do-Calculus" in resp.answer
            or "P_Y_do_X" in resp.answer
            or "calculus" in resp.answer.lower()
        )


# ---------------------------------------------------------------------------
# TestCompare -- Physics
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
# TestCompare -- Economics
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
# TestCompare -- Music
# ---------------------------------------------------------------------------


class TestCompareMusic:
    def test_compare_thirds(self, decoder):
        pq = _make_parsed(
            raw="Compare major and minor thirds.",
            intent=Intent.COMPARE,
            keywords=["major", "minor", "third"],
        )
        resp = decoder.decode(pq, domain="music")
        assert resp.confidence > 0.3
        assert "Major" in resp.answer or "Minor" in resp.answer


# ---------------------------------------------------------------------------
# TestGeneral
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
            keywords=["Hamilton", "Rule"],
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
# TestFallback
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
        assert resp.answer != ""

    def test_empty_keywords_still_returns(self, decoder):
        pq = _make_parsed(
            raw="Explain.",
            intent=Intent.EXPLAIN,
            keywords=[],
        )
        resp = decoder.decode(pq, domain="physics")
        assert isinstance(resp, QAResponse)
