# tests/test_parser.py
"""Tests for QuestionParser — regex intent/variable/causal/condition extraction."""
from __future__ import annotations

import pytest

from mowm.qa.parser import Intent, ParsedQuestion, QuestionParser


@pytest.fixture
def parser() -> QuestionParser:
    return QuestionParser()


# ---------------------------------------------------------------------------
# TestParseIntent
# ---------------------------------------------------------------------------


class TestParseIntent:
    def test_compute_intent(self, parser):
        pq = parser.parse("Calculate F given m=5 and a=2")
        assert pq.intent == Intent.COMPUTE

    def test_evaluate_intent(self, parser):
        pq = parser.parse("Does the condition hold if signed_contract and paid_deposit?")
        assert pq.intent == Intent.EVALUATE

    def test_causal_intent(self, parser):
        pq = parser.parse("What happens if force increases by 10?")
        assert pq.intent == Intent.CAUSAL

    def test_explain_intent(self, parser):
        pq = parser.parse("Explain Newton's Second Law")
        assert pq.intent == Intent.EXPLAIN

    def test_compare_intent(self, parser):
        pq = parser.parse("Compare car_A vs car_B")
        assert pq.intent == Intent.COMPARE

    def test_general_intent_fallback(self, parser):
        pq = parser.parse("Tell me something interesting about physics")
        assert pq.intent == Intent.GENERAL

    def test_causal_beats_compute(self, parser):
        pq = parser.parse("Calculate what happens when temperature increases by 5")
        assert pq.intent == Intent.CAUSAL


# ---------------------------------------------------------------------------
# TestParseVariables
# ---------------------------------------------------------------------------


class TestParseVariables:
    def test_integer_variable(self, parser):
        pq = parser.parse("m=10")
        assert pq.variables["m"] == pytest.approx(10.0)

    def test_float_variable(self, parser):
        pq = parser.parse("a = 9.8")
        assert pq.variables["a"] == pytest.approx(9.8)

    def test_negative_variable(self, parser):
        pq = parser.parse("x=-5.0")
        assert pq.variables["x"] == pytest.approx(-5.0)

    def test_scientific_notation(self, parser):
        pq = parser.parse("m=1e3")
        assert pq.variables["m"] == pytest.approx(1000.0)

    def test_multiple_variables(self, parser):
        pq = parser.parse("F = m * a, where m=10 and a=9.8")
        assert "m" in pq.variables
        assert "a" in pq.variables

    def test_no_variables(self, parser):
        pq = parser.parse("What is the speed of light?")
        assert pq.variables == {}


# ---------------------------------------------------------------------------
# TestParseTarget
# ---------------------------------------------------------------------------


class TestParseTarget:
    def test_what_is_target(self, parser):
        pq = parser.parse("What is F given m=5 and a=2?")
        assert pq.target == "F"

    def test_no_target_in_general_question(self, parser):
        pq = parser.parse("Calculate force")
        assert pq.target is None

    def test_target_case_preserved(self, parser):
        pq = parser.parse("What is velocity?")
        assert pq.target == "velocity"


# ---------------------------------------------------------------------------
# TestParseCausal
# ---------------------------------------------------------------------------


class TestParseCausal:
    def test_increases_positive_delta(self, parser):
        pq = parser.parse("force increases by 10")
        assert pq.causal_node == "force"
        assert pq.causal_delta == pytest.approx(10.0)

    def test_decreases_negative_delta(self, parser):
        pq = parser.parse("temperature decreases by 3.5")
        assert pq.causal_node == "temperature"
        assert pq.causal_delta == pytest.approx(-3.5)

    def test_no_causal_info(self, parser):
        pq = parser.parse("What is F?")
        assert pq.causal_node is None
        assert pq.causal_delta is None

    def test_causal_node_trimmed(self, parser):
        pq = parser.parse("the mass increases by 2")
        assert pq.causal_node is not None
        assert pq.causal_node.strip() == pq.causal_node


# ---------------------------------------------------------------------------
# TestParseConditions
# ---------------------------------------------------------------------------


class TestParseConditions:
    def test_single_condition(self, parser):
        pq = parser.parse("Check if signed_contract")
        assert "signed_contract" in pq.conditions

    def test_multiple_conditions_split_by_and(self, parser):
        pq = parser.parse("Evaluate if signed_contract and paid_deposit")
        assert "signed_contract" in pq.conditions
        assert "paid_deposit" in pq.conditions

    def test_no_conditions_when_no_if(self, parser):
        pq = parser.parse("Calculate F where m=5 a=2")
        assert pq.conditions == []

    def test_three_conditions(self, parser):
        pq = parser.parse("Check if alpha and beta and gamma")
        assert len(pq.conditions) == 3


# ---------------------------------------------------------------------------
# TestParseKeywords
# ---------------------------------------------------------------------------


class TestParseKeywords:
    def test_stopwords_excluded(self, parser):
        pq = parser.parse("What is the force of gravity?")
        stopwords = {"what", "is", "the", "of"}
        for kw in pq.keywords:
            assert kw.lower() not in stopwords

    def test_keywords_no_duplicates(self, parser):
        pq = parser.parse("force force force")
        assert pq.keywords.count("force") == 1

    def test_keywords_not_empty_for_real_question(self, parser):
        pq = parser.parse("Calculate gravitational force")
        assert len(pq.keywords) > 0


# ---------------------------------------------------------------------------
# TestParsedQuestionDataclass
# ---------------------------------------------------------------------------


class TestParsedQuestionDataclass:
    def test_raw_field_preserved(self, parser):
        q = "Calculate F given m=5 and a=2"
        pq = parser.parse(q)
        assert pq.raw == q

    def test_returns_parsed_question_instance(self, parser):
        pq = parser.parse("What is velocity?")
        assert isinstance(pq, ParsedQuestion)

    def test_default_intent_is_general(self):
        pq = ParsedQuestion(raw="hello")
        assert pq.intent == Intent.GENERAL
