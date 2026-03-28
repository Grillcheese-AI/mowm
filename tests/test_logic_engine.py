# tests/test_logic_engine.py
"""Tests for LogicEngine — dual-mode deductive/probabilistic logic evaluation."""
from __future__ import annotations

from mowm.logic_engine import LogicEngine, LogicMode


class TestLogicMode:
    def test_enum_deductive(self):
        assert LogicMode.DEDUCTIVE.value == "deductive"

    def test_enum_probabilistic(self):
        assert LogicMode.PROBABILISTIC.value == "probabilistic"


class TestLogicEngineDeductive:
    def setup_method(self):
        self.engine = LogicEngine()

    def test_and_true_true(self):
        result = self.engine.evaluate("A AND B", {"A": True, "B": True}, mode="deductive")
        assert result is True

    def test_and_true_false(self):
        result = self.engine.evaluate("A AND B", {"A": True, "B": False}, mode="deductive")
        assert result is False

    def test_or_false_true(self):
        result = self.engine.evaluate("A OR B", {"A": False, "B": True}, mode="deductive")
        assert result is True

    def test_or_false_false(self):
        result = self.engine.evaluate("A OR B", {"A": False, "B": False}, mode="deductive")
        assert result is False

    def test_not_true(self):
        result = self.engine.evaluate("NOT A", {"A": True}, mode="deductive")
        assert result is False

    def test_not_false(self):
        result = self.engine.evaluate("NOT A", {"A": False}, mode="deductive")
        assert result is True

    def test_implies_true_true(self):
        result = self.engine.evaluate("A IMPLIES B", {"A": True, "B": True}, mode="deductive")
        assert result is True

    def test_implies_true_false(self):
        result = self.engine.evaluate("A IMPLIES B", {"A": True, "B": False}, mode="deductive")
        assert result is False

    def test_implies_false_false_vacuous(self):
        result = self.engine.evaluate("A IMPLIES B", {"A": False, "B": False}, mode="deductive")
        assert result is True


class TestLogicEngineProbabilistic:
    def setup_method(self):
        self.engine = LogicEngine()

    def test_and_returns_float(self):
        result = self.engine.evaluate("A AND B", {"A": 0.9, "B": 0.8}, mode="probabilistic")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_high_inputs_high_result(self):
        result = self.engine.evaluate("A AND B", {"A": 0.95, "B": 0.95}, mode="probabilistic")
        assert result > 0.5

    def test_low_inputs_low_result(self):
        result = self.engine.evaluate("A AND B", {"A": 0.05, "B": 0.05}, mode="probabilistic")
        assert result < 0.5

    def test_or_high_result(self):
        result = self.engine.evaluate("A OR B", {"A": 0.9, "B": 0.1}, mode="probabilistic")
        assert isinstance(result, float)
        assert result > 0.5


class TestLogicEngineAutoMode:
    def setup_method(self):
        self.engine = LogicEngine()

    def test_logic_domain_deductive(self):
        result = self.engine.evaluate_axiom(
            "A AND B", {"A": True, "B": True}, domain="logic.propositional"
        )
        assert result is True

    def test_legal_domain_deductive(self):
        result = self.engine.evaluate_axiom(
            "A AND B", {"A": True, "B": False}, domain="legal.contract"
        )
        assert result is False

    def test_economics_domain_probabilistic(self):
        result = self.engine.evaluate_axiom(
            "A OR B", {"A": 0.8, "B": 0.6}, domain="economics.market"
        )
        assert isinstance(result, float)

    def test_social_domain_probabilistic(self):
        result = self.engine.evaluate_axiom(
            "A AND B", {"A": 0.7, "B": 0.7}, domain="social.network"
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestCheckObligation:
    def setup_method(self):
        self.engine = LogicEngine()

    def test_all_conditions_true_no_exceptions(self):
        result = self.engine.check_obligation(
            conditions=["rain", "umbrella"],
            facts={"rain": True, "umbrella": True},
        )
        assert result is True

    def test_one_condition_false(self):
        result = self.engine.check_obligation(
            conditions=["rain", "umbrella"],
            facts={"rain": True, "umbrella": False},
        )
        assert result is False

    def test_exception_defeats_obligation(self):
        result = self.engine.check_obligation(
            conditions=["rain"],
            facts={"rain": True, "indoors": True},
            exceptions=["indoors"],
        )
        assert result is False

    def test_exception_not_present_obligation_holds(self):
        result = self.engine.check_obligation(
            conditions=["rain"],
            facts={"rain": True, "indoors": False},
            exceptions=["indoors"],
        )
        assert result is True

    def test_no_conditions(self):
        result = self.engine.check_obligation(
            conditions=[],
            facts={},
        )
        assert result is True


class TestConversionProbability:
    def setup_method(self):
        self.engine = LogicEngine()

    def test_high_positive_weights(self):
        result = self.engine.conversion_probability(
            features={"price_low": 1.0, "quality_high": 1.0},
            weights={"price_low": 3.0, "quality_high": 3.0},
        )
        assert result > 0.9

    def test_high_negative_weights(self):
        result = self.engine.conversion_probability(
            features={"price_high": 1.0, "quality_low": 1.0},
            weights={"price_high": -3.0, "quality_low": -3.0},
        )
        assert result < 0.1

    def test_zero_weights_near_half(self):
        result = self.engine.conversion_probability(
            features={"x": 1.0},
            weights={"x": 0.0},
        )
        assert abs(result - 0.5) < 0.01
