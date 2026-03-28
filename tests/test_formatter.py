# tests/test_formatter.py
"""Tests for ResponseFormatter — pure formatting helpers."""
from __future__ import annotations

import pytest

from mowm.qa.formatter import ResponseFormatter


@pytest.fixture
def fmt() -> ResponseFormatter:
    return ResponseFormatter()


# ---------------------------------------------------------------------------
# TestFormatCompute
# ---------------------------------------------------------------------------


class TestFormatCompute:
    def test_includes_axiom_name(self, fmt):
        out = fmt.format_compute("Newton's Second Law", "F = m * a", {"m": 10, "a": 9.8}, 98.0, "N")
        assert "Newton's Second Law" in out

    def test_includes_formula(self, fmt):
        out = fmt.format_compute("Newton's Second Law", "F = m * a", {"m": 10, "a": 9.8}, 98.0, "N")
        assert "F = m * a" in out

    def test_includes_variable_values(self, fmt):
        out = fmt.format_compute("Newton's Second Law", "F = m * a", {"m": 10, "a": 9.8}, 98.0, "N")
        assert "m" in out and "10" in out
        assert "a" in out and "9.8" in out

    def test_includes_result_with_unit(self, fmt):
        out = fmt.format_compute("Newton's Second Law", "F = m * a", {"m": 10, "a": 9.8}, 98.0, "N")
        assert "98.0" in out
        assert "N" in out

    def test_no_unit(self, fmt):
        out = fmt.format_compute("identity", "y = x", {"x": 5}, 5)
        assert "5" in out
        # Should not crash and should not append stray space
        assert "Result: 5" in out

    def test_empty_variables(self, fmt):
        out = fmt.format_compute("constant", "pi = 3.14", {}, 3.14)
        assert "3.14" in out


# ---------------------------------------------------------------------------
# TestFormatEvaluate
# ---------------------------------------------------------------------------


class TestFormatEvaluate:
    def test_holds_when_true(self, fmt):
        out = fmt.format_evaluate(["signed_contract", "paid_deposit"], True)
        assert "holds" in out
        assert "does not hold" not in out

    def test_does_not_hold_when_false(self, fmt):
        out = fmt.format_evaluate(["signed_contract", "paid_deposit"], False)
        assert "does not hold" in out

    def test_lists_conditions(self, fmt):
        out = fmt.format_evaluate(["cond_a", "cond_b"], True)
        assert "cond_a" in out
        assert "cond_b" in out

    def test_shows_exceptions(self, fmt):
        out = fmt.format_evaluate(["cond_a"], False, exceptions=["cond_a raised ValueError"])
        assert "cond_a raised ValueError" in out

    def test_no_exceptions_key_absent(self, fmt):
        out = fmt.format_evaluate(["cond_a"], True)
        assert "Exceptions" not in out

    def test_empty_conditions(self, fmt):
        out = fmt.format_evaluate([], True)
        assert "holds" in out


# ---------------------------------------------------------------------------
# TestFormatCausal
# ---------------------------------------------------------------------------


class TestFormatCausal:
    def test_increases_positive_delta(self, fmt):
        out = fmt.format_causal("force", 10.0, ["acceleration increases"])
        assert "increases" in out
        assert "10.0" in out

    def test_decreases_negative_delta(self, fmt):
        out = fmt.format_causal("temperature", -5.0, ["pressure decreases"])
        assert "decreases" in out
        assert "5.0" in out

    def test_shows_node(self, fmt):
        out = fmt.format_causal("velocity", 3.0, [])
        assert "velocity" in out

    def test_lists_effects(self, fmt):
        out = fmt.format_causal("mass", 2.0, ["gravity increases", "momentum increases"])
        assert "gravity increases" in out
        assert "momentum increases" in out

    def test_no_downstream_effects_when_empty(self, fmt):
        out = fmt.format_causal("energy", 1.0, [])
        assert "no downstream effects" in out.lower()

    def test_no_downstream_effects_when_none(self, fmt):
        out = fmt.format_causal("energy", 1.0, None)
        assert "no downstream effects" in out.lower()


# ---------------------------------------------------------------------------
# TestFormatExplain
# ---------------------------------------------------------------------------


class TestFormatExplain:
    def test_includes_axiom_name(self, fmt):
        out = fmt.format_explain("Ohm's Law", "V = I * R", "electronics")
        assert "Ohm's Law" in out

    def test_includes_formula(self, fmt):
        out = fmt.format_explain("Ohm's Law", "V = I * R", "electronics")
        assert "V = I * R" in out

    def test_includes_domain(self, fmt):
        out = fmt.format_explain("Ohm's Law", "V = I * R", "electronics")
        assert "electronics" in out

    def test_includes_context_when_provided(self, fmt):
        out = fmt.format_explain(
            "Ohm's Law", "V = I * R", "electronics", context="Relates voltage to current."
        )
        assert "Relates voltage to current." in out

    def test_no_context_key_absent(self, fmt):
        out = fmt.format_explain("Ohm's Law", "V = I * R", "electronics")
        assert "Context" not in out


# ---------------------------------------------------------------------------
# TestFormatCompare
# ---------------------------------------------------------------------------


class TestFormatCompare:
    def test_lists_item_name(self, fmt):
        out = fmt.format_compare([{"name": "car_A", "value": 120, "unit": "km/h"}])
        assert "car_A" in out

    def test_lists_item_value(self, fmt):
        out = fmt.format_compare([{"name": "car_A", "value": 120, "unit": "km/h"}])
        assert "120" in out

    def test_lists_item_unit(self, fmt):
        out = fmt.format_compare([{"name": "car_A", "value": 120, "unit": "km/h"}])
        assert "km/h" in out

    def test_multiple_items(self, fmt):
        items = [
            {"name": "car_A", "value": 120, "unit": "km/h"},
            {"name": "car_B", "value": 95, "unit": "km/h"},
        ]
        out = fmt.format_compare(items)
        assert "car_A" in out
        assert "car_B" in out
        assert "95" in out

    def test_empty_returns_no_items_message(self, fmt):
        out = fmt.format_compare([])
        assert "No items to compare" in out

    def test_item_without_unit(self, fmt):
        out = fmt.format_compare([{"name": "alpha", "value": 0.5}])
        assert "alpha" in out
        assert "0.5" in out
