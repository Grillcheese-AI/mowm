# tests/test_executor.py
"""Tests for SafeFormulaExecutor — AST-based formula evaluation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mowm.executor import SafeFormulaExecutor


class TestFormulaParser:
    def setup_method(self):
        self.ex = SafeFormulaExecutor()

    def test_parse_simple_multiply(self):
        tree = self.ex.parse("m * a")
        assert tree is not None
        assert tree.op == "*"

    def test_parse_equation_extracts_rhs(self):
        tree = self.ex.parse("F = m * a")
        assert tree is not None
        assert tree.op == "*"

    def test_parse_numeric_literal(self):
        tree = self.ex.parse("3.14")
        assert tree is not None
        assert tree.value == pytest.approx(3.14)

    def test_parse_addition(self):
        tree = self.ex.parse("x + y")
        assert tree.op == "+"

    def test_parse_subtraction(self):
        tree = self.ex.parse("x - y")
        assert tree.op == "-"

    def test_parse_division(self):
        tree = self.ex.parse("x / y")
        assert tree.op == "/"

    def test_parse_power(self):
        tree = self.ex.parse("v ^ 2")
        assert tree.op == "^"

    def test_parse_function_sin(self):
        tree = self.ex.parse("sin(theta)")
        assert tree.op == "sin"

    def test_parse_function_sqrt(self):
        tree = self.ex.parse("sqrt(x)")
        assert tree.op == "sqrt"

    def test_parse_function_cos(self):
        tree = self.ex.parse("cos(x)")
        assert tree.op == "cos"

    def test_parse_function_exp(self):
        tree = self.ex.parse("exp(x)")
        assert tree.op == "exp"

    def test_parse_function_log(self):
        tree = self.ex.parse("log(x)")
        assert tree.op == "log"

    def test_parse_nested_expression(self):
        tree = self.ex.parse("0.5 * m * v ^ 2")
        assert tree is not None

    def test_parse_parenthesized(self):
        tree = self.ex.parse("(x + y) * z")
        assert tree.op == "*"

    def test_parse_function_with_expression(self):
        tree = self.ex.parse("sqrt(2 * g * h)")
        assert tree.op == "sqrt"

    def test_parse_caches_result(self):
        tree1 = self.ex.parse("m * a")
        tree2 = self.ex.parse("m * a")
        assert tree1 is tree2

    def test_parse_negative_literal(self):
        tree = self.ex.parse("-9.8")
        assert tree is not None


class TestFormulaExecution:
    def setup_method(self):
        self.ex = SafeFormulaExecutor()

    def test_newtons_second_law(self):
        result = self.ex.execute("F = m * a", {"m": 10.0, "a": 9.8})
        assert result == pytest.approx(98.0)

    def test_kinetic_energy(self):
        result = self.ex.execute("0.5 * m * v ^ 2", {"m": 2.0, "v": 3.0})
        assert result == pytest.approx(9.0)

    def test_simple_addition(self):
        result = self.ex.execute("x + y", {"x": 3.0, "y": 4.0})
        assert result == pytest.approx(7.0)

    def test_subtraction(self):
        result = self.ex.execute("x - y", {"x": 10.0, "y": 3.0})
        assert result == pytest.approx(7.0)

    def test_division(self):
        result = self.ex.execute("x / y", {"x": 10.0, "y": 2.0})
        assert result == pytest.approx(5.0)

    def test_division_by_zero_safe(self):
        result = self.ex.execute("x / y", {"x": 10.0, "y": 0.0})
        assert np.isfinite(result)

    def test_sin_function(self):
        result = self.ex.execute("sin(theta)", {"theta": math.pi / 2})
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cos_function(self):
        result = self.ex.execute("cos(theta)", {"theta": 0.0})
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_sqrt_function(self):
        result = self.ex.execute("sqrt(x)", {"x": 16.0})
        assert result == pytest.approx(4.0)

    def test_exp_function(self):
        result = self.ex.execute("exp(x)", {"x": 0.0})
        assert result == pytest.approx(1.0)

    def test_log_function(self):
        result = self.ex.execute("log(x)", {"x": math.e})
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_complex_formula(self):
        result = self.ex.execute("sqrt(2 * g * h)", {"g": 9.8, "h": 5.0})
        assert result == pytest.approx(math.sqrt(98.0), abs=1e-4)

    def test_operator_precedence_mul_before_add(self):
        result = self.ex.execute("a + b * c", {"a": 1.0, "b": 2.0, "c": 3.0})
        assert result == pytest.approx(7.0)

    def test_parentheses_override_precedence(self):
        result = self.ex.execute("(a + b) * c", {"a": 1.0, "b": 2.0, "c": 3.0})
        assert result == pytest.approx(9.0)

    def test_missing_variable_returns_zero(self):
        result = self.ex.execute("x + y", {"x": 5.0})
        assert result == pytest.approx(5.0)

    def test_constant_only(self):
        result = self.ex.execute("42", {})
        assert result == pytest.approx(42.0)

    def test_equation_with_lhs(self):
        result = self.ex.execute("E = m * c ^ 2", {"m": 1.0, "c": 3e8})
        assert result == pytest.approx(9e16, rel=1e-6)

    def test_negative_numbers(self):
        result = self.ex.execute("F = -1 * k * x", {"k": 100.0, "x": 0.5})
        assert result == pytest.approx(-50.0)

    def test_chained_multiply(self):
        result = self.ex.execute("a * b * c", {"a": 2.0, "b": 3.0, "c": 4.0})
        assert result == pytest.approx(24.0)


class TestFormulaExecutorEdgeCases:
    def setup_method(self):
        self.ex = SafeFormulaExecutor()

    def test_empty_formula_raises(self):
        with pytest.raises(ValueError):
            self.ex.execute("", {"x": 1.0})

    def test_whitespace_handling(self):
        result = self.ex.execute("  m  *  a  ", {"m": 5.0, "a": 2.0})
        assert result == pytest.approx(10.0)

    def test_large_exponent_clamped(self):
        result = self.ex.execute("exp(x)", {"x": 1000.0})
        assert np.isfinite(result)

    def test_sqrt_of_negative_uses_abs(self):
        result = self.ex.execute("sqrt(x)", {"x": -4.0})
        assert result == pytest.approx(2.0)

    def test_log_of_zero_safe(self):
        result = self.ex.execute("log(x)", {"x": 0.0})
        assert np.isfinite(result)


# ── SymPy-powered methods ──────────────────────────────────────────────


class TestSympySolveFor:
    def setup_method(self):
        self.ex = SafeFormulaExecutor()

    def test_solve_for_acceleration(self):
        result = self.ex.solve_for("F = m * a", "a")
        assert "a = " in result
        assert "F" in result
        assert "m" in result

    def test_solve_for_mass(self):
        result = self.ex.solve_for("F = m * a", "m")
        assert "m = " in result

    def test_solve_for_velocity(self):
        result = self.ex.solve_for("KE = 0.5 * m * v**2", "v")
        assert "v = " in result

    def test_solve_for_missing_var_raises(self):
        with pytest.raises(ValueError, match="not in"):
            self.ex.solve_for("F = m * a", "z")


class TestSympyAutoSolve:
    """Test that execute() auto-solves for missing variables."""

    def setup_method(self):
        self.ex = SafeFormulaExecutor()

    def test_auto_solve_mass(self):
        # F=ma, given F=98 and a=9.8, solve for m -> 10
        result = self.ex.execute("F = m * a", {"F": 98.0, "a": 9.8})
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_auto_solve_force(self):
        # F=ma, given m=10 and a=9.8, solve for F -> 98
        result = self.ex.execute("F = m * a", {"m": 10.0, "a": 9.8})
        assert result == pytest.approx(98.0, rel=1e-4)

    def test_auto_solve_resistance(self):
        # V=IR, given V=50 and I=5, solve for R -> 10
        result = self.ex.execute("V = I * R", {"V": 50.0, "I": 5.0})
        assert result == pytest.approx(10.0, rel=1e-4)


class TestSympySubstitute:
    """Test symbolic_substitute for proving cancellations."""

    def setup_method(self):
        self.ex = SafeFormulaExecutor()

    def test_mass_cancels_in_free_fall(self):
        # W=mg substituted into a=F/m proves a=g
        result = self.ex.symbolic_substitute(
            "W = m * g", "a = W / m", "W",
        )
        assert "a = " in result
        assert "g" in result
        # m should NOT appear (it cancels)
        assert "m" not in result.split("= ")[1]

    def test_force_momentum_relationship(self):
        # F=ma substituted into p=m*v via m proves p = F*v/a
        result = self.ex.symbolic_substitute(
            "F = m * a", "p = m * v", "m",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_shared_var_raises(self):
        with pytest.raises(ValueError):
            self.ex.symbolic_substitute(
                "F = m * a", "V = I * R", "z",
            )
