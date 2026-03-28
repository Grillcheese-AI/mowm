# mowm/executor.py
"""SafeFormulaExecutor — AST-based formula evaluation for axiom strings.

Parses formula strings like "F = m * a" into expression trees and evaluates
them with named variable bindings. Uses a recursive descent parser. NO eval()
or exec() — all evaluation goes through the AST.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from mowm.base import BaseExecutor
from mowm.discovery import _SAFE_OPS, _UNARY_OPS


@dataclass
class NamedExprNode:
    """Expression tree node supporting named variables."""

    op: str | None = None
    value: float | None = None
    var_name: str | None = None
    left: NamedExprNode | None = None
    right: NamedExprNode | None = None

    def evaluate(self, inputs: dict[str, float]) -> float:
        if self.op is None:
            if self.var_name is not None:
                return float(inputs.get(self.var_name, 0.0))
            return float(self.value if self.value is not None else 0.0)

        fn = _SAFE_OPS.get(self.op)
        if fn is None:
            if self.op == "^":
                left_val = self.left.evaluate(inputs) if self.left else 0.0
                right_val = self.right.evaluate(inputs) if self.right else 0.0
                try:
                    result = float(left_val ** right_val)
                except Exception:
                    result = 0.0
                return _clamp(result)
            return 0.0

        left_val = self.left.evaluate(inputs) if self.left else 0.0

        if self.op in _UNARY_OPS:
            try:
                arr = np.array([left_val], dtype=np.float64)
                result = float(fn(arr)[0])
            except Exception:
                result = 0.0
        else:
            right_val = self.right.evaluate(inputs) if self.right else 0.0
            try:
                arr_l = np.array([left_val], dtype=np.float64)
                arr_r = np.array([right_val], dtype=np.float64)
                result = float(fn(arr_l, arr_r)[0])
            except Exception:
                result = 0.0

        return _clamp(result)


def _clamp(val: float) -> float:
    if not np.isfinite(val):
        return 0.0
    return float(np.clip(val, -1e18, 1e18))


_TOKEN_RE = re.compile(
    r"(\d+\.?\d*(?:[eE][+-]?\d+)?)"
    r"|([a-zA-Z_]\w*)"
    r"|(\^)"
    r"|([+\-*/])"
    r"|([()])",
)

_FUNCTIONS = {"sin", "cos", "sqrt", "exp", "log", "square", "NOT"}


def _tokenize(formula: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(formula):
        num, ident, power, arith, paren = m.groups()
        if num is not None:
            tokens.append(("NUM", num))
        elif ident is not None:
            if ident in _FUNCTIONS:
                tokens.append(("FUNC", ident))
            else:
                tokens.append(("ID", ident))
        elif power is not None:
            tokens.append(("OP", "^"))
        elif arith is not None:
            tokens.append(("OP", arith))
        elif paren is not None:
            tokens.append(("LPAREN" if paren == "(" else "RPAREN", paren))
    return tokens


class _Parser:
    """Recursive descent parser. Precedence: ^ > * / > + -"""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, typ: str) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or tok[0] != typ:
            return (typ, "")
        return self.consume()

    def parse_expr(self) -> NamedExprNode:
        left = self.parse_term()
        while True:
            tok = self.peek()
            if tok and tok[0] == "OP" and tok[1] in ("+", "-"):
                op = self.consume()[1]
                right = self.parse_term()
                left = NamedExprNode(op=op, left=left, right=right)
            else:
                break
        return left

    def parse_term(self) -> NamedExprNode:
        left = self.parse_power()
        while True:
            tok = self.peek()
            if tok and tok[0] == "OP" and tok[1] in ("*", "/"):
                op = self.consume()[1]
                right = self.parse_power()
                left = NamedExprNode(op=op, left=left, right=right)
            else:
                break
        return left

    def parse_power(self) -> NamedExprNode:
        base = self.parse_unary()
        tok = self.peek()
        if tok and tok[0] == "OP" and tok[1] == "^":
            self.consume()
            exp = self.parse_power()
            return NamedExprNode(op="^", left=base, right=exp)
        return base

    def parse_unary(self) -> NamedExprNode:
        tok = self.peek()
        if tok and tok[0] == "OP" and tok[1] == "-":
            self.consume()
            operand = self.parse_unary()
            return NamedExprNode(
                op="*", left=NamedExprNode(value=-1.0), right=operand,
            )
        return self.parse_atom()

    def parse_atom(self) -> NamedExprNode:
        tok = self.peek()
        if tok is None:
            return NamedExprNode(value=0.0)

        if tok[0] == "NUM":
            self.consume()
            return NamedExprNode(value=float(tok[1]))

        if tok[0] == "FUNC":
            func_name = self.consume()[1]
            self.expect("LPAREN")
            arg = self.parse_expr()
            self.expect("RPAREN")
            return NamedExprNode(op=func_name, left=arg)

        if tok[0] == "ID":
            self.consume()
            return NamedExprNode(var_name=tok[1])

        if tok[0] == "LPAREN":
            self.consume()
            node = self.parse_expr()
            self.expect("RPAREN")
            return node

        self.consume()
        return NamedExprNode(value=0.0)


def _parse_formula(formula: str) -> NamedExprNode:
    formula = formula.strip()
    if not formula:
        raise ValueError("Empty formula string")

    if "=" in formula and "==" not in formula:
        parts = formula.split("=", 1)
        if len(parts) == 2:
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            if re.match(r"^[a-zA-Z_]\w*$", lhs) and rhs:
                formula = rhs

    tokens = _tokenize(formula)
    if not tokens:
        raise ValueError(f"Could not tokenize formula: {formula!r}")

    parser = _Parser(tokens)
    return parser.parse_expr()


class SafeFormulaExecutor(BaseExecutor):
    """Execute formulas via AST parsing (no eval/exec).

    Supports: +, -, *, /, ^, sin, cos, exp, sqrt, square, log.
    Handles equation form "F = m * a" (extracts RHS).
    Caches parsed trees for repeated evaluation.
    """

    def __init__(self) -> None:
        self._cache: dict[str, NamedExprNode] = {}

    def parse(self, formula_str: str) -> NamedExprNode:
        if formula_str in self._cache:
            return self._cache[formula_str]
        tree = _parse_formula(formula_str)
        self._cache[formula_str] = tree
        return tree

    def execute(
        self,
        formula_str: str,
        inputs: dict[str, float],
    ) -> float:
        """Evaluate a formula with named variables.

        First attempts SymPy symbolic solve (handles missing variables).
        Falls back to AST evaluation if SymPy is unavailable or fails.
        """
        # Try SymPy path first — can solve for missing variables
        try:
            result = self._execute_sympy(formula_str, inputs)
            if np.isfinite(result):
                return result
        except Exception:
            pass

        # Fallback: AST evaluation
        tree = self.parse(formula_str)
        return tree.evaluate(inputs)

    # ── SymPy-powered methods ────────────────────────────────────────────

    def _parse_sympy(self, formula_str: str):
        """Parse formula string into a SymPy expression.

        If formula contains '=', parses as LHS - RHS = 0.
        """
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        formula_str = formula_str.strip()
        if not formula_str:
            raise ValueError("Empty formula string")

        transforms = standard_transformations + (
            implicit_multiplication_application,
        )

        # Extract all identifiers and declare as real symbols
        # to prevent 'I' being treated as imaginary unit
        import re as _re
        idents = set(_re.findall(r"[A-Za-z_]\w*", formula_str))
        _funcs = {
            "sin", "cos", "tan", "sqrt", "exp", "log",
            "abs", "pi", "E", "oo",
        }
        local_dict = {
            name: sp.Symbol(name, real=True)
            for name in idents if name not in _funcs
        }

        if "=" in formula_str and "==" not in formula_str:
            lhs_str, rhs_str = formula_str.split("=", 1)
            lhs = parse_expr(
                lhs_str.strip(), local_dict=local_dict,
                transformations=transforms,
            )
            rhs = parse_expr(
                rhs_str.strip(), local_dict=local_dict,
                transformations=transforms,
            )
            return lhs - rhs

        return parse_expr(
            formula_str, local_dict=local_dict,
            transformations=transforms,
        )

    def _execute_sympy(
        self, formula_str: str, inputs: dict[str, float],
    ) -> float:
        """Evaluate using SymPy — can auto-solve for one missing variable.

        If all variables are provided, evaluates directly.
        If exactly one variable is missing, solves for it algebraically.
        """
        import sympy as sp

        expr = self._parse_sympy(formula_str)
        subs = {sp.Symbol(k, real=True): v for k, v in inputs.items()}

        if "=" in formula_str and "==" not in formula_str:
            free = expr.free_symbols
            provided = set(subs.keys())
            missing = free - provided

            if len(missing) == 1:
                target = list(missing)[0]
                solutions = sp.solve(expr.subs(subs), target)
                if not solutions:
                    raise ValueError(f"No solution for {target}")
                return float(solutions[0].evalf())
            elif len(missing) == 0:
                return float(expr.subs(subs).evalf())
            else:
                raise ValueError(
                    f"Multiple unknowns: {missing}"
                )
        else:
            result = expr.subs(subs).evalf()
            return float(result)

    def solve_for(self, formula_str: str, target_var: str) -> str:
        """Symbolically rearrange an equation to isolate a target variable.

        Args:
            formula_str: Equation string (e.g., "F = m * a").
            target_var: Variable to isolate (e.g., "a").

        Returns:
            Rearranged equation string (e.g., "a = F/m").

        Example:
            >>> ex.solve_for("F = m * a", "a")
            'a = F/m'
        """
        import sympy as sp

        expr = self._parse_sympy(formula_str)
        target = sp.Symbol(target_var, real=True)

        if target not in expr.free_symbols:
            raise ValueError(
                f"Variable '{target_var}' not in '{formula_str}'"
            )

        solutions = sp.solve(expr, target)
        if not solutions:
            raise ValueError(
                f"Could not isolate '{target_var}' in '{formula_str}'"
            )

        return f"{target_var} = {solutions[0]}"

    def symbolic_substitute(
        self, eq1: str, eq2: str, shared_var: str,
    ) -> str:
        """Substitute eq1 into eq2 to eliminate a shared variable.

        This is the core operation for proving structural cancellations
        (e.g., proving mass cancels in free fall: W=mg into a=F/m -> a=g).

        Args:
            eq1: First equation (e.g., "W = m * g").
            eq2: Second equation (e.g., "a = F / m").
            shared_var: Variable connecting them (e.g., "F").

        Returns:
            Simplified result string (e.g., "a = g").

        Example:
            >>> ex.symbolic_substitute("W = m * g", "a = F / m", "F")
            'a = g'
        """
        import sympy as sp

        expr1 = self._parse_sympy(eq1)
        expr2 = self._parse_sympy(eq2)
        shared = sp.Symbol(shared_var, real=True)

        # Isolate shared variable in eq1
        isolated = sp.solve(expr1, shared)
        if not isolated:
            raise ValueError(
                f"Cannot isolate '{shared_var}' in '{eq1}'"
            )

        # Substitute into eq2
        combined = expr2.subs(shared, isolated[0])
        simplified = sp.simplify(combined)

        # Format as "target = expression"
        free_syms = sorted(simplified.free_symbols, key=str)
        if free_syms:
            target = free_syms[0]
            final = sp.solve(simplified, target)
            if final:
                return f"{target} = {final[0]}"

        return str(simplified)
