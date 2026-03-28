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
        tree = self.parse(formula_str)
        return tree.evaluate(inputs)
