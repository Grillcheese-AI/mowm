# mowm/logic_engine.py
"""Dual-mode logic engine: deductive (boolean) and probabilistic (sigmoid) evaluation.

Implements:
  - LogicMode enum (DEDUCTIVE, PROBABILISTIC)
  - Recursive-descent parser for AND/OR/NOT/IMPLIES expressions
  - Auto-mode detection from domain prefix
  - Deontic obligation checking
  - Conversion probability via sigmoid of weighted features
"""
from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any

from mowm.base import BaseLogicEngine


class LogicMode(Enum):
    DEDUCTIVE = "deductive"
    PROBABILISTIC = "probabilistic"


# ── Domain → mode mapping ────────────────────────────────────────────────────

_DEDUCTIVE_PREFIXES = {"logic", "legal", "ethics"}
_PROBABILISTIC_PREFIXES = {"economics", "social", "marketing", "identity"}


def _auto_mode(domain: str) -> LogicMode:
    """Detect LogicMode from the first segment of a dotted domain string."""
    prefix = domain.split(".")[0].lower()
    if prefix in _DEDUCTIVE_PREFIXES:
        return LogicMode.DEDUCTIVE
    if prefix in _PROBABILISTIC_PREFIXES:
        return LogicMode.PROBABILISTIC
    # Default to deductive for unknown domains
    return LogicMode.DEDUCTIVE


# ── Numerics ─────────────────────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


# ── Tokenizer ────────────────────────────────────────────────────────────────

_TOKEN_PATTERN = re.compile(
    r"\bIMPLIES\b|\bAND\b|\bOR\b|\bNOT\b|[()]|[A-Za-z_][A-Za-z0-9_]*"
)


def _tokenize_logic(formula: str) -> list[str]:
    """Tokenize a logic formula into keywords, identifiers, and parentheses."""
    return _TOKEN_PATTERN.findall(formula)


# ── Recursive-descent parser ─────────────────────────────────────────────────


class _LogicParser:
    """Parse a token stream into a Python callable (deductive or probabilistic).

    Grammar (lowest to highest precedence):
        impl  := or ( IMPLIES or )*
        or    := and ( OR and )*
        and   := not ( AND not )*
        not   := NOT not | atom
        atom  := IDENT | '(' impl ')'
    """

    def __init__(self, tokens: list[str], facts: dict[str, Any], mode: LogicMode) -> None:
        self._tokens = tokens
        self._pos = 0
        self._facts = facts
        self._mode = mode

    # -- helpers ---------------------------------------------------------------

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self, expected: str | None = None) -> str:
        tok = self._tokens[self._pos]
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self._pos += 1
        return tok

    # -- grammar productions ---------------------------------------------------

    def parse(self) -> bool | float:
        result = self._parse_impl()
        if self._pos != len(self._tokens):
            raise ValueError(f"Unexpected token: {self._peek()!r}")
        return result

    def _parse_impl(self) -> bool | float:
        left = self._parse_or()
        while self._peek() == "IMPLIES":
            self._consume("IMPLIES")
            right = self._parse_or()
            if self._mode == LogicMode.DEDUCTIVE:
                left = (not left) or right  # type: ignore[arg-type]
            else:
                # probabilistic: P(A->B) ≈ sigmoid(-logit(A) + logit(B))
                # Simple: 1 - A*(1-B)
                left = float(1.0 - float(left) * (1.0 - float(right)))
        return left

    def _parse_or(self) -> bool | float:
        left = self._parse_and()
        while self._peek() == "OR":
            self._consume("OR")
            right = self._parse_and()
            if self._mode == LogicMode.DEDUCTIVE:
                left = left or right
            else:
                left = _sigmoid(math.log(float(left) / (1.0 - float(left) + 1e-9) +
                                         float(right) / (1.0 - float(right) + 1e-9)))
        return left

    def _parse_and(self) -> bool | float:
        left = self._parse_not()
        while self._peek() == "AND":
            self._consume("AND")
            right = self._parse_not()
            if self._mode == LogicMode.DEDUCTIVE:
                left = left and right
            else:
                left = _sigmoid(math.log(float(left) / (1.0 - float(left) + 1e-9) +
                                         float(right) / (1.0 - float(right) + 1e-9)))
        return left

    def _parse_not(self) -> bool | float:
        if self._peek() == "NOT":
            self._consume("NOT")
            operand = self._parse_not()
            if self._mode == LogicMode.DEDUCTIVE:
                return not operand
            else:
                return 1.0 - float(operand)
        return self._parse_atom()

    def _parse_atom(self) -> bool | float:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of formula")
        if tok == "(":
            self._consume("(")
            result = self._parse_impl()
            self._consume(")")
            return result
        if tok in ("AND", "OR", "NOT", "IMPLIES", ")"):
            raise ValueError(f"Unexpected keyword/token in atom position: {tok!r}")
        # It's an identifier
        self._consume()
        if tok not in self._facts:
            raise KeyError(f"Unknown variable in formula: {tok!r}")
        val = self._facts[tok]
        if self._mode == LogicMode.PROBABILISTIC:
            return float(val)
        return bool(val)


# ── LogicEngine ───────────────────────────────────────────────────────────────


class LogicEngine(BaseLogicEngine):
    """Dual-mode logic engine implementing BaseLogicEngine.

    Deductive mode: classical boolean evaluation (AND/OR/NOT/IMPLIES).
    Probabilistic mode: soft sigmoid-based evaluation over [0, 1].
    """

    # -- BaseLogicEngine interface ---------------------------------------------

    def evaluate(
        self,
        formula_str: str,
        facts: dict[str, bool | float],
        mode: str | None = None,
    ) -> bool | float:
        """Evaluate a logic formula.

        Args:
            formula_str: Expression using AND, OR, NOT, IMPLIES and variable names.
            facts: Variable bindings (bool for deductive, float in [0,1] for probabilistic).
            mode: "deductive", "probabilistic", or None (defaults to "deductive").

        Returns:
            bool in deductive mode, float in [0,1] in probabilistic mode.
        """
        if mode is None or mode == "deductive":
            logic_mode = LogicMode.DEDUCTIVE
        elif mode == "probabilistic":
            logic_mode = LogicMode.PROBABILISTIC
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'deductive' or 'probabilistic'.")

        tokens = _tokenize_logic(formula_str)
        parser = _LogicParser(tokens, facts, logic_mode)
        return parser.parse()

    def check_obligation(
        self,
        conditions: list[str],
        facts: dict[str, bool],
        exceptions: list[str] | None = None,
    ) -> bool:
        """Deontic obligation check: obligation holds iff all conditions are True
        and no exception is True.

        Args:
            conditions: Variable names that must all be True.
            facts: Boolean fact bindings.
            exceptions: Variable names that defeat the obligation if True.

        Returns:
            True if obligation holds, False otherwise.
        """
        # All conditions must hold
        for cond in conditions:
            if not facts.get(cond, False):
                return False
        # Any active exception defeats the obligation
        if exceptions:
            for exc in exceptions:
                if facts.get(exc, False):
                    return False
        return True

    # -- Extended interface ----------------------------------------------------

    def evaluate_axiom(
        self,
        formula_str: str,
        facts: dict[str, bool | float],
        domain: str = "logic",
    ) -> bool | float:
        """Evaluate a formula with mode auto-detected from domain prefix.

        Args:
            formula_str: Logic expression.
            facts: Variable bindings.
            domain: Dotted domain string, e.g. "logic.propositional" or "economics.market".

        Returns:
            bool (deductive domains) or float (probabilistic domains).
        """
        logic_mode = _auto_mode(domain)
        tokens = _tokenize_logic(formula_str)
        parser = _LogicParser(tokens, facts, logic_mode)
        return parser.parse()

    def conversion_probability(
        self,
        features: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Compute conversion probability as sigmoid of a weighted feature sum.

        Args:
            features: Feature name → value mapping.
            weights: Feature name → weight mapping.

        Returns:
            Probability in [0, 1].
        """
        score = sum(weights.get(name, 0.0) * val for name, val in features.items())
        return _sigmoid(score)
