# mowm/qa/parser.py
"""QuestionParser — regex-based intent and entity extraction for Q&A pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    """Question intent categories (priority: CAUSAL > EVALUATE > EXPLAIN > COMPARE > COMPUTE > GENERAL)."""

    COMPUTE = "COMPUTE"
    EVALUATE = "EVALUATE"
    CAUSAL = "CAUSAL"
    EXPLAIN = "EXPLAIN"
    COMPARE = "COMPARE"
    GENERAL = "GENERAL"


@dataclass
class ParsedQuestion:
    """Structured output from QuestionParser.parse()."""

    raw: str
    intent: Intent = Intent.GENERAL
    variables: dict[str, float] = field(default_factory=dict)
    target: str | None = None
    causal_node: str | None = None
    causal_delta: float | None = None
    conditions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stop-words filtered from keywords
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "does",
        "do", "for", "from", "has", "have", "how", "i", "if", "in", "is",
        "it", "its", "me", "much", "my", "no", "not", "of", "on", "or",
        "so", "that", "the", "then", "there", "this", "to", "was", "we",
        "what", "when", "where", "which", "who", "will", "with", "would",
        "you",
    }
)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Variable assignments: "m=10", "a = 9.8", "x=-5.0", "m=1e3"
_VAR_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*=\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)

# Target variable: "What is F" / "what is the value of F"
_TARGET_RE = re.compile(
    r"(?:what\s+is(?:\s+the\s+\w+\s+of)?)\s+([A-Za-z_]\w*)",
    re.IGNORECASE,
)

# Causal node + direction + magnitude
_CAUSAL_RE = re.compile(
    r"\b([A-Za-z_][\w\s]*?)\s+(increases?|decreases?)\s+by\s+"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Conditions after "if": "if cond_a and cond_b"
_CONDITIONS_RE = re.compile(r"\bif\s+(.+)", re.IGNORECASE)

# Intent keyword sets (checked in priority order)
_CAUSAL_KEYWORDS = re.compile(
    r"\b(causes?|leads?\s+to|results?\s+in|increases?\s+by|decreases?\s+by|effect\s+of|impact\s+of)\b",
    re.IGNORECASE,
)
_EVALUATE_KEYWORDS = re.compile(
    r"\b(holds?|true|false|valid|satisf(?:ies|y)|evaluate|check(?:s)?|is\s+it\s+true)\b",
    re.IGNORECASE,
)
_EXPLAIN_KEYWORDS = re.compile(
    r"\b(explain|why|describe|definition\s+of|what\s+does\s+\w+\s+mean|meaning\s+of)\b",
    re.IGNORECASE,
)
_COMPARE_KEYWORDS = re.compile(
    r"\b(compar(?:e|ing|ison)|vs\.?|versus|differ(?:ent|ence)|which\s+is\s+(?:greater|better|faster|more|less))\b",
    re.IGNORECASE,
)
_COMPUTE_KEYWORDS = re.compile(
    r"\b(calculat(?:e|ing)|comput(?:e|ing)|what\s+is|find|determin(?:e|ing)|how\s+much|how\s+many|solv(?:e|ing))\b",
    re.IGNORECASE,
)

# Token splitter for keywords
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")


class QuestionParser:
    """Parse a natural-language question into a structured ParsedQuestion."""

    def parse(self, question: str) -> ParsedQuestion:
        """Extract intent, variables, target, causal info, conditions, and keywords.

        Args:
            question: Raw natural-language question string.

        Returns:
            ParsedQuestion dataclass with all extracted fields.
        """
        pq = ParsedQuestion(raw=question)
        pq.intent = self._detect_intent(question)
        pq.variables = self._extract_variables(question)
        pq.target = self._extract_target(question)
        pq.causal_node, pq.causal_delta = self._extract_causal(question)
        pq.conditions = self._extract_conditions(question)
        pq.keywords = self._extract_keywords(question)
        return pq

    # ------------------------------------------------------------------
    # Intent detection (priority: CAUSAL > EVALUATE > EXPLAIN > COMPARE > COMPUTE > GENERAL)
    # ------------------------------------------------------------------

    def _detect_intent(self, text: str) -> Intent:
        if _CAUSAL_KEYWORDS.search(text):
            return Intent.CAUSAL
        if _EVALUATE_KEYWORDS.search(text):
            return Intent.EVALUATE
        if _EXPLAIN_KEYWORDS.search(text):
            return Intent.EXPLAIN
        if _COMPARE_KEYWORDS.search(text):
            return Intent.COMPARE
        if _COMPUTE_KEYWORDS.search(text):
            return Intent.COMPUTE
        return Intent.GENERAL

    # ------------------------------------------------------------------
    # Variable extraction
    # ------------------------------------------------------------------

    def _extract_variables(self, text: str) -> dict[str, float]:
        variables: dict[str, float] = {}
        for name, value_str in _VAR_RE.findall(text):
            try:
                variables[name] = float(value_str)
            except ValueError:
                pass
        return variables

    # ------------------------------------------------------------------
    # Target variable
    # ------------------------------------------------------------------

    def _extract_target(self, text: str) -> str | None:
        m = _TARGET_RE.search(text)
        if m:
            return m.group(1)
        return None

    # ------------------------------------------------------------------
    # Causal node and delta
    # ------------------------------------------------------------------

    def _extract_causal(self, text: str) -> tuple[str | None, float | None]:
        m = _CAUSAL_RE.search(text)
        if not m:
            return None, None
        node = m.group(1).strip()
        direction = m.group(2).lower()
        magnitude = float(m.group(3))
        delta = magnitude if direction.startswith("increase") else -magnitude
        return node, delta

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def _extract_conditions(self, text: str) -> list[str]:
        m = _CONDITIONS_RE.search(text)
        if not m:
            return []
        tail = m.group(1)
        # Split on "and" boundaries (word boundary, case-insensitive)
        parts = re.split(r"\s+and\s+", tail, flags=re.IGNORECASE)
        # Strip punctuation (?, !, .) from condition names
        return [
            p.strip().rstrip("?!.,;:") for p in parts
            if p.strip().rstrip("?!.,;:")
        ]

    # ------------------------------------------------------------------
    # Keywords
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text)
        seen: set[str] = set()
        keywords: list[str] = []
        for tok in tokens:
            lower = tok.lower()
            if lower not in _STOPWORDS and lower not in seen:
                seen.add(lower)
                keywords.append(tok)
        return keywords
