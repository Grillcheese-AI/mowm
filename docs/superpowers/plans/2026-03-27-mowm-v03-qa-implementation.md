# MoWM v0.3.0 Q&A Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Q&A pipeline that answers questions using MoWM's world model reasoning -- direct computation for formulaic questions, LLM-augmented for open-ended ones.

**Architecture:** Question -> classify domain -> parse intent/variables -> try direct solve (executor/logic/causal) -> fall back to augmented solve (LLM with MoWM context) -> format response with evidence trail.

**Tech Stack:** Python 3.12, numpy, cubemind, httpx (for LLM API calls)

---

## Pre-flight Checklist

Before starting any task, confirm:

```bash
cd c:\Users\grill\Documents\GitHub\mowm
uv run pytest tests/ -v -q --tb=short -x
# Must show 381 tests passing — this is the v0.2.0 baseline
```

All new code goes into `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\` directory.
All new tests go into `c:\Users\grill\Documents\GitHub\mowm\tests\` directory.
Constants: `K = 4`, `L = 32` in all tests.
Line length: 100 characters (ruff enforced).
Max 1000 lines per file.
NO external LLM dependency for core functionality (template fallback always works).
LLM calls use httpx (already in dev deps) for OpenRouter/local endpoints.

---

## Task 1: qa/formatter.py -- ResponseFormatter

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\formatter.py`

No dependencies on other QA modules. Pure formatting logic.

### Step 1.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_formatter.py`

```python
# tests/test_formatter.py
"""Tests for ResponseFormatter — structured answer to natural text."""
from __future__ import annotations

import pytest

from mowm.qa.formatter import ResponseFormatter


class TestFormatCompute:
    def setup_method(self):
        self.fmt = ResponseFormatter()

    def test_basic_compute(self):
        result = self.fmt.format_compute(
            axiom_name="Newton's Second Law",
            formula="F = m * a",
            variables={"m": 10.0, "a": 9.8},
            result=98.0,
            unit="N",
        )
        assert "Newton's Second Law" in result
        assert "F = m * a" in result
        assert "98.0" in result
        assert "N" in result

    def test_compute_without_unit(self):
        result = self.fmt.format_compute(
            axiom_name="Kinetic Energy",
            formula="KE = 0.5 * m * v^2",
            variables={"m": 2.0, "v": 3.0},
            result=9.0,
        )
        assert "Kinetic Energy" in result
        assert "9.0" in result
        assert "KE = 0.5 * m * v^2" in result

    def test_compute_shows_variable_values(self):
        result = self.fmt.format_compute(
            axiom_name="Ohm's Law",
            formula="V = I * R",
            variables={"I": 2.0, "R": 5.0},
            result=10.0,
            unit="V",
        )
        assert "I = 2.0" in result
        assert "R = 5.0" in result

    def test_compute_empty_variables(self):
        result = self.fmt.format_compute(
            axiom_name="Constant",
            formula="c = 299792458",
            variables={},
            result=299792458.0,
        )
        assert "299792458.0" in result


class TestFormatEvaluate:
    def setup_method(self):
        self.fmt = ResponseFormatter()

    def test_obligation_holds(self):
        result = self.fmt.format_evaluate(
            conditions=["signed_contract", "paid_deposit"],
            result=True,
        )
        assert "holds" in result.lower() or "yes" in result.lower()
        assert "signed_contract" in result
        assert "paid_deposit" in result

    def test_obligation_fails(self):
        result = self.fmt.format_evaluate(
            conditions=["signed_contract", "paid_deposit"],
            result=False,
        )
        assert "not" in result.lower() or "no" in result.lower()

    def test_obligation_with_exceptions(self):
        result = self.fmt.format_evaluate(
            conditions=["signed_contract"],
            result=False,
            exceptions=["force_majeure"],
        )
        assert "force_majeure" in result

    def test_obligation_empty_conditions(self):
        result = self.fmt.format_evaluate(
            conditions=[],
            result=True,
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestFormatCausal:
    def setup_method(self):
        self.fmt = ResponseFormatter()

    def test_single_effect(self):
        result = self.fmt.format_causal(
            node="force",
            delta=10.0,
            effects={"acceleration": 9.0},
        )
        assert "force" in result
        assert "10.0" in result
        assert "acceleration" in result
        assert "9.0" in result

    def test_multiple_effects(self):
        result = self.fmt.format_causal(
            node="ad_spend",
            delta=1000.0,
            effects={"traffic": 800.0, "sales": 480.0, "stock": -288.0},
        )
        assert "ad_spend" in result
        assert "traffic" in result
        assert "sales" in result
        assert "stock" in result
        assert "increases" in result.lower() or "+" in result
        assert "decreases" in result.lower() or "-" in result

    def test_negative_delta(self):
        result = self.fmt.format_causal(
            node="price",
            delta=-5.0,
            effects={"demand": 10.0},
        )
        assert "price" in result
        assert "demand" in result

    def test_no_effects(self):
        result = self.fmt.format_causal(
            node="isolated_node",
            delta=10.0,
            effects={},
        )
        assert "isolated_node" in result
        assert "no downstream" in result.lower() or "no effect" in result.lower()


class TestFormatExplain:
    def setup_method(self):
        self.fmt = ResponseFormatter()

    def test_explain_with_context(self):
        result = self.fmt.format_explain(
            axiom_name="Newton's Second Law",
            formula="F = m * a",
            domain="physics.dynamics",
            context="Force equals mass times acceleration.",
        )
        assert "Newton's Second Law" in result
        assert "F = m * a" in result
        assert "Force equals mass times acceleration" in result

    def test_explain_without_context(self):
        result = self.fmt.format_explain(
            axiom_name="Ohm's Law",
            formula="V = I * R",
            domain="physics.em",
            context="",
        )
        assert "Ohm's Law" in result
        assert "V = I * R" in result

    def test_explain_includes_domain(self):
        result = self.fmt.format_explain(
            axiom_name="Modus Ponens",
            formula="(P AND (P IMPLIES Q)) IMPLIES Q",
            domain="logic.deductive",
            context="A fundamental rule of inference.",
        )
        assert "logic" in result.lower()


class TestFormatCompare:
    def setup_method(self):
        self.fmt = ResponseFormatter()

    def test_compare_two_results(self):
        result = self.fmt.format_compare(
            items=[
                {"name": "Object A", "value": 98.0, "unit": "N"},
                {"name": "Object B", "value": 49.0, "unit": "N"},
            ],
        )
        assert "Object A" in result
        assert "Object B" in result
        assert "98.0" in result
        assert "49.0" in result

    def test_compare_empty(self):
        result = self.fmt.format_compare(items=[])
        assert isinstance(result, str)
```

### Step 1.2 -- Implement formatter.py

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\formatter.py`

```python
# mowm/qa/formatter.py
"""ResponseFormatter — convert structured QA results into natural language.

Pure formatting module with no dependencies on other QA components.
Each method takes structured data and returns a human-readable string.
"""
from __future__ import annotations


class ResponseFormatter:
    """Format structured QA results as natural language responses.

    Provides specialized formatters for each question intent type:
    compute, evaluate, causal, explain, and compare.
    """

    def format_compute(
        self,
        axiom_name: str,
        formula: str,
        variables: dict[str, float],
        result: float,
        unit: str = "",
    ) -> str:
        """Format a computation result.

        Example output:
            Using Newton's Second Law (F = m * a):
            With m = 10.0, a = 9.8:
            F = 98.0 N
        """
        lines = [f"Using {axiom_name} ({formula}):"]

        if variables:
            var_parts = [f"{k} = {v}" for k, v in variables.items()]
            lines.append(f"With {', '.join(var_parts)}:")

        # Extract target variable from formula LHS if possible
        target = self._extract_target(formula)
        result_str = f"{result}"
        if unit:
            result_str = f"{result} {unit}"
        if target:
            lines.append(f"{target} = {result_str}")
        else:
            lines.append(f"Result = {result_str}")

        return "\n".join(lines)

    def format_evaluate(
        self,
        conditions: list[str],
        result: bool,
        exceptions: list[str] | None = None,
    ) -> str:
        """Format a logic evaluation result.

        Example output (holds):
            Yes, the obligation holds.
            Conditions met: signed_contract, paid_deposit.

        Example output (fails):
            No, the obligation does not hold.
            Conditions: signed_contract, paid_deposit.
            No exceptions apply.
        """
        if result:
            lines = ["Yes, the obligation holds."]
            if conditions:
                cond_str = ", ".join(conditions)
                lines.append(f"Conditions met: {cond_str}.")
            else:
                lines.append("No conditions required.")
        else:
            lines = ["No, the obligation does not hold."]
            if conditions:
                cond_str = ", ".join(conditions)
                lines.append(f"Conditions: {cond_str}.")

        if exceptions:
            exc_str = ", ".join(exceptions)
            lines.append(f"Exceptions considered: {exc_str}.")
        elif not result and conditions:
            lines.append("No exceptions apply.")

        return "\n".join(lines)

    def format_causal(
        self,
        node: str,
        delta: float,
        effects: dict[str, float],
    ) -> str:
        """Format causal propagation results.

        Example output:
            If force increases by 10.0:
            -> acceleration increases by 9.0 (+)
            -> velocity increases by 7.2 (+)
        """
        direction = "increases" if delta >= 0 else "decreases"
        lines = [f"If {node} {direction} by {abs(delta)}:"]

        if not effects:
            lines.append("No downstream effects detected.")
            return "\n".join(lines)

        for target, effect_val in effects.items():
            if effect_val >= 0:
                lines.append(
                    f"-> {target} increases by {abs(effect_val):.4g} (+)"
                )
            else:
                lines.append(
                    f"-> {target} decreases by {abs(effect_val):.4g} (-)"
                )

        return "\n".join(lines)

    def format_explain(
        self,
        axiom_name: str,
        formula: str,
        domain: str,
        context: str,
    ) -> str:
        """Format an explanation response.

        Example output:
            Newton's Second Law (physics.dynamics):
            Formula: F = m * a
            Force equals mass times acceleration.
        """
        top_domain = domain.split(".")[0]
        lines = [f"{axiom_name} ({domain}):"]
        lines.append(f"Formula: {formula}")
        lines.append(f"Domain: {top_domain}")

        if context:
            lines.append("")
            lines.append(context)

        return "\n".join(lines)

    def format_compare(
        self,
        items: list[dict],
    ) -> str:
        """Format a comparison of multiple computed results.

        Each item dict should have: name, value, and optionally unit.

        Example output:
            Comparison:
            - Object A: 98.0 N
            - Object B: 49.0 N
        """
        if not items:
            return "No items to compare."

        lines = ["Comparison:"]
        for item in items:
            name = item.get("name", "Unknown")
            value = item.get("value", 0.0)
            unit = item.get("unit", "")
            val_str = f"{value}"
            if unit:
                val_str = f"{value} {unit}"
            lines.append(f"- {name}: {val_str}")

        return "\n".join(lines)

    @staticmethod
    def _extract_target(formula: str) -> str | None:
        """Extract the LHS target variable from a formula like 'F = m * a'."""
        if "=" not in formula or "==" in formula:
            return None
        lhs = formula.split("=", 1)[0].strip()
        # Only return if LHS is a simple identifier
        if lhs.isidentifier():
            return lhs
        return None
```

### Step 1.3 -- Run tests

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run pytest tests/test_formatter.py -v --tb=short`
- [ ] Verify all tests pass

---

## Task 2: qa/parser.py -- QuestionParser

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\parser.py`

Depends on: nothing (standalone regex-based parser).

### Step 2.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_parser.py`

```python
# tests/test_parser.py
"""Tests for QuestionParser — regex-based question parsing."""
from __future__ import annotations

import pytest

from mowm.qa.parser import ParsedQuestion, QuestionIntent, QuestionParser


class TestQuestionIntentDetection:
    def setup_method(self):
        self.parser = QuestionParser()

    def test_compute_what_is(self):
        parsed = self.parser.parse("What is F when m=10 and a=9.8?")
        assert parsed.intent == QuestionIntent.COMPUTE

    def test_compute_calculate(self):
        parsed = self.parser.parse("Calculate the force when m=5, a=2")
        assert parsed.intent == QuestionIntent.COMPUTE

    def test_compute_find(self):
        parsed = self.parser.parse("Find the velocity when d=100 and t=10")
        assert parsed.intent == QuestionIntent.COMPUTE

    def test_compute_how_much(self):
        parsed = self.parser.parse("How much energy when m=2 and v=3?")
        assert parsed.intent == QuestionIntent.COMPUTE

    def test_evaluate_is_if(self):
        parsed = self.parser.parse(
            "Is the obligation valid if signed_contract and paid_deposit?"
        )
        assert parsed.intent == QuestionIntent.EVALUATE

    def test_evaluate_does(self):
        parsed = self.parser.parse(
            "Does the contract hold when signed_contract is true?"
        )
        assert parsed.intent == QuestionIntent.EVALUATE

    def test_causal_what_happens(self):
        parsed = self.parser.parse(
            "What happens if force increases by 10?"
        )
        assert parsed.intent == QuestionIntent.CAUSAL

    def test_causal_effect_of(self):
        parsed = self.parser.parse(
            "What is the effect of increasing ad_spend by 1000?"
        )
        assert parsed.intent == QuestionIntent.CAUSAL

    def test_explain_why(self):
        parsed = self.parser.parse(
            "Why do heavier objects fall at the same rate?"
        )
        assert parsed.intent == QuestionIntent.EXPLAIN

    def test_explain_how(self):
        parsed = self.parser.parse("How does gravity work?")
        assert parsed.intent == QuestionIntent.EXPLAIN

    def test_explain_keyword(self):
        parsed = self.parser.parse("Explain Newton's Second Law")
        assert parsed.intent == QuestionIntent.EXPLAIN

    def test_compare(self):
        parsed = self.parser.parse("Compare force and momentum")
        assert parsed.intent == QuestionIntent.COMPARE

    def test_general_fallback(self):
        parsed = self.parser.parse("Tell me about the universe")
        assert parsed.intent == QuestionIntent.GENERAL


class TestVariableExtraction:
    def setup_method(self):
        self.parser = QuestionParser()

    def test_extract_equals_format(self):
        parsed = self.parser.parse("What is F when m=10 and a=9.8?")
        assert parsed.variables.get("m") == pytest.approx(10.0)
        assert parsed.variables.get("a") == pytest.approx(9.8)

    def test_extract_equals_with_spaces(self):
        parsed = self.parser.parse("What is F when m = 10, a = 9.8?")
        assert parsed.variables.get("m") == pytest.approx(10.0)
        assert parsed.variables.get("a") == pytest.approx(9.8)

    def test_extract_no_variables(self):
        parsed = self.parser.parse("Why does gravity exist?")
        assert parsed.variables == {}

    def test_extract_single_variable(self):
        parsed = self.parser.parse("What is KE when v=3?")
        assert parsed.variables.get("v") == pytest.approx(3.0)

    def test_extract_negative_value(self):
        parsed = self.parser.parse("What is F when x=-5.0?")
        assert parsed.variables.get("x") == pytest.approx(-5.0)

    def test_extract_scientific_notation(self):
        parsed = self.parser.parse("What is E when m=1e3?")
        assert parsed.variables.get("m") == pytest.approx(1000.0)


class TestTargetVariable:
    def setup_method(self):
        self.parser = QuestionParser()

    def test_what_is_target(self):
        parsed = self.parser.parse("What is F when m=10?")
        assert parsed.target_var == "F"

    def test_find_target(self):
        parsed = self.parser.parse("Find KE when m=2 and v=3")
        assert parsed.target_var == "KE"

    def test_calculate_target(self):
        parsed = self.parser.parse("Calculate the force when m=5")
        assert parsed.target_var == "force"

    def test_no_target_for_explain(self):
        parsed = self.parser.parse("Why does F=ma?")
        # EXPLAIN intent does not need a target variable
        assert parsed.intent == QuestionIntent.EXPLAIN


class TestCausalParsing:
    def setup_method(self):
        self.parser = QuestionParser()

    def test_causal_node_extraction(self):
        parsed = self.parser.parse(
            "What happens if force increases by 10?"
        )
        assert parsed.causal_node == "force"
        assert parsed.causal_delta == pytest.approx(10.0)

    def test_causal_decrease(self):
        parsed = self.parser.parse(
            "What happens if price decreases by 5?"
        )
        assert parsed.causal_node == "price"
        assert parsed.causal_delta == pytest.approx(-5.0)

    def test_causal_effect_of_increasing(self):
        parsed = self.parser.parse(
            "What is the effect of increasing ad_spend by 1000?"
        )
        assert parsed.causal_node == "ad_spend"
        assert parsed.causal_delta == pytest.approx(1000.0)


class TestConditionExtraction:
    def setup_method(self):
        self.parser = QuestionParser()

    def test_extract_conditions_and(self):
        parsed = self.parser.parse(
            "Is the obligation valid if signed_contract and paid_deposit?"
        )
        assert "signed_contract" in parsed.conditions
        assert "paid_deposit" in parsed.conditions

    def test_extract_conditions_comma(self):
        parsed = self.parser.parse(
            "Does it hold when signed_contract, paid_deposit?"
        )
        assert "signed_contract" in parsed.conditions
        assert "paid_deposit" in parsed.conditions


class TestKeywordExtraction:
    def setup_method(self):
        self.parser = QuestionParser()

    def test_keywords_from_compute(self):
        parsed = self.parser.parse("What is F when m=10 and a=9.8?")
        assert len(parsed.keywords) > 0

    def test_keywords_from_causal(self):
        parsed = self.parser.parse("What happens if force increases?")
        assert "force" in parsed.keywords


class TestParsedQuestionDataclass:
    def test_original_preserved(self):
        parser = QuestionParser()
        q = "What is F when m=10?"
        parsed = parser.parse(q)
        assert parsed.original == q

    def test_all_fields_present(self):
        parser = QuestionParser()
        parsed = parser.parse("What is F when m=10?")
        assert hasattr(parsed, "original")
        assert hasattr(parsed, "intent")
        assert hasattr(parsed, "variables")
        assert hasattr(parsed, "target_var")
        assert hasattr(parsed, "conditions")
        assert hasattr(parsed, "causal_node")
        assert hasattr(parsed, "causal_delta")
        assert hasattr(parsed, "keywords")
```

### Step 2.2 -- Implement parser.py

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\parser.py`

```python
# mowm/qa/parser.py
"""QuestionParser — regex-based natural language question parsing.

Extracts structured information from questions:
  - Intent classification (compute, evaluate, causal, explain, compare)
  - Variable extraction (m=10.0, a=9.8)
  - Target variable identification (what is F -> target="F")
  - Causal node and delta extraction
  - Condition extraction for logic evaluation
  - Keyword extraction for domain matching
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QuestionIntent(Enum):
    """Classification of question intent."""

    COMPUTE = "compute"
    EVALUATE = "evaluate"
    CAUSAL = "causal"
    EXPLAIN = "explain"
    COMPARE = "compare"
    GENERAL = "general"


@dataclass
class ParsedQuestion:
    """Structured representation of a parsed question.

    Attributes:
        original: The original question text.
        intent: Detected question intent.
        variables: Extracted variable assignments {name: value}.
        target_var: The variable being asked about (e.g. "F").
        conditions: Logic conditions for EVALUATE intent.
        causal_node: Node name for CAUSAL intent.
        causal_delta: Change magnitude for CAUSAL intent.
        keywords: Extracted domain-relevant keywords.
    """

    original: str
    intent: QuestionIntent
    variables: dict[str, float] = field(default_factory=dict)
    target_var: str | None = None
    conditions: list[str] = field(default_factory=list)
    causal_node: str | None = None
    causal_delta: float | None = None
    keywords: list[str] = field(default_factory=list)


# -- Intent detection patterns ------------------------------------------------

_COMPUTE_PATTERNS = [
    re.compile(
        r"(?:what\s+is|find|calculate|compute|how\s+much)\b",
        re.IGNORECASE,
    ),
]

_EVALUATE_PATTERNS = [
    re.compile(
        r"(?:^is\s+(?:the|there|it)\b|^does\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:valid|hold|true|obligat)\b.*\b(?:if|when)\b",
        re.IGNORECASE,
    ),
]

_CAUSAL_PATTERNS = [
    re.compile(
        r"\bwhat\s+happens\s+if\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:effect|impact)\s+of\s+(?:increasing|decreasing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bif\s+\w+\s+(?:increases?|decreases?)\b",
        re.IGNORECASE,
    ),
]

_EXPLAIN_PATTERNS = [
    re.compile(r"^(?:why|how\s+does|explain)\b", re.IGNORECASE),
]

_COMPARE_PATTERNS = [
    re.compile(r"^compare\b", re.IGNORECASE),
    re.compile(r"\bcompare\s+\w+\s+and\b", re.IGNORECASE),
]

# -- Variable extraction patterns ---------------------------------------------

# Matches: m=10, m = 10, m=10.0, m = -9.8, m=1e3, m = 1.5e-2
_VAR_PATTERN = re.compile(
    r"\b([a-zA-Z_]\w*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

# -- Causal extraction patterns -----------------------------------------------

# "force increases by 10" or "price decreases by 5"
_CAUSAL_DELTA_PATTERN = re.compile(
    r"\b(\w+)\s+(increases?|decreases?)\s+by\s+"
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# "effect of increasing ad_spend by 1000"
_CAUSAL_EFFECT_PATTERN = re.compile(
    r"\b(?:effect|impact)\s+of\s+(increasing|decreasing)\s+"
    r"(\w+)\s+by\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# -- Target extraction patterns -----------------------------------------------

# "What is F when ...", "Find KE when ...", "Calculate the force when ..."
_TARGET_WHAT_IS = re.compile(
    r"(?:what\s+is|find|calculate)\s+(?:the\s+)?(\w+)",
    re.IGNORECASE,
)

# -- Condition extraction patterns --------------------------------------------

# "if A and B", "when A, B"
_CONDITION_PATTERN = re.compile(
    r"(?:if|when)\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)

# -- Keyword extraction -------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "when", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it",
    "its", "how", "why", "where", "there", "here", "up", "about",
    "then", "them", "they", "he", "she", "we", "you", "me", "him",
    "her", "us", "my", "your", "his", "our", "their",
})

_WORD_RE = re.compile(r"[a-zA-Z_]\w*")


class QuestionParser:
    """Parse natural language questions into structured components.

    Uses regex pattern matching for intent detection, variable
    extraction, target identification, and keyword extraction.
    No ML dependencies -- fast, deterministic, interpretable.
    """

    def parse(self, question: str) -> ParsedQuestion:
        """Parse a question into structured components.

        Args:
            question: Natural language question string.

        Returns:
            ParsedQuestion with all extracted fields populated.
        """
        intent = self._detect_intent(question)
        variables = self._extract_variables(question)
        target_var = self._extract_target(question, intent)
        conditions = self._extract_conditions(question, intent)
        causal_node, causal_delta = self._extract_causal(
            question, intent,
        )
        keywords = self._extract_keywords(question)

        return ParsedQuestion(
            original=question,
            intent=intent,
            variables=variables,
            target_var=target_var,
            conditions=conditions,
            causal_node=causal_node,
            causal_delta=causal_delta,
            keywords=keywords,
        )

    def _detect_intent(self, question: str) -> QuestionIntent:
        """Classify question intent using regex patterns.

        Priority order: CAUSAL > EVALUATE > EXPLAIN > COMPARE
        > COMPUTE > GENERAL.
        """
        # CAUSAL first — "what happens if X increases" must not
        # be captured by COMPUTE's "what" pattern
        for pat in _CAUSAL_PATTERNS:
            if pat.search(question):
                return QuestionIntent.CAUSAL

        # EVALUATE — "is the obligation valid if..."
        for pat in _EVALUATE_PATTERNS:
            if pat.search(question):
                return QuestionIntent.EVALUATE

        # EXPLAIN — "why / how does / explain"
        for pat in _EXPLAIN_PATTERNS:
            if pat.search(question):
                return QuestionIntent.EXPLAIN

        # COMPARE — "compare X and Y"
        for pat in _COMPARE_PATTERNS:
            if pat.search(question):
                return QuestionIntent.COMPARE

        # COMPUTE — "what is / find / calculate / how much"
        # Check AFTER causal/explain since "what" is ambiguous
        for pat in _COMPUTE_PATTERNS:
            if pat.search(question):
                return QuestionIntent.COMPUTE

        return QuestionIntent.GENERAL

    def _extract_variables(self, question: str) -> dict[str, float]:
        """Extract variable assignments like m=10, a=9.8."""
        variables: dict[str, float] = {}
        for match in _VAR_PATTERN.finditer(question):
            name = match.group(1)
            value = float(match.group(2))
            if name.lower() not in _STOPWORDS:
                variables[name] = value
        return variables

    def _extract_target(
        self,
        question: str,
        intent: QuestionIntent,
    ) -> str | None:
        """Extract the target variable being solved for."""
        if intent not in (
            QuestionIntent.COMPUTE,
            QuestionIntent.GENERAL,
        ):
            return None

        match = _TARGET_WHAT_IS.search(question)
        if match:
            target = match.group(1)
            if target.lower() not in _STOPWORDS:
                return target
        return None

    def _extract_causal(
        self,
        question: str,
        intent: QuestionIntent,
    ) -> tuple[str | None, float | None]:
        """Extract causal node and delta for CAUSAL intent."""
        if intent != QuestionIntent.CAUSAL:
            return None, None

        # Try "effect of increasing X by N" first
        match = _CAUSAL_EFFECT_PATTERN.search(question)
        if match:
            direction = match.group(1).lower()
            node = match.group(2)
            delta = float(match.group(3))
            if "decreas" in direction:
                delta = -delta
            return node, delta

        # Try "X increases/decreases by N"
        match = _CAUSAL_DELTA_PATTERN.search(question)
        if match:
            node = match.group(1)
            direction = match.group(2).lower()
            delta = float(match.group(3))
            if "decreas" in direction:
                delta = -delta
            return node, delta

        return None, None

    def _extract_conditions(
        self,
        question: str,
        intent: QuestionIntent,
    ) -> list[str]:
        """Extract logic conditions for EVALUATE intent."""
        if intent != QuestionIntent.EVALUATE:
            return []

        match = _CONDITION_PATTERN.search(question)
        if not match:
            return []

        clause = match.group(1).strip()
        # Split on "and" or commas
        parts = re.split(r"\s+and\s+|,\s*", clause)
        conditions: list[str] = []
        for part in parts:
            part = part.strip().rstrip("?").strip()
            # Extract identifiers (skip "is true", "is false", etc.)
            words = _WORD_RE.findall(part)
            for w in words:
                if (
                    w.lower() not in _STOPWORDS
                    and w.lower()
                    not in ("true", "false", "yes", "no")
                ):
                    conditions.append(w)
        return conditions

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract domain-relevant keywords from the question."""
        words = _WORD_RE.findall(question)
        keywords: list[str] = []
        seen: set[str] = set()
        for w in words:
            low = w.lower()
            if low not in _STOPWORDS and low not in seen and len(w) > 1:
                keywords.append(w)
                seen.add(low)
        return keywords
```

### Step 2.3 -- Run tests

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run pytest tests/test_parser.py -v --tb=short`
- [ ] Verify all tests pass

---

## Task 3: qa/classifier.py -- DomainClassifier

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\classifier.py`

Depends on: AxiomLibrary, WorldEncoder (from cubemind).

### Step 3.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_classifier.py`

```python
# tests/test_classifier.py
"""Tests for DomainClassifier — domain routing via cosine sim and keywords."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.qa.classifier import DomainClassifier

K, L = 4, 32


def _make_library() -> AxiomLibrary:
    """Create a small library with physics and logic axioms."""
    lib = AxiomLibrary(k=K, l=L)
    # Physics
    ax1 = lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    )
    lib.register(ax1)
    ax2 = lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    )
    lib.register(ax2)
    # Logic
    ax3 = lib.make_axiom(
        "Modus Ponens", "logic.deductive",
        "(P AND (P IMPLIES Q)) IMPLIES Q",
    )
    lib.register(ax3)
    # Economics
    ax4 = lib.make_axiom(
        "Supply and Demand", "economics.market",
        "price = demand / supply",
    )
    lib.register(ax4)
    return lib


class TestDomainClassifierKeywordFallback:
    def setup_method(self):
        lib = _make_library()
        self.clf = DomainClassifier(library=lib, k=K, l=L)

    def test_classify_returns_ranked_list(self):
        result = self.clf.classify("What is the force?")
        assert isinstance(result, list)
        assert len(result) > 0
        # Each entry is (domain, score)
        domain, score = result[0]
        assert isinstance(domain, str)
        assert isinstance(score, float)

    def test_top_domain_returns_string(self):
        domain = self.clf.top_domain("What is the force?")
        assert isinstance(domain, str)
        assert len(domain) > 0

    def test_physics_question_matches_physics(self):
        domain = self.clf.top_domain(
            "What is the force on a 10kg object?"
        )
        assert domain == "physics"

    def test_logic_question_matches_logic(self):
        domain = self.clf.top_domain(
            "Is the obligation valid if signed_contract?"
        )
        # "obligation" is close to logic domain keywords
        assert domain in ("logic", "economics")

    def test_classify_scores_are_normalized(self):
        result = self.clf.classify("Calculate the kinetic energy")
        for _, score in result:
            assert 0.0 <= score <= 1.0

    def test_classify_empty_question(self):
        result = self.clf.classify("")
        assert isinstance(result, list)

    def test_top_domain_with_unknown_topic(self):
        domain = self.clf.top_domain("xyzzy frobulate quux")
        # Should still return some domain (fallback)
        assert isinstance(domain, str)


class TestDomainClassifierVSA:
    def setup_method(self):
        lib = _make_library()
        self.clf = DomainClassifier(library=lib, k=K, l=L)

    def test_classify_embedding(self):
        # Create a random embedding and classify it
        rng = np.random.default_rng(42)
        embedding = rng.standard_normal((K, L)).astype(np.float32)
        result = self.clf.classify_embedding(embedding)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_classify_embedding_shape(self):
        rng = np.random.default_rng(42)
        embedding = rng.standard_normal((K, L)).astype(np.float32)
        result = self.clf.classify_embedding(embedding)
        for domain, score in result:
            assert isinstance(domain, str)
            assert isinstance(score, float)


class TestDomainClassifierEdgeCases:
    def test_empty_library(self):
        lib = AxiomLibrary(k=K, l=L)
        clf = DomainClassifier(library=lib, k=K, l=L)
        result = clf.classify("What is force?")
        assert result == []

    def test_single_domain_library(self):
        lib = AxiomLibrary(k=K, l=L)
        ax = lib.make_axiom(
            "Newton's Second Law", "physics.dynamics", "F = m * a",
        )
        lib.register(ax)
        clf = DomainClassifier(library=lib, k=K, l=L)
        domain = clf.top_domain("What is the force?")
        assert domain == "physics"
```

### Step 3.2 -- Implement classifier.py

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\classifier.py`

```python
# mowm/qa/classifier.py
"""DomainClassifier — route questions to the most relevant domain.

Two classification strategies:
  1. VSA cosine similarity: encode question via WorldEncoder, compare
     against domain centroid vectors computed from AxiomLibrary.
  2. Keyword fallback: match question words against axiom names when
     no codebook is available.

Both strategies return a ranked list of (domain, score) pairs.
"""
from __future__ import annotations

import re

import numpy as np

from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes

from mowm.axiom_library import Axiom, AxiomLibrary


_WORD_RE = re.compile(r"[a-zA-Z_]\w*")


class DomainClassifier:
    """Route questions to the most relevant domain.

    Uses VSA cosine similarity against domain centroids computed
    from the axiom library. Falls back to keyword matching if
    no codebook is loaded.

    Args:
        library: AxiomLibrary with registered axioms.
        k: Number of VSA blocks.
        l: Block length.
        codebook_path: Optional path to NV-Embed codebook .npz.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        k: int = 80,
        l: int = 128,  # noqa: E741
        codebook_path: str | None = None,
    ) -> None:
        self._library = library
        self._k = k
        self._l = l
        self._bc = BlockCodes(k=k, l=l)
        self._encoder = WorldEncoder(k=k, l=l)
        self._codebook_path = codebook_path

        # Build domain centroids from the library
        self._domain_centroids: dict[str, np.ndarray] = {}
        self._domain_keywords: dict[str, set[str]] = {}
        self._build_centroids()

    def _build_centroids(self) -> None:
        """Compute domain centroids by averaging axiom vectors per domain."""
        domain_groups = self._library.domains()
        for domain_name, axioms in domain_groups.items():
            if not axioms:
                continue
            vecs = np.stack(
                [a.vector for a in axioms], axis=0,
            )
            centroid = vecs.mean(axis=0).astype(np.float32)
            # Normalize per block
            norms = np.linalg.norm(centroid, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._domain_centroids[domain_name] = (
                centroid / norms
            ).astype(np.float32)

            # Build keyword set from axiom names and formulas
            keywords: set[str] = set()
            for ax in axioms:
                words = _WORD_RE.findall(ax.name.lower())
                keywords.update(words)
                formula_words = _WORD_RE.findall(ax.formula_str.lower())
                keywords.update(formula_words)
            self._domain_keywords[domain_name] = keywords

    def classify(self, question: str) -> list[tuple[str, float]]:
        """Return ranked list of (domain, score) pairs.

        Combines VSA cosine similarity with keyword matching.

        Args:
            question: Natural language question string.

        Returns:
            List of (domain_name, score) sorted by descending score.
            Scores are in [0, 1].
        """
        if not self._domain_centroids:
            return []

        # VSA similarity
        query_vec = self._encoder.encode_action(question)
        vsa_scores = self._compute_vsa_scores(query_vec)

        # Keyword similarity
        kw_scores = self._compute_keyword_scores(question)

        # Blend: 60% VSA + 40% keyword
        combined: dict[str, float] = {}
        all_domains = set(vsa_scores.keys()) | set(kw_scores.keys())
        for domain in all_domains:
            vsa_s = vsa_scores.get(domain, 0.0)
            kw_s = kw_scores.get(domain, 0.0)
            combined[domain] = 0.6 * vsa_s + 0.4 * kw_s

        # Normalize to [0, 1]
        max_score = max(combined.values()) if combined else 1.0
        if max_score > 0:
            combined = {
                d: s / max_score for d, s in combined.items()
            }

        ranked = sorted(
            combined.items(), key=lambda x: x[1], reverse=True,
        )
        return ranked

    def classify_embedding(
        self,
        embedding: np.ndarray,
    ) -> list[tuple[str, float]]:
        """Classify from a pre-computed VSA block-code embedding.

        Args:
            embedding: Block-code vector of shape (k, l).

        Returns:
            Ranked list of (domain, score) pairs.
        """
        if not self._domain_centroids:
            return []

        scores = self._compute_vsa_scores(embedding)
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {d: s / max_score for d, s in scores.items()}

        return sorted(
            scores.items(), key=lambda x: x[1], reverse=True,
        )

    def top_domain(self, question: str) -> str:
        """Return the single best-matching domain name.

        Args:
            question: Natural language question string.

        Returns:
            Domain name string. Returns "general" if no domains
            are available.
        """
        ranked = self.classify(question)
        if not ranked:
            return "general"
        return ranked[0][0]

    def _compute_vsa_scores(
        self,
        query_vec: np.ndarray,
    ) -> dict[str, float]:
        """Compute cosine similarity of query against all centroids."""
        scores: dict[str, float] = {}
        for domain, centroid in self._domain_centroids.items():
            sim = self._bc.similarity(query_vec, centroid)
            # Clamp to [0, 1] — negative similarity treated as 0
            scores[domain] = float(max(0.0, sim))
        return scores

    def _compute_keyword_scores(
        self,
        question: str,
    ) -> dict[str, float]:
        """Compute keyword overlap between question and domain keywords."""
        q_words = set(
            w.lower() for w in _WORD_RE.findall(question)
        )
        scores: dict[str, float] = {}
        for domain, keywords in self._domain_keywords.items():
            if not keywords:
                scores[domain] = 0.0
                continue
            overlap = len(q_words & keywords)
            # Normalize by question word count to avoid length bias
            if q_words:
                scores[domain] = overlap / len(q_words)
            else:
                scores[domain] = 0.0
        return scores
```

### Step 3.3 -- Run tests

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run pytest tests/test_classifier.py -v --tb=short`
- [ ] Verify all tests pass

---

## Task 4: qa/direct_solver.py -- DirectSolver

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\direct_solver.py`

Depends on: AxiomLibrary, SafeFormulaExecutor, LogicEngine, CausalGraph, QuestionParser, ResponseFormatter, QAResponse (from `__init__.py`, but we forward-declare it here).

### Step 4.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_direct_solver.py`

```python
# tests/test_direct_solver.py
"""Tests for DirectSolver — compute answers via executor/logic/causal."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa.direct_solver import DirectSolver
from mowm.qa.parser import ParsedQuestion, QuestionIntent

K, L = 4, 32


def _make_library() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    ax = lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    )
    lib.register(ax)
    ax2 = lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    )
    lib.register(ax2)
    ax3 = lib.make_axiom(
        "Ohm's Law", "physics.em", "V = I * R",
    )
    lib.register(ax3)
    return lib


def _make_causal_graph() -> CausalGraph:
    g = CausalGraph(k=K, l=L)
    rng = np.random.default_rng(42)
    for name in ("force", "acceleration", "velocity"):
        vec = rng.standard_normal((K, L)).astype(np.float32)
        g.add_node(name, "physics", vec)
    g.add_edge("force", "acceleration", "+", 0.9)
    g.add_edge("acceleration", "velocity", "+", 0.8)
    return g


class TestDirectSolverCompute:
    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.logic = LogicEngine()
        self.solver = DirectSolver(
            library=self.lib,
            executor=self.executor,
            logic=self.logic,
        )

    def test_can_solve_compute_with_matching_axiom(self):
        parsed = ParsedQuestion(
            original="What is F when m=10, a=9.8?",
            intent=QuestionIntent.COMPUTE,
            variables={"m": 10.0, "a": 9.8},
            target_var="F",
        )
        assert self.solver.can_solve(parsed, "physics") is True

    def test_solve_compute_newton(self):
        parsed = ParsedQuestion(
            original="What is F when m=10, a=9.8?",
            intent=QuestionIntent.COMPUTE,
            variables={"m": 10.0, "a": 9.8},
            target_var="F",
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "direct"
        assert response.confidence > 0.0
        assert response.computed_values.get("F") == pytest.approx(98.0)
        assert len(response.evidence) > 0
        assert "Newton" in response.evidence[0]

    def test_solve_compute_kinetic_energy(self):
        parsed = ParsedQuestion(
            original="What is KE when m=2, v=3?",
            intent=QuestionIntent.COMPUTE,
            variables={"m": 2.0, "v": 3.0},
            target_var="KE",
        )
        response = self.solver.solve(parsed, "physics")
        assert response.computed_values.get("KE") == pytest.approx(9.0)

    def test_cannot_solve_missing_variables(self):
        parsed = ParsedQuestion(
            original="What is F when m=10?",
            intent=QuestionIntent.COMPUTE,
            variables={"m": 10.0},
            target_var="F",
        )
        # Missing "a" variable -- can still try but result will be 0
        # can_solve returns True if axiom matches, solve handles missing
        assert self.solver.can_solve(parsed, "physics") is True

    def test_cannot_solve_no_matching_axiom(self):
        parsed = ParsedQuestion(
            original="What is Z when x=1?",
            intent=QuestionIntent.COMPUTE,
            variables={"x": 1.0},
            target_var="Z",
        )
        assert self.solver.can_solve(parsed, "physics") is False


class TestDirectSolverEvaluate:
    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.logic = LogicEngine()
        self.solver = DirectSolver(
            library=self.lib,
            executor=self.executor,
            logic=self.logic,
        )

    def test_can_solve_evaluate(self):
        parsed = ParsedQuestion(
            original="Is the obligation valid?",
            intent=QuestionIntent.EVALUATE,
            conditions=["signed_contract", "paid_deposit"],
        )
        assert self.solver.can_solve(parsed, "logic") is True

    def test_solve_evaluate_obligation_holds(self):
        parsed = ParsedQuestion(
            original="Is the obligation valid?",
            intent=QuestionIntent.EVALUATE,
            conditions=["signed_contract", "paid_deposit"],
            variables={},
        )
        response = self.solver.solve(
            parsed, "logic",
            facts={"signed_contract": True, "paid_deposit": True},
        )
        assert response.mode == "direct"
        assert response.computed_values.get("result") is True

    def test_solve_evaluate_obligation_fails(self):
        parsed = ParsedQuestion(
            original="Is the obligation valid?",
            intent=QuestionIntent.EVALUATE,
            conditions=["signed_contract", "paid_deposit"],
            variables={},
        )
        response = self.solver.solve(
            parsed, "logic",
            facts={"signed_contract": True, "paid_deposit": False},
        )
        assert response.computed_values.get("result") is False

    def test_evaluate_with_logic_formula(self):
        parsed = ParsedQuestion(
            original="Is A AND B true?",
            intent=QuestionIntent.EVALUATE,
            conditions=["A", "B"],
            variables={},
        )
        response = self.solver.solve(
            parsed, "logic",
            facts={"A": True, "B": True},
        )
        assert response.computed_values.get("result") is True


class TestDirectSolverCausal:
    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.logic = LogicEngine()
        self.causal = _make_causal_graph()
        self.solver = DirectSolver(
            library=self.lib,
            executor=self.executor,
            logic=self.logic,
            causal=self.causal,
        )

    def test_can_solve_causal_known_node(self):
        parsed = ParsedQuestion(
            original="What happens if force increases by 10?",
            intent=QuestionIntent.CAUSAL,
            causal_node="force",
            causal_delta=10.0,
        )
        assert self.solver.can_solve(parsed, "physics") is True

    def test_can_solve_causal_unknown_node(self):
        parsed = ParsedQuestion(
            original="What happens if magic increases by 10?",
            intent=QuestionIntent.CAUSAL,
            causal_node="magic",
            causal_delta=10.0,
        )
        assert self.solver.can_solve(parsed, "physics") is False

    def test_solve_causal_propagation(self):
        parsed = ParsedQuestion(
            original="What happens if force increases by 10?",
            intent=QuestionIntent.CAUSAL,
            causal_node="force",
            causal_delta=10.0,
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "direct"
        vals = response.computed_values
        assert "acceleration" in vals
        assert vals["acceleration"] == pytest.approx(9.0)
        assert "velocity" in vals
        assert vals["velocity"] == pytest.approx(7.2)

    def test_causal_no_graph(self):
        solver_no_causal = DirectSolver(
            library=self.lib,
            executor=self.executor,
            logic=self.logic,
            causal=None,
        )
        parsed = ParsedQuestion(
            original="What happens if force increases by 10?",
            intent=QuestionIntent.CAUSAL,
            causal_node="force",
            causal_delta=10.0,
        )
        assert solver_no_causal.can_solve(parsed, "physics") is False


class TestDirectSolverExplainAndGeneral:
    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.logic = LogicEngine()
        self.solver = DirectSolver(
            library=self.lib,
            executor=self.executor,
            logic=self.logic,
        )

    def test_cannot_solve_explain(self):
        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
        )
        assert self.solver.can_solve(parsed, "physics") is False

    def test_cannot_solve_general(self):
        parsed = ParsedQuestion(
            original="Tell me about the universe",
            intent=QuestionIntent.GENERAL,
        )
        assert self.solver.can_solve(parsed, "physics") is False

    def test_cannot_solve_compare(self):
        parsed = ParsedQuestion(
            original="Compare force and energy",
            intent=QuestionIntent.COMPARE,
        )
        assert self.solver.can_solve(parsed, "physics") is False
```

### Step 4.2 -- Implement direct_solver.py

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\direct_solver.py`

```python
# mowm/qa/direct_solver.py
"""DirectSolver — compute answers without an LLM.

Routes parsed questions to the appropriate computation engine:
  - COMPUTE: find matching axiom, execute formula via SafeFormulaExecutor
  - EVALUATE: logic evaluation via LogicEngine (obligation check or formula)
  - CAUSAL: propagate change through CausalGraph

Returns structured QAResponse with computed values and evidence trail.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa.formatter import ResponseFormatter
from mowm.qa.parser import ParsedQuestion, QuestionIntent

if TYPE_CHECKING:
    pass

# Forward reference — QAResponse is defined in qa/__init__.py but we
# need to avoid a circular import. We define a minimal version here
# and the __init__.py re-exports the canonical one.
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QAResponse:
    """Structured response from the Q&A pipeline.

    Attributes:
        answer: Natural language answer string.
        mode: "direct" or "augmented".
        confidence: Confidence score in [0, 1].
        domain: Matched domain name.
        evidence: List of axiom/formula references used.
        computed_values: Dict of computed results.
        reasoning_trace: Step-by-step reasoning log.
    """

    answer: str
    mode: Literal["direct", "augmented"]
    confidence: float
    domain: str
    evidence: list[str] = field(default_factory=list)
    computed_values: dict = field(default_factory=dict)
    reasoning_trace: list[str] = field(default_factory=list)


class DirectSolver:
    """Compute answers directly using executor, logic engine, and causal graph.

    Handles COMPUTE, EVALUATE, and CAUSAL intents without any LLM.

    Args:
        library: AxiomLibrary for finding matching axioms.
        executor: SafeFormulaExecutor for formula evaluation.
        logic: LogicEngine for boolean/probabilistic evaluation.
        causal: Optional CausalGraph for causal propagation.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
        logic: LogicEngine,
        causal: CausalGraph | None = None,
    ) -> None:
        self._library = library
        self._executor = executor
        self._logic = logic
        self._causal = causal
        self._formatter = ResponseFormatter()

    def can_solve(
        self,
        parsed: ParsedQuestion,
        domain: str,
    ) -> bool:
        """Check if this question can be solved directly.

        Returns True if:
          - COMPUTE intent + matching axiom formula found
          - EVALUATE intent + conditions are provided
          - CAUSAL intent + causal graph has the node
        """
        if parsed.intent == QuestionIntent.COMPUTE:
            axiom = self._find_matching_axiom(parsed, domain)
            return axiom is not None

        if parsed.intent == QuestionIntent.EVALUATE:
            return len(parsed.conditions) > 0

        if parsed.intent == QuestionIntent.CAUSAL:
            if self._causal is None:
                return False
            if parsed.causal_node is None:
                return False
            return parsed.causal_node in self._causal.nodes

        # EXPLAIN, COMPARE, GENERAL cannot be solved directly
        return False

    def solve(
        self,
        parsed: ParsedQuestion,
        domain: str,
        facts: dict[str, bool] | None = None,
    ) -> QAResponse:
        """Solve the question directly.

        Args:
            parsed: Parsed question structure.
            domain: Top-level domain name.
            facts: Optional boolean facts for logic evaluation.

        Returns:
            QAResponse with computed answer and evidence.
        """
        if parsed.intent == QuestionIntent.COMPUTE:
            return self._solve_compute(parsed, domain)
        if parsed.intent == QuestionIntent.EVALUATE:
            return self._solve_evaluate(parsed, domain, facts)
        if parsed.intent == QuestionIntent.CAUSAL:
            return self._solve_causal(parsed, domain)

        return QAResponse(
            answer="Cannot solve this question directly.",
            mode="direct",
            confidence=0.0,
            domain=domain,
            reasoning_trace=["No direct solver available for this intent."],
        )

    def _solve_compute(
        self,
        parsed: ParsedQuestion,
        domain: str,
    ) -> QAResponse:
        """Solve a COMPUTE question via formula execution."""
        trace: list[str] = []
        trace.append(f"Intent: COMPUTE, target={parsed.target_var}")
        trace.append(f"Variables: {parsed.variables}")

        axiom = self._find_matching_axiom(parsed, domain)
        if axiom is None:
            return QAResponse(
                answer="No matching axiom found for this computation.",
                mode="direct",
                confidence=0.0,
                domain=domain,
                reasoning_trace=trace,
            )

        trace.append(f"Matched axiom: {axiom.name} ({axiom.formula_str})")

        try:
            result = self._executor.execute(
                axiom.formula_str, parsed.variables,
            )
        except Exception as exc:
            trace.append(f"Execution error: {exc}")
            return QAResponse(
                answer=f"Error computing {axiom.formula_str}: {exc}",
                mode="direct",
                confidence=0.0,
                domain=domain,
                evidence=[f"{axiom.name}: {axiom.formula_str}"],
                reasoning_trace=trace,
            )

        trace.append(f"Result: {result}")

        target = parsed.target_var or self._extract_lhs(axiom.formula_str)
        computed_values: dict = {}
        if target:
            computed_values[target] = result
        else:
            computed_values["result"] = result

        answer_text = self._formatter.format_compute(
            axiom_name=axiom.name,
            formula=axiom.formula_str,
            variables=parsed.variables,
            result=result,
        )

        return QAResponse(
            answer=answer_text,
            mode="direct",
            confidence=0.95,
            domain=domain,
            evidence=[f"{axiom.name}: {axiom.formula_str}"],
            computed_values=computed_values,
            reasoning_trace=trace,
        )

    def _solve_evaluate(
        self,
        parsed: ParsedQuestion,
        domain: str,
        facts: dict[str, bool] | None = None,
    ) -> QAResponse:
        """Solve an EVALUATE question via logic engine."""
        trace: list[str] = []
        trace.append(f"Intent: EVALUATE, conditions={parsed.conditions}")

        if facts is None:
            # Default: all conditions True (optimistic)
            facts = {c: True for c in parsed.conditions}

        trace.append(f"Facts: {facts}")

        result = self._logic.check_obligation(
            conditions=parsed.conditions,
            facts=facts,
        )

        trace.append(f"Obligation result: {result}")

        answer_text = self._formatter.format_evaluate(
            conditions=parsed.conditions,
            result=result,
        )

        return QAResponse(
            answer=answer_text,
            mode="direct",
            confidence=0.95,
            domain=domain,
            evidence=[f"Obligation check: {parsed.conditions}"],
            computed_values={"result": result},
            reasoning_trace=trace,
        )

    def _solve_causal(
        self,
        parsed: ParsedQuestion,
        domain: str,
    ) -> QAResponse:
        """Solve a CAUSAL question via graph propagation."""
        trace: list[str] = []
        trace.append(
            f"Intent: CAUSAL, node={parsed.causal_node}, "
            f"delta={parsed.causal_delta}"
        )

        if self._causal is None or parsed.causal_node is None:
            return QAResponse(
                answer="No causal graph available.",
                mode="direct",
                confidence=0.0,
                domain=domain,
                reasoning_trace=trace,
            )

        delta = parsed.causal_delta if parsed.causal_delta is not None else 1.0

        try:
            effects = self._causal.propagate(
                parsed.causal_node, delta,
            )
        except KeyError as exc:
            trace.append(f"Node not found: {exc}")
            return QAResponse(
                answer=f"Causal node '{parsed.causal_node}' not found.",
                mode="direct",
                confidence=0.0,
                domain=domain,
                reasoning_trace=trace,
            )

        trace.append(f"Propagated effects: {effects}")

        answer_text = self._formatter.format_causal(
            node=parsed.causal_node,
            delta=delta,
            effects=effects,
        )

        return QAResponse(
            answer=answer_text,
            mode="direct",
            confidence=0.85,
            domain=domain,
            evidence=[
                f"Causal propagation from {parsed.causal_node}"
            ],
            computed_values=effects,
            reasoning_trace=trace,
        )

    def _find_matching_axiom(
        self,
        parsed: ParsedQuestion,
        domain: str,
    ) -> Axiom | None:
        """Find an axiom whose formula contains the target variable.

        Search strategy:
          1. Look for axioms in the matching domain
          2. Check if the formula LHS matches the target variable
          3. Fall back to keyword matching in axiom name
        """
        target = parsed.target_var
        if target is None:
            return None

        target_lower = target.lower()
        all_axioms = self._library.all_axioms()

        # First pass: exact LHS match in matching domain
        for ax in all_axioms:
            ax_domain = ax.domain.split(".")[0]
            if ax_domain != domain:
                continue
            lhs = self._extract_lhs(ax.formula_str)
            if lhs and lhs == target:
                return ax

        # Second pass: target appears anywhere in formula, same domain
        for ax in all_axioms:
            ax_domain = ax.domain.split(".")[0]
            if ax_domain != domain:
                continue
            if target in ax.formula_str:
                return ax

        # Third pass: keyword match in axiom name (any domain)
        for ax in all_axioms:
            if target_lower in ax.name.lower():
                return ax
            lhs = self._extract_lhs(ax.formula_str)
            if lhs and lhs == target:
                return ax

        return None

    @staticmethod
    def _extract_lhs(formula: str) -> str | None:
        """Extract LHS variable from 'F = m * a' style formulas."""
        if "=" not in formula or "==" in formula:
            return None
        lhs = formula.split("=", 1)[0].strip()
        if re.match(r"^[a-zA-Z_]\w*$", lhs):
            return lhs
        return None
```

### Step 4.3 -- Run tests

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run pytest tests/test_direct_solver.py -v --tb=short`
- [ ] Verify all tests pass

---

## Task 5: qa/augmented_solver.py -- AugmentedSolver

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\augmented_solver.py`

Depends on: AxiomLibrary, MoWMPipeline (optional), ResponseFormatter, QAResponse, httpx.

### Step 5.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_augmented_solver.py`

```python
# tests/test_augmented_solver.py
"""Tests for AugmentedSolver — LLM-augmented answers with template fallback."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.qa.augmented_solver import AugmentedSolver
from mowm.qa.direct_solver import QAResponse
from mowm.qa.parser import ParsedQuestion, QuestionIntent

K, L = 4, 32


def _make_library() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    ax1 = lib.make_axiom(
        "Newton's Second Law", "physics.dynamics", "F = m * a",
    )
    lib.register(ax1)
    ax2 = lib.make_axiom(
        "Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2",
    )
    lib.register(ax2)
    ax3 = lib.make_axiom(
        "Gravitational Force", "physics.dynamics",
        "F = G * m1 * m2 / r^2",
    )
    lib.register(ax3)
    return lib


class TestAugmentedSolverTemplateFallback:
    """Test template-based responses when no LLM is available."""

    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.solver = AugmentedSolver(
            library=self.lib,
            executor=self.executor,
            llm_endpoint=None,  # No LLM — use template fallback
        )

    def test_solve_explain_template(self):
        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
            keywords=["F", "ma", "force"],
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "augmented"
        assert response.confidence > 0.0
        assert len(response.answer) > 0
        assert len(response.evidence) > 0

    def test_solve_general_template(self):
        parsed = ParsedQuestion(
            original="Tell me about gravity",
            intent=QuestionIntent.GENERAL,
            keywords=["gravity"],
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "augmented"
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0

    def test_solve_compare_template(self):
        parsed = ParsedQuestion(
            original="Compare force and energy",
            intent=QuestionIntent.COMPARE,
            keywords=["force", "energy"],
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "augmented"
        assert len(response.answer) > 0

    def test_solve_with_partial_computation(self):
        parsed = ParsedQuestion(
            original="Explain F=ma with m=10, a=9.8",
            intent=QuestionIntent.EXPLAIN,
            variables={"m": 10.0, "a": 9.8},
            keywords=["F", "force", "mass", "acceleration"],
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "augmented"
        # Should include computed values as supporting evidence
        assert len(response.evidence) > 0

    def test_evidence_includes_axiom_names(self):
        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
            keywords=["force", "Second", "Law"],
        )
        response = self.solver.solve(parsed, "physics")
        # Should reference relevant axioms
        assert any(
            "Newton" in e or "Second" in e or "Force" in e
            for e in response.evidence
        )


class TestAugmentedSolverContextBuilding:
    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.solver = AugmentedSolver(
            library=self.lib,
            executor=self.executor,
            llm_endpoint=None,
        )

    def test_build_context_includes_domain(self):
        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
            keywords=["force"],
        )
        context = self.solver._build_context(parsed, "physics")
        assert "physics" in context.lower()

    def test_build_context_includes_axioms(self):
        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
            keywords=["force"],
        )
        context = self.solver._build_context(parsed, "physics")
        assert "axiom" in context.lower() or "law" in context.lower() \
            or "F = m * a" in context

    def test_build_context_includes_question(self):
        q = "Why does F=ma?"
        parsed = ParsedQuestion(
            original=q,
            intent=QuestionIntent.EXPLAIN,
            keywords=["force"],
        )
        context = self.solver._build_context(parsed, "physics")
        assert q in context


class TestAugmentedSolverLLMCall:
    """Test LLM call path with mocked httpx."""

    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.solver = AugmentedSolver(
            library=self.lib,
            executor=self.executor,
            llm_endpoint="http://localhost:8000/v1/chat/completions",
        )

    @patch("mowm.qa.augmented_solver.httpx")
    def test_llm_call_success(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Force equals mass times acceleration."
                    }
                }
            ],
        }
        mock_httpx.post.return_value = mock_response

        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
            keywords=["force"],
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "augmented"
        assert "Force equals mass" in response.answer
        mock_httpx.post.assert_called_once()

    @patch("mowm.qa.augmented_solver.httpx")
    def test_llm_call_failure_falls_back_to_template(self, mock_httpx):
        mock_httpx.post.side_effect = Exception("Connection refused")

        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
            keywords=["force"],
        )
        response = self.solver.solve(parsed, "physics")
        # Should fall back to template
        assert response.mode == "augmented"
        assert len(response.answer) > 0

    @patch("mowm.qa.augmented_solver.httpx")
    def test_llm_call_timeout_falls_back(self, mock_httpx):
        mock_httpx.post.side_effect = TimeoutError("Timed out")

        parsed = ParsedQuestion(
            original="Explain gravity",
            intent=QuestionIntent.EXPLAIN,
            keywords=["gravity"],
        )
        response = self.solver.solve(parsed, "physics")
        assert response.mode == "augmented"
        assert len(response.answer) > 0


class TestAugmentedSolverLLMPromptFormat:
    def setup_method(self):
        self.lib = _make_library()
        self.executor = SafeFormulaExecutor()
        self.solver = AugmentedSolver(
            library=self.lib,
            executor=self.executor,
            llm_endpoint="http://localhost:8000/v1/chat/completions",
        )

    def test_build_llm_payload(self):
        parsed = ParsedQuestion(
            original="Why does F=ma?",
            intent=QuestionIntent.EXPLAIN,
            keywords=["force"],
        )
        context = self.solver._build_context(parsed, "physics")
        payload = self.solver._build_llm_payload(context)
        assert "messages" in payload
        assert len(payload["messages"]) >= 2
        # System message
        assert payload["messages"][0]["role"] == "system"
        # User message contains context
        assert payload["messages"][1]["role"] == "user"


class TestAugmentedSolverOpenRouterAPIKey:
    def test_openrouter_uses_env_key(self):
        lib = _make_library()
        executor = SafeFormulaExecutor()
        solver = AugmentedSolver(
            library=lib,
            executor=executor,
            llm_endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key="test-key-123",
        )
        assert solver._api_key == "test-key-123"

    def test_no_api_key_defaults_none(self):
        lib = _make_library()
        executor = SafeFormulaExecutor()
        solver = AugmentedSolver(
            library=lib,
            executor=executor,
            llm_endpoint=None,
        )
        assert solver._api_key is None
```

### Step 5.2 -- Implement augmented_solver.py

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\augmented_solver.py`

```python
# mowm/qa/augmented_solver.py
"""AugmentedSolver — package MoWM reasoning context for LLM generation.

Workflow:
  1. Select relevant axioms from AxiomLibrary (top 5 by similarity)
  2. Execute any computable axioms for supporting evidence
  3. Build structured prompt with domain, axioms, computed values
  4. Call LLM endpoint via httpx POST (OpenRouter format)
  5. Fall back to template-based response if no LLM available

Template fallback ensures core functionality works without any LLM.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.qa.direct_solver import QAResponse
from mowm.qa.formatter import ResponseFormatter
from mowm.qa.parser import ParsedQuestion, QuestionIntent

# System prompt for LLM calls
_SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. Use the provided domain "
    "knowledge and computed values to answer the question. Cite specific "
    "formulas and values. Be concise and accurate."
)


class AugmentedSolver:
    """Generate answers via LLM with MoWM reasoning context.

    When an LLM endpoint is available, sends structured context
    (domain, axioms, computed values) to the LLM for fluent
    natural language generation.

    When no LLM is available, falls back to template-based
    responses using the structured context directly.

    Args:
        library: AxiomLibrary for finding relevant axioms.
        executor: SafeFormulaExecutor for supporting computations.
        llm_endpoint: URL for LLM API (OpenRouter/local format).
        api_key: API key for the LLM endpoint.
        model: Model name for OpenRouter (default: qwen/qwen-2.5-72b).
        timeout: Request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
        llm_endpoint: str | None = None,
        api_key: str | None = None,
        model: str = "qwen/qwen-2.5-72b-instruct",
        timeout: float = 30.0,
    ) -> None:
        self._library = library
        self._executor = executor
        self._llm_endpoint = llm_endpoint
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._model = model
        self._timeout = timeout
        self._formatter = ResponseFormatter()

    def solve(
        self,
        parsed: ParsedQuestion,
        domain: str,
        direct_result: QAResponse | None = None,
    ) -> QAResponse:
        """Generate answer via LLM or template with MoWM context.

        Args:
            parsed: Parsed question structure.
            domain: Top-level domain name.
            direct_result: Optional partial result from DirectSolver.

        Returns:
            QAResponse with augmented answer and evidence.
        """
        trace: list[str] = []
        trace.append(f"Intent: {parsed.intent.value}")
        trace.append(f"Domain: {domain}")

        # Find relevant axioms
        relevant_axioms = self._find_relevant_axioms(parsed, domain)
        evidence = [f"{ax.name}: {ax.formula_str}" for ax in relevant_axioms]
        trace.append(f"Relevant axioms: {len(relevant_axioms)}")

        # Execute any computable axioms for supporting evidence
        computed_values = self._compute_supporting_values(
            parsed, relevant_axioms,
        )
        if computed_values:
            trace.append(f"Computed values: {computed_values}")

        # Include direct solver results if available
        if direct_result and direct_result.computed_values:
            computed_values.update(direct_result.computed_values)
            evidence.extend(direct_result.evidence)

        # Build context
        context = self._build_context(
            parsed, domain, relevant_axioms, computed_values,
        )

        # Try LLM call first, fall back to template
        answer = self._generate_answer(
            context, parsed, domain, relevant_axioms,
            computed_values, trace,
        )

        return QAResponse(
            answer=answer,
            mode="augmented",
            confidence=0.75 if self._llm_endpoint else 0.60,
            domain=domain,
            evidence=evidence,
            computed_values=computed_values,
            reasoning_trace=trace,
        )

    def _find_relevant_axioms(
        self,
        parsed: ParsedQuestion,
        domain: str,
    ) -> list[Axiom]:
        """Select top relevant axioms from library.

        Uses keyword matching to filter axioms by domain and
        question keywords.
        """
        all_axioms = self._library.all_axioms()
        if not all_axioms:
            return []

        # Score axioms by keyword overlap with question
        scored: list[tuple[Axiom, int]] = []
        q_words = set(w.lower() for w in parsed.keywords)

        for ax in all_axioms:
            score = 0
            ax_words = set(
                w.lower()
                for w in ax.name.split()
                + ax.formula_str.replace("=", " ").split()
            )
            # Domain match bonus
            if ax.domain.split(".")[0] == domain:
                score += 3

            # Keyword overlap
            score += len(q_words & ax_words)

            scored.append((ax, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ax for ax, _ in scored[:5]]

    def _compute_supporting_values(
        self,
        parsed: ParsedQuestion,
        axioms: list[Axiom],
    ) -> dict[str, float]:
        """Execute computable axioms with available variables."""
        computed: dict[str, float] = {}
        if not parsed.variables:
            return computed

        for ax in axioms:
            if "=" not in ax.formula_str or "==" in ax.formula_str:
                continue
            lhs = ax.formula_str.split("=", 1)[0].strip()
            if not lhs.isidentifier():
                continue
            try:
                result = self._executor.execute(
                    ax.formula_str, parsed.variables,
                )
                computed[lhs] = result
            except Exception:
                pass

        return computed

    def _build_context(
        self,
        parsed: ParsedQuestion,
        domain: str,
        axioms: list[Axiom] | None = None,
        computed_values: dict[str, float] | None = None,
    ) -> str:
        """Build structured reasoning context for LLM prompt.

        Args:
            parsed: Parsed question structure.
            domain: Domain name.
            axioms: Relevant axioms (optional, will be found if None).
            computed_values: Pre-computed values (optional).

        Returns:
            Structured context string.
        """
        if axioms is None:
            axioms = self._find_relevant_axioms(parsed, domain)
        if computed_values is None:
            computed_values = {}

        lines: list[str] = []
        lines.append(f"Domain: {domain}")
        lines.append("")

        if axioms:
            lines.append("Relevant axioms:")
            for ax in axioms:
                lines.append(f"- {ax.name} ({ax.formula_str})")
            lines.append("")

        if computed_values:
            lines.append("Computed values:")
            for var, val in computed_values.items():
                lines.append(f"- {var} = {val}")
            lines.append("")

        lines.append(f"Question: {parsed.original}")

        return "\n".join(lines)

    def _build_llm_payload(
        self,
        context: str,
    ) -> dict[str, Any]:
        """Build the OpenRouter-compatible API payload.

        Returns:
            Dict with model, messages, temperature, max_tokens.
        """
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }

    def _call_llm(self, context: str) -> str | None:
        """Call LLM endpoint via httpx POST.

        Returns the generated text, or None if the call fails.
        """
        if httpx is None or self._llm_endpoint is None:
            return None

        payload = self._build_llm_payload(context)
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = httpx.post(
                self._llm_endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            if response.status_code != 200:
                return None

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return None

            message = choices[0].get("message", {})
            return message.get("content")

        except Exception:
            return None

    def _generate_answer(
        self,
        context: str,
        parsed: ParsedQuestion,
        domain: str,
        axioms: list[Axiom],
        computed_values: dict[str, float],
        trace: list[str],
    ) -> str:
        """Generate answer via LLM or template fallback."""
        # Try LLM first
        if self._llm_endpoint:
            trace.append("Attempting LLM call...")
            llm_answer = self._call_llm(context)
            if llm_answer:
                trace.append("LLM response received.")
                return llm_answer
            trace.append("LLM call failed, falling back to template.")

        # Template fallback
        return self._template_answer(
            parsed, domain, axioms, computed_values,
        )

    def _template_answer(
        self,
        parsed: ParsedQuestion,
        domain: str,
        axioms: list[Axiom],
        computed_values: dict[str, float],
    ) -> str:
        """Generate a template-based answer without LLM.

        Less fluent than LLM but always works.
        """
        if parsed.intent == QuestionIntent.EXPLAIN:
            return self._template_explain(axioms, computed_values)

        if parsed.intent == QuestionIntent.COMPARE:
            return self._template_compare(axioms)

        # GENERAL and other fallback
        return self._template_general(
            parsed, domain, axioms, computed_values,
        )

    def _template_explain(
        self,
        axioms: list[Axiom],
        computed_values: dict[str, float],
    ) -> str:
        """Template for EXPLAIN intent."""
        if not axioms:
            return "No relevant axioms found for this explanation."

        lines: list[str] = []
        primary = axioms[0]
        lines.append(f"Based on {primary.name} ({primary.formula_str}):")
        lines.append("")

        if computed_values:
            for var, val in computed_values.items():
                lines.append(f"  {var} = {val}")
            lines.append("")

        if len(axioms) > 1:
            lines.append("Related principles:")
            for ax in axioms[1:]:
                lines.append(f"  - {ax.name}: {ax.formula_str}")

        return "\n".join(lines)

    def _template_compare(self, axioms: list[Axiom]) -> str:
        """Template for COMPARE intent."""
        if not axioms:
            return "No relevant axioms found for comparison."

        lines = ["Comparison of relevant principles:"]
        for ax in axioms:
            lines.append(
                f"- {ax.name} ({ax.domain}): {ax.formula_str}"
            )
        return "\n".join(lines)

    def _template_general(
        self,
        parsed: ParsedQuestion,
        domain: str,
        axioms: list[Axiom],
        computed_values: dict[str, float],
    ) -> str:
        """Template for GENERAL intent and other fallbacks."""
        lines: list[str] = []

        if axioms:
            lines.append(f"In the domain of {domain}, relevant principles:")
            for ax in axioms[:3]:
                lines.append(f"- {ax.name}: {ax.formula_str}")

        if computed_values:
            lines.append("")
            lines.append("Supporting computations:")
            for var, val in computed_values.items():
                lines.append(f"  {var} = {val}")

        if not lines:
            lines.append(
                f"No specific knowledge found for '{parsed.original}' "
                f"in the {domain} domain."
            )

        return "\n".join(lines)
```

### Step 5.3 -- Run tests

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run pytest tests/test_augmented_solver.py -v --tb=short`
- [ ] Verify all tests pass

---

## Task 6: qa/__init__.py -- QAPipeline orchestrator + integration tests

**File:** `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\__init__.py`

Depends on: all previous QA modules + MoWMPipeline.

### Step 6.1 -- Write tests

- [ ] Create test file `c:\Users\grill\Documents\GitHub\mowm\tests\test_qa_pipeline.py`

```python
# tests/test_qa_pipeline.py
"""Tests for QAPipeline — end-to-end Q&A orchestrator."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa import QAPipeline, QAResponse

K, L = 4, 32


def _make_library() -> AxiomLibrary:
    lib = AxiomLibrary(k=K, l=L)
    # Physics
    for name, domain, formula in [
        ("Newton's Second Law", "physics.dynamics", "F = m * a"),
        ("Kinetic Energy", "physics.energy", "KE = 0.5 * m * v^2"),
        ("Ohm's Law", "physics.em", "V = I * R"),
    ]:
        ax = lib.make_axiom(name, domain, formula)
        lib.register(ax)
    # Logic
    ax = lib.make_axiom(
        "Modus Ponens", "logic.deductive",
        "(P AND (P IMPLIES Q)) IMPLIES Q",
    )
    lib.register(ax)
    return lib


def _make_causal_graph() -> CausalGraph:
    g = CausalGraph(k=K, l=L)
    rng = np.random.default_rng(42)
    for name in ("force", "acceleration", "velocity"):
        vec = rng.standard_normal((K, L)).astype(np.float32)
        g.add_node(name, "physics", vec)
    g.add_edge("force", "acceleration", "+", 0.9)
    g.add_edge("acceleration", "velocity", "+", 0.8)
    return g


def _make_pipeline() -> QAPipeline:
    lib = _make_library()
    executor = SafeFormulaExecutor()
    logic = LogicEngine()
    causal = _make_causal_graph()
    return QAPipeline(
        library=lib,
        executor=executor,
        logic=logic,
        causal=causal,
        k=K,
        l=L,
    )


class TestQAPipelineDirectMode:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_ask_compute_newton(self):
        response = self.pipeline.ask(
            "What is F when m=10 and a=9.8?"
        )
        assert isinstance(response, QAResponse)
        assert response.mode == "direct"
        assert response.computed_values.get("F") == pytest.approx(98.0)
        assert response.confidence > 0.5

    def test_ask_compute_kinetic_energy(self):
        response = self.pipeline.ask(
            "What is KE when m=2 and v=3?"
        )
        assert response.mode == "direct"
        assert response.computed_values.get("KE") == pytest.approx(9.0)

    def test_ask_evaluate_obligation(self):
        response = self.pipeline.ask(
            "Is the obligation valid if signed_contract and paid_deposit?",
            facts={"signed_contract": True, "paid_deposit": True},
        )
        assert response.mode == "direct"
        assert response.computed_values.get("result") is True

    def test_ask_causal(self):
        response = self.pipeline.ask(
            "What happens if force increases by 10?"
        )
        assert response.mode == "direct"
        assert "acceleration" in response.computed_values

    def test_ask_returns_evidence(self):
        response = self.pipeline.ask(
            "What is F when m=10 and a=9.8?"
        )
        assert len(response.evidence) > 0

    def test_ask_returns_reasoning_trace(self):
        response = self.pipeline.ask(
            "What is F when m=10 and a=9.8?"
        )
        assert len(response.reasoning_trace) > 0


class TestQAPipelineAugmentedMode:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_ask_explain_falls_back_to_augmented(self):
        response = self.pipeline.ask(
            "Why do heavier objects fall at the same rate?"
        )
        assert response.mode == "augmented"
        assert len(response.answer) > 0

    def test_ask_compare_falls_back_to_augmented(self):
        response = self.pipeline.ask(
            "Compare force and momentum"
        )
        assert response.mode == "augmented"

    def test_ask_general_falls_back_to_augmented(self):
        response = self.pipeline.ask(
            "Tell me about the universe"
        )
        assert response.mode == "augmented"


class TestQAPipelineBatch:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_ask_batch(self):
        questions = [
            "What is F when m=10 and a=9.8?",
            "Why does gravity exist?",
            "What happens if force increases by 10?",
        ]
        responses = self.pipeline.ask_batch(questions)
        assert len(responses) == 3
        assert all(isinstance(r, QAResponse) for r in responses)

    def test_ask_batch_empty(self):
        responses = self.pipeline.ask_batch([])
        assert responses == []

    def test_ask_batch_single(self):
        responses = self.pipeline.ask_batch(
            ["What is F when m=5 and a=2?"]
        )
        assert len(responses) == 1
        assert responses[0].computed_values.get("F") == pytest.approx(10.0)


class TestQAPipelineEdgeCases:
    def setup_method(self):
        self.pipeline = _make_pipeline()

    def test_empty_question(self):
        response = self.pipeline.ask("")
        assert isinstance(response, QAResponse)

    def test_nonsense_question(self):
        response = self.pipeline.ask("xyzzy frobulate quux blargh")
        assert isinstance(response, QAResponse)
        assert response.mode == "augmented"

    def test_response_has_domain(self):
        response = self.pipeline.ask(
            "What is F when m=10 and a=9.8?"
        )
        assert isinstance(response.domain, str)
        assert len(response.domain) > 0

    def test_confidence_is_bounded(self):
        response = self.pipeline.ask(
            "What is F when m=10 and a=9.8?"
        )
        assert 0.0 <= response.confidence <= 1.0


class TestQAResponseDataclass:
    def test_qa_response_fields(self):
        r = QAResponse(
            answer="F = 98.0 N",
            mode="direct",
            confidence=0.95,
            domain="physics",
            evidence=["Newton's Second Law: F = m * a"],
            computed_values={"F": 98.0},
            reasoning_trace=["Step 1"],
        )
        assert r.answer == "F = 98.0 N"
        assert r.mode == "direct"
        assert r.confidence == 0.95
        assert r.domain == "physics"
        assert len(r.evidence) == 1
        assert r.computed_values["F"] == 98.0

    def test_qa_response_defaults(self):
        r = QAResponse(
            answer="test",
            mode="direct",
            confidence=0.5,
            domain="general",
        )
        assert r.evidence == []
        assert r.computed_values == {}
        assert r.reasoning_trace == []
```

### Step 6.2 -- Implement qa/__init__.py

- [ ] Create `c:\Users\grill\Documents\GitHub\mowm\mowm\qa\__init__.py`

```python
# mowm/qa/__init__.py
"""MoWM Q&A Pipeline — answer questions using world model reasoning.

Public API:
    QAPipeline — Main orchestrator (ask, ask_batch)
    QAResponse — Structured response dataclass

Two modes:
    Direct: compute answers via formula execution, logic, or causal graphs.
    Augmented: package MoWM reasoning as context for LLM generation.

The system auto-selects: if the question maps to a computable axiom,
use direct mode. Otherwise, fall back to augmented mode. Template
fallback ensures the pipeline works without any LLM dependency.
"""
from __future__ import annotations

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa.augmented_solver import AugmentedSolver
from mowm.qa.classifier import DomainClassifier
from mowm.qa.direct_solver import DirectSolver, QAResponse
from mowm.qa.parser import QuestionParser

__all__ = ["QAPipeline", "QAResponse"]


class QAPipeline:
    """End-to-end Q&A pipeline using MoWM world model reasoning.

    Orchestrates: classify -> parse -> direct solve -> augmented solve.

    Args:
        library: AxiomLibrary with registered axioms.
        executor: SafeFormulaExecutor for formula evaluation.
        logic: LogicEngine for boolean/probabilistic evaluation.
        causal: Optional CausalGraph for causal propagation.
        k: Number of VSA blocks (default 80, use 4 for tests).
        l: Block length (default 128, use 32 for tests).
        codebook_path: Optional path to NV-Embed codebook .npz.
        llm_endpoint: Optional URL for LLM API.
        api_key: Optional API key for LLM endpoint.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
        logic: LogicEngine,
        causal: CausalGraph | None = None,
        k: int = 80,
        l: int = 128,  # noqa: E741
        codebook_path: str | None = None,
        llm_endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._library = library
        self._executor = executor
        self._logic = logic
        self._causal = causal

        self._classifier = DomainClassifier(
            library=library,
            k=k,
            l=l,
            codebook_path=codebook_path,
        )
        self._parser = QuestionParser()
        self._direct_solver = DirectSolver(
            library=library,
            executor=executor,
            logic=logic,
            causal=causal,
        )
        self._augmented_solver = AugmentedSolver(
            library=library,
            executor=executor,
            llm_endpoint=llm_endpoint,
            api_key=api_key,
        )

    def ask(
        self,
        question: str,
        facts: dict[str, bool] | None = None,
    ) -> QAResponse:
        """Answer a question using the full MoWM reasoning pipeline.

        Flow:
          1. Classify domain (cosine sim + keywords)
          2. Parse question structure (intent, variables, target)
          3. Try direct solve first (executor/logic/causal)
          4. Fall back to augmented solve if direct can't handle it
          5. Return QAResponse with answer + evidence + reasoning trace

        Args:
            question: Natural language question string.
            facts: Optional boolean facts for logic evaluation.

        Returns:
            QAResponse with answer, confidence, evidence, and trace.
        """
        # Step 1: Classify domain
        domain = self._classifier.top_domain(question)

        # Step 2: Parse question
        parsed = self._parser.parse(question)

        # Step 3: Try direct solve
        if self._direct_solver.can_solve(parsed, domain):
            response = self._direct_solver.solve(
                parsed, domain, facts=facts,
            )
            if response.confidence > 0.0:
                return response

        # Step 4: Fall back to augmented solve
        # Pass any partial direct results as context
        direct_partial = None
        if self._direct_solver.can_solve(parsed, domain):
            try:
                direct_partial = self._direct_solver.solve(
                    parsed, domain, facts=facts,
                )
            except Exception:
                pass

        response = self._augmented_solver.solve(
            parsed, domain, direct_result=direct_partial,
        )
        return response

    def ask_batch(
        self,
        questions: list[str],
        facts: dict[str, bool] | None = None,
    ) -> list[QAResponse]:
        """Answer multiple questions.

        Args:
            questions: List of natural language question strings.
            facts: Optional shared boolean facts for logic evaluation.

        Returns:
            List of QAResponse, one per question.
        """
        return [self.ask(q, facts=facts) for q in questions]
```

### Step 6.3 -- Update mowm/__init__.py to export QA pipeline

- [ ] Edit `c:\Users\grill\Documents\GitHub\mowm\mowm\__init__.py`

Add the following import and update `__all__`:

```python
# After existing imports, add:
from mowm.qa import QAPipeline, QAResponse

# Update __version__:
__version__ = "0.3.0"

# Add to __all__:
#   "QAPipeline",
#   "QAResponse",
```

The full updated `__init__.py` should be:

```python
# mowm/__init__.py
"""MoWM — General-Purpose Mixture of World Models.

Self-organizing recursive expert worlds with a living axiom library,
14 domain modules, discovery loop via symbolic regression, and
dynamic expert spawn/prune.

Public API (v0.3.0):
    MoWMPipeline — End-to-end pipeline orchestrator
    AxiomLibrary — Global world axiom storage
    Axiom — Single axiom dataclass
    World — Recursive expert world model
    MoWMRouter — Dynamic expert routing
    DiscoveryLoop — Symbolic regression discovery
    SafeFormulaExecutor — Safe formula evaluation executor
    CausalGraph — VSA-backed causal graph
    LogicEngine — Dual-mode deductive/probabilistic logic
    SequenceEncoder — DNA / temporal / token sequence encoder
    PhasorEngine — Harmonic phasor bank for frequency-domain encoding
    IdentityState — Personal world model with Plutchik emotions
    GridWorld — 2-D simulation environment
    MoWMTrainer — Training loop with prediction + confidence loss
    QAPipeline — Natural language Q&A over world model reasoning
    QAResponse — Structured Q&A response dataclass
"""

__version__ = "0.3.0"

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.discovery import DiscoveryLoop
from mowm.executor import SafeFormulaExecutor
from mowm.identity_state import IdentityState
from mowm.logic_engine import LogicEngine
from mowm.phasor_engine import PhasorEngine
from mowm.pipeline import MoWMPipeline, StepResult
from mowm.qa import QAPipeline, QAResponse
from mowm.router import MoWMRouter
from mowm.sequence_encoder import SequenceEncoder
from mowm.sim_environment import GridWorld
from mowm.training import MoWMTrainer
from mowm.world import World

__all__ = [
    "Axiom",
    "AxiomLibrary",
    "CausalGraph",
    "DiscoveryLoop",
    "GridWorld",
    "IdentityState",
    "LogicEngine",
    "MoWMPipeline",
    "MoWMRouter",
    "MoWMTrainer",
    "PhasorEngine",
    "QAPipeline",
    "QAResponse",
    "SafeFormulaExecutor",
    "SequenceEncoder",
    "StepResult",
    "World",
]
```

### Step 6.4 -- Run all tests

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run pytest tests/test_formatter.py tests/test_parser.py tests/test_classifier.py tests/test_direct_solver.py tests/test_augmented_solver.py tests/test_qa_pipeline.py -v --tb=short`
- [ ] Verify all new tests pass

### Step 6.5 -- Run full test suite

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run pytest tests/ -v -q --tb=short -x`
- [ ] Verify all 381 existing tests still pass plus all new tests

### Step 6.6 -- Lint

- [ ] Run: `cd c:\Users\grill\Documents\GitHub\mowm && uv run ruff check mowm/qa/ tests/test_formatter.py tests/test_parser.py tests/test_classifier.py tests/test_direct_solver.py tests/test_augmented_solver.py tests/test_qa_pipeline.py`
- [ ] Fix any lint errors

---

## Summary

| Task | File | Lines (est.) | Tests (est.) | Depends on |
|------|------|-------------|-------------|------------|
| 1 | `mowm/qa/formatter.py` | ~140 | ~18 | None |
| 2 | `mowm/qa/parser.py` | ~250 | ~26 | None |
| 3 | `mowm/qa/classifier.py` | ~160 | ~12 | AxiomLibrary, WorldEncoder |
| 4 | `mowm/qa/direct_solver.py` | ~270 | ~18 | Executor, Logic, Causal, Parser, Formatter |
| 5 | `mowm/qa/augmented_solver.py` | ~300 | ~14 | AxiomLibrary, Executor, Formatter, httpx |
| 6 | `mowm/qa/__init__.py` | ~120 | ~20 | All QA modules |
| **Total** | **6 files** | **~1240** | **~108** | |

### New files created:
```
mowm/qa/__init__.py
mowm/qa/formatter.py
mowm/qa/parser.py
mowm/qa/classifier.py
mowm/qa/direct_solver.py
mowm/qa/augmented_solver.py
tests/test_formatter.py
tests/test_parser.py
tests/test_classifier.py
tests/test_direct_solver.py
tests/test_augmented_solver.py
tests/test_qa_pipeline.py
```

### Files modified:
```
mowm/__init__.py  (add QAPipeline, QAResponse exports, bump to v0.3.0)
```
