# MoWM v0.3.0 Design Spec — Q&A Pipeline

**Date:** 2026-03-27
**Author:** Nicolas Cloutier + Claude
**Status:** Draft
**Module:** `mowm/` (extends v0.2.0)
**Depends on:** v0.2.0 (executor, causal_graph, logic_engine, sequence_encoder, identity_state, pipeline)

## Overview

v0.2.0 can reason but can't talk. v0.3.0 adds a Q&A pipeline that takes
a natural language question, routes it through the world model, computes
an answer using the appropriate domain expert, and returns a natural
language response.

Two modes:
- **Direct mode (B):** MoWM computes the answer itself via formula execution,
  logic evaluation, or causal reasoning. Fast, interpretable, no LLM needed.
- **Augmented mode (A):** MoWM reasons first, then packages its findings as
  structured context for an LLM (Qwen2.5 via grilly or API) to generate
  a fluent natural language response.

The system auto-selects: if the question maps to a computable axiom,
use direct mode. Otherwise, fall back to augmented mode.

## Architecture

```
Question (text)
    |
    v
NV-Embed-v2 encode (via codebook projection)
    |
    v
Domain Classifier (cosine sim to domain centroids)
    |
    v
Question Parser (extract variables, intent)
    |
    +--> Direct Mode                  +--> Augmented Mode
    |    (computable?)                |    (open-ended)
    |                                 |
    v                                 v
AxiomExecutor                    MoWM Reasoning Context
  + LogicEngine                    (selected axioms,
  + CausalGraph                     causal effects,
    |                                formula results)
    v                                 |
Structured Answer                     v
  {"value": 98.0,                LLM Generate
   "unit": "N",                    (Qwen2.5 or API)
   "formula": "F=m*a",              |
   "confidence": 0.95}              v
    |                           Natural Language Answer
    v
Format as text
    |
    v
QAResponse (text + evidence + confidence)
```

## Module Structure (new files only)

```
mowm/
    qa/
        __init__.py           # Public: QAPipeline, QAResponse
        classifier.py         # DomainClassifier — cosine sim to NV-Embed centroids
        parser.py             # QuestionParser — extract intent, variables, constraints
        direct_solver.py      # DirectSolver — compute answers via executor/logic/causal
        augmented_solver.py   # AugmentedSolver — package context for LLM generation
        formatter.py          # ResponseFormatter — structured answer -> natural text
    tests/
        test_classifier.py
        test_parser.py
        test_direct_solver.py
        test_augmented_solver.py
        test_qa_pipeline.py
```

## Component Design

### 1. QAResponse (dataclass)

```python
@dataclass
class QAResponse:
    answer: str                    # Natural language answer
    mode: Literal["direct", "augmented"]  # Which mode produced it
    confidence: float              # [0, 1]
    domain: str                    # Matched domain
    evidence: list[str]            # List of axioms/formulas used
    computed_values: dict          # {"F": 98.0, "unit": "N"} if direct
    reasoning_trace: list[str]     # Step-by-step reasoning
```

### 2. DomainClassifier (`classifier.py`)

Routes a question to the most relevant domain using NV-Embed-v2
codebook centroids projected into VSA space.

```python
class DomainClassifier:
    def __init__(self, codebook_path: str, k: int = 80, l: int = 128):
        # Load NV-Embed centroids + projection matrix from codebook
        # Project all domain centroids into VSA space

    def classify(self, question: str) -> list[tuple[str, float]]:
        """Return ranked list of (domain, score) pairs.

        Encodes question via WorldEncoder, computes similarity
        against all domain centroid VSA vectors.
        """

    def classify_embedding(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Classify from a pre-computed NV-Embed embedding."""

    def top_domain(self, question: str) -> str:
        """Return single best-matching domain name."""
```

If NV-Embed codebook is not available, falls back to keyword-based
classification using axiom name matching from the AxiomLibrary.

### 3. QuestionParser (`parser.py`)

Extracts structured information from a natural language question.

```python
class QuestionIntent(Enum):
    COMPUTE = "compute"        # "What is F when m=10, a=9.8?"
    EVALUATE = "evaluate"      # "Is the contract valid?"
    CAUSAL = "causal"          # "What happens if X increases?"
    EXPLAIN = "explain"        # "Why does F=ma?"
    COMPARE = "compare"        # "Compare X and Y"
    GENERAL = "general"        # Anything else

@dataclass
class ParsedQuestion:
    original: str
    intent: QuestionIntent
    variables: dict[str, float]   # {"m": 10.0, "a": 9.8}
    target_var: str | None        # "F" (what to solve for)
    conditions: list[str]         # ["signed_contract", "paid_deposit"]
    causal_node: str | None       # "force" (for causal queries)
    causal_delta: float | None    # +10.0 (change magnitude)
    keywords: list[str]           # Extracted domain keywords

class QuestionParser:
    def parse(self, question: str) -> ParsedQuestion:
        """Parse a question into structured components.

        Uses regex patterns + keyword extraction:
          - "What is X when Y=N, Z=M?" -> COMPUTE, vars={Y:N, Z:M}, target=X
          - "Is X true if A and B?" -> EVALUATE, conditions=[A, B]
          - "What happens if X increases by N?" -> CAUSAL, node=X, delta=N
          - "Why/How/Explain X" -> EXPLAIN
          - "Compare X and Y" -> COMPARE
        """
```

### 4. DirectSolver (`direct_solver.py`)

Computes answers without an LLM when the question maps to a known axiom.

```python
class DirectSolver:
    def __init__(self, library: AxiomLibrary, executor: SafeFormulaExecutor,
                 logic: LogicEngine, causal: CausalGraph | None = None):

    def can_solve(self, parsed: ParsedQuestion, domain: str) -> bool:
        """Check if this question can be solved directly.

        True if:
          - COMPUTE intent + matching axiom formula + all variables provided
          - EVALUATE intent + logic formula + all conditions known
          - CAUSAL intent + causal graph has the node
        """

    def solve(self, parsed: ParsedQuestion, domain: str) -> QAResponse:
        """Solve the question directly.

        COMPUTE: find matching axiom -> executor.execute(formula, vars)
        EVALUATE: logic_engine.evaluate(formula, facts)
        CAUSAL: causal_graph.propagate(node, delta)
        """

    def _find_matching_axiom(self, parsed: ParsedQuestion, domain: str) -> Axiom | None:
        """Find an axiom whose formula contains the target variable."""

    def _format_compute_answer(self, axiom: Axiom, result: float,
                                parsed: ParsedQuestion) -> QAResponse:
        """Format: 'F = 98.0 N (from Newton's Second Law: F = m * a)'"""
```

### 5. AugmentedSolver (`augmented_solver.py`)

Packages MoWM reasoning as context for LLM generation.

```python
class AugmentedSolver:
    def __init__(self, library: AxiomLibrary, pipeline: MoWMPipeline,
                 llm_endpoint: str | None = None):

    def solve(self, parsed: ParsedQuestion, domain: str) -> QAResponse:
        """Generate answer via LLM with MoWM reasoning context.

        1. Select relevant axioms from library (top 5 by similarity)
        2. Run pipeline.step() to get world model prediction
        3. Execute any computable axioms for supporting evidence
        4. Package as structured prompt:
           "Domain: physics
            Relevant axioms: [Newton's Second Law: F=m*a, ...]
            Computed values: {F: 98.0}
            Causal effects: {accel: +9.0, velocity: +7.2}
            Question: {original question}
            Answer:"
        5. Send to LLM endpoint for generation
        """

    def _build_context(self, parsed: ParsedQuestion, domain: str) -> str:
        """Build structured reasoning context for LLM."""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM endpoint (local grilly or API).

        Supports:
          - Local: grilly inference server
          - API: OpenRouter / Anthropic / OpenAI
          - Fallback: template-based response (no LLM)
        """
```

When no LLM is available, AugmentedSolver falls back to a template-based
response using the reasoning context directly — still useful, just less
fluent.

### 6. ResponseFormatter (`formatter.py`)

Converts structured answers into clean natural language.

```python
class ResponseFormatter:
    def format_compute(self, axiom_name: str, formula: str,
                       variables: dict, result: float, unit: str = "") -> str:
        """'Using Newton's Second Law (F = m * a):
            With m = 10.0 kg, a = 9.8 m/s²:
            F = 98.0 N'"""

    def format_evaluate(self, conditions: list[str], result: bool,
                        exceptions: list[str] | None = None) -> str:
        """'The obligation holds because: signed_contract ✓, paid_deposit ✓.
            No exceptions apply.'"""

    def format_causal(self, node: str, delta: float,
                      effects: dict[str, float]) -> str:
        """'If force increases by 10.0:
            → acceleration increases by 9.0 (+)
            → velocity increases by 7.2 (+)'"""

    def format_explain(self, axiom: Axiom, context: str) -> str:
        """Format explanation with axiom details and LLM context."""
```

### 7. QAPipeline (`qa/__init__.py`)

The main orchestrator.

```python
class QAPipeline:
    def __init__(self, mowm_pipeline: MoWMPipeline,
                 codebook_path: str | None = None,
                 llm_endpoint: str | None = None):
        # Sets up classifier, parser, direct solver, augmented solver

    def ask(self, question: str) -> QAResponse:
        """Answer a question using the full MoWM reasoning pipeline.

        1. Classify domain
        2. Parse question structure
        3. Try direct solve first
        4. Fall back to augmented solve if needed
        5. Return QAResponse with answer + evidence + reasoning trace
        """

    def ask_batch(self, questions: list[str]) -> list[QAResponse]:
        """Answer multiple questions."""
```

## Example Interactions

### Direct mode — Physics computation
```
Q: "What is the force on a 10kg object accelerating at 9.8 m/s²?"
→ Domain: physics (0.94)
→ Intent: COMPUTE, vars={m:10.0, a:9.8}, target=F
→ Axiom: Newton's Second Law (F = m * a)
→ Result: 98.0

A: "Using Newton's Second Law (F = m × a):
    With m = 10.0 kg, a = 9.8 m/s²:
    F = 98.0 N"
    [confidence: 0.98, mode: direct]
```

### Direct mode — Legal evaluation
```
Q: "Is there a legal obligation if the contract is signed but deposit not paid?"
→ Domain: logic (0.87)
→ Intent: EVALUATE, conditions=[signed_contract, paid_deposit]
→ Facts: {signed_contract: True, paid_deposit: False}
→ check_obligation → False

A: "No, the obligation does not hold.
    Required conditions: signed_contract ✓, paid_deposit ✗
    Missing: paid_deposit"
    [confidence: 0.95, mode: direct]
```

### Direct mode — Causal reasoning
```
Q: "What happens if ad spend increases by 1000?"
→ Domain: causal (0.82)
→ Intent: CAUSAL, node=ad_spend, delta=+1000
→ propagate → {traffic: +800, sales: +480, stock: -288}

A: "If ad_spend increases by 1000.0:
    → traffic increases by 800.0 (+)
    → sales increases by 480.0 (+)
    → stock decreases by 288.0 (−)"
    [confidence: 0.85, mode: direct]
```

### Augmented mode — Open-ended explanation
```
Q: "Why do heavier objects fall at the same rate as lighter ones?"
→ Domain: physics (0.91)
→ Intent: EXPLAIN
→ Can't solve directly (no specific computation)
→ Relevant axioms: [Gravitational Force, Newton's Second Law, ...]
→ Context sent to LLM

A: "All objects fall at the same rate because gravitational force
    is proportional to mass (F = mg), and acceleration equals force
    divided by mass (a = F/m). The mass cancels: a = mg/m = g.
    This means acceleration due to gravity (g ≈ 9.8 m/s²) is
    independent of mass."
    [confidence: 0.80, mode: augmented, evidence: [Newton's Second Law, ...]]
```

## LLM Integration

AugmentedSolver supports three backends:

1. **Local grilly server** — if grilly inference is running locally
   (endpoint: `http://localhost:8000/v1/completions`)
2. **OpenRouter API** — for cloud models (Qwen2.5, Claude, etc.)
   (requires `OPENROUTER_API_KEY` env var)
3. **Template fallback** — if no LLM available, uses the structured
   context directly with fill-in-the-blank templates

The LLM receives a structured prompt with all MoWM reasoning pre-computed:
```
System: You are a precise reasoning assistant. Use the provided domain
knowledge and computed values to answer the question. Cite specific
formulas and values.

Context:
- Domain: physics
- Relevant axioms: Newton's Second Law (F = m * a), Kinetic Energy (KE = 0.5 * m * v²)
- Computed values: F = 98.0 N
- Causal effects: acceleration → velocity (+)

Question: {question}
Answer:
```

## Test Strategy

Tests use k=4, l=32 as before. LLM tests mock the endpoint.

| Test | What it validates |
|------|-------------------|
| `test_classifier.py` | Domain classification (cosine sim), top_domain, fallback |
| `test_parser.py` | Parse COMPUTE/EVALUATE/CAUSAL/EXPLAIN intents, variable extraction |
| `test_direct_solver.py` | Physics computation, logic evaluation, causal propagation |
| `test_augmented_solver.py` | Context building, LLM prompt format, template fallback |
| `test_qa_pipeline.py` | End-to-end: ask() routes correctly, batch, edge cases |

## Key Design Decisions

1. **Direct-first, augment-second.** Always try to compute the answer
   directly. LLM is a fallback for open-ended questions, not the default.
   This makes MoWM interpretable and fast for formulaic questions.

2. **No LLM dependency for core Q&A.** The template fallback means MoWM
   can answer questions without any LLM running. Quality is lower but
   it always works.

3. **Evidence trail in every response.** QAResponse includes the axioms
   used, formulas executed, and reasoning steps. Full transparency.

4. **QuestionParser is regex-based, not ML.** Simple, fast, deterministic.
   Handles the structured patterns we need (compute/evaluate/causal).
   ML-based NLU can be added later as a parser strategy (GoF Strategy).
