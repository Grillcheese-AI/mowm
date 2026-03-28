# MoWM v0.4.0 Design Spec — Reasoning Decoder (Be The LLM)

**Date:** 2026-03-28
**Author:** Nicolas Cloutier + Claude
**Status:** Draft
**Module:** `mowm/` (extends v0.3.0)
**Depends on:** v0.3.0 (QAPipeline, DirectSolver, AxiomLibrary, CausalGraph, LogicEngine, Executor)

## Overview

v0.3.0 can answer formulaic questions directly (F=98.0N) but punts
open-ended questions (EXPLAIN/COMPARE/GENERAL) to an external LLM.
v0.4.0 removes that dependency entirely: MoWM generates its own
natural language responses by reasoning through its axiom graph.

**The core insight:** MoWM already has everything it needs to explain,
compare, and generalize — it just doesn't know how to *compose text*
from what it knows. The reasoning decoder walks the axiom graph,
traces causal chains, executes supporting formulas, and composes
a coherent response from structured templates that grow in complexity
with the reasoning depth.

This is NOT a language model. It's a **knowledge compiler** — it turns
structured world model reasoning into readable text the same way a
textbook turns equations into explanations.

## What v0.4.0 Replaces

| Question Type | v0.3.0 (with LLM) | v0.4.0 (self-contained) |
|---|---|---|
| COMPUTE | Direct: F = 98.0 N | Same (already works) |
| EVALUATE | Direct: obligation holds/fails | Same (already works) |
| CAUSAL | Direct: propagation effects | Same (already works) |
| EXPLAIN | Template or LLM | **Reasoning decoder**: axiom chain + causal trace |
| COMPARE | Template or LLM | **Reasoning decoder**: dual execution + structural diff |
| GENERAL | Template or LLM | **Reasoning decoder**: knowledge graph walk + composition |

## Architecture

```
Question: "Why do heavier objects fall at the same rate?"
    |
    v
QuestionParser -> EXPLAIN intent, keywords=[heavier, objects, fall, rate]
    |
    v
DomainClassifier -> physics (0.92)
    |
    v
ReasoningDecoder:
  1. Knowledge Search
     - Find axioms: Gravitational Force (F=Gm1m2/r^2), Newton's 2nd (F=ma)
     - Find related: Weight (W=mg), free fall acceleration
  2. Axiom Chain Discovery
     - F_gravity = m*g  (gravitational force on mass m)
     - a = F/m = m*g/m = g  (mass cancels!)
     - Therefore: acceleration is independent of mass
  3. Causal Trace
     - mass -> gravitational_force (+)
     - mass -> inertia (+)
     - gravitational_force -> acceleration (+)
     - inertia -> acceleration (-)
     - Net: mass cancels out
  4. Response Composition
     - Intro: state the principle
     - Chain: walk through the axiom derivation
     - Conclusion: answer the "why"
    |
    v
"Objects fall at the same rate regardless of mass because gravitational
 force is proportional to mass (F = m*g), and acceleration equals force
 divided by mass (a = F/m). When you divide, mass cancels out: a = g.
 So all objects accelerate at g = 9.8 m/s^2 regardless of their mass.

 Axioms used:
   - Newton's Second Law: F = m * a
   - Weight: W = m * g
   - Gravitational acceleration: a = F/m = mg/m = g"
```

## Module Structure

```
mowm/
    reasoning/
        __init__.py           # Public: ReasoningDecoder
        knowledge_search.py   # Find relevant axioms + concepts
        chain_builder.py      # Build axiom derivation chains
        causal_tracer.py      # Trace causal graph for explanations
        comparator.py         # Dual-execute and diff for COMPARE
        composer.py           # Compose natural language from reasoning
    tests/
        test_knowledge_search.py
        test_chain_builder.py
        test_causal_tracer.py
        test_comparator.py
        test_composer.py
        test_reasoning_decoder.py
```

## Component Design

### 1. KnowledgeSearch (`knowledge_search.py`)

Finds relevant axioms, related concepts, and supporting knowledge
for a question. Goes beyond simple keyword matching — traces
connections through the axiom library via VSA similarity.

```python
@dataclass
class KnowledgeContext:
    primary_axioms: list[Axiom]    # Most directly relevant (top 3-5)
    supporting_axioms: list[Axiom] # Related by domain/similarity (next 5-10)
    concepts: list[str]            # Extracted concept names
    domain_chain: list[str]        # e.g. ["physics.dynamics", "physics.energy"]
    variables_mentioned: set[str]  # All variables found in relevant formulas

class KnowledgeSearch:
    def __init__(self, library: AxiomLibrary, k: int, l: int)

    def search(self, keywords: list[str], domain: str,
               target_var: str | None = None) -> KnowledgeContext:
        """Multi-hop knowledge search through the axiom library.

        1. Direct keyword match -> primary axioms
        2. VSA similarity from primary axiom vectors -> supporting axioms
        3. Domain chain: follow sub-domain connections
        4. Variable tracing: find all axioms sharing variables with primaries
        """

    def find_derivation_path(self, start_axiom: Axiom,
                              target_concept: str) -> list[Axiom]:
        """Find a chain of axioms connecting start to target concept.

        Uses shared variables as edges: if axiom A produces variable X
        and axiom B consumes variable X, there's a derivation link A -> B.
        """
```

### 2. ChainBuilder (`chain_builder.py`)

Builds step-by-step derivation chains from axioms. This is the core
"reasoning" — connecting axioms through shared variables to derive
a conclusion.

```python
@dataclass
class DerivationStep:
    axiom: Axiom                  # The axiom applied
    substitutions: dict[str, float | str]  # Variable values at this step
    result_var: str               # What this step derives
    result_value: float | str     # The derived value/expression
    explanation: str              # Human-readable step explanation

@dataclass
class DerivationChain:
    steps: list[DerivationStep]
    conclusion: str               # Final derived statement
    variables_used: set[str]
    axioms_used: list[str]        # Axiom names in order

class ChainBuilder:
    def __init__(self, executor: SafeFormulaExecutor, library: AxiomLibrary)

    def build_chain(self, context: KnowledgeContext,
                     variables: dict[str, float],
                     target: str | None = None) -> DerivationChain:
        """Build a derivation chain from knowledge context.

        Strategy:
          1. Start with axioms containing provided variables
          2. Execute each, producing new variable values
          3. Feed results into next axiom (variable propagation)
          4. Continue until target is derived or no more steps
        """

    def build_explanation_chain(self, context: KnowledgeContext,
                                 question_keywords: list[str]) -> DerivationChain:
        """Build a chain optimized for explanation (not computation).

        For EXPLAIN intent: finds the axiom sequence that best
        explains the concept, even without numeric values.
        Uses symbolic substitution: "a = F/m = mg/m = g" style.
        """

    def simplify_chain(self, chain: DerivationChain) -> DerivationChain:
        """Remove redundant steps, combine trivial substitutions."""
```

### 3. CausalTracer (`causal_tracer.py`)

Uses the CausalGraph to trace cause-effect relationships for
explanations. Produces a narrative of how variables influence
each other.

```python
@dataclass
class CausalNarrative:
    cause: str                    # Starting cause
    effects: list[tuple[str, str, float]]  # (target, direction, strength)
    cancellations: list[tuple[str, str]]   # Variables that cancel out
    net_effect: str               # Summary of net result
    graph_path: list[str]         # Node sequence in causal graph

class CausalTracer:
    def __init__(self, causal: CausalGraph)

    def trace_explanation(self, question_keywords: list[str],
                           domain: str) -> CausalNarrative | None:
        """Trace causal relationships relevant to the question.

        Identifies:
          - Positive feedback loops (reinforcing)
          - Negative feedback loops (balancing)
          - Variable cancellations (the key to many "why" answers)
          - Net direction of effect
        """

    def find_cancellation(self, var_a: str, var_b: str) -> str | None:
        """Check if two paths through var_a and var_b cancel.

        This is how MoWM explains "why heavier objects fall at same rate":
        mass increases gravitational force (+) but also increases inertia (+),
        and these cancel in the acceleration equation.
        """
```

### 4. Comparator (`comparator.py`)

For COMPARE intent: executes both sides, finds structural differences
in the axioms, and composes a comparison.

```python
@dataclass
class ComparisonResult:
    items: list[dict]             # [{name, axiom, value, domain}, ...]
    similarities: list[str]       # Shared properties
    differences: list[str]        # Distinguishing properties
    structural_diff: str          # How the formulas differ structurally

class Comparator:
    def __init__(self, library: AxiomLibrary, executor: SafeFormulaExecutor)

    def compare(self, item_names: list[str], domain: str,
                 variables: dict[str, float] | None = None) -> ComparisonResult:
        """Compare two or more concepts by finding their axioms,
        executing formulas with shared variables, and analyzing
        structural differences.

        "Compare force and momentum":
          - Force: F = m*a (instantaneous)
          - Momentum: p = m*v (accumulated)
          - Shared: both depend on mass
          - Difference: force depends on acceleration, momentum on velocity
          - Structural: force is rate of change of momentum (F = dp/dt)
        """
```

### 5. Composer (`composer.py`)

The text generation engine. Composes natural language from structured
reasoning results. Uses template-based generation with increasing
complexity based on reasoning depth.

```python
class Composer:
    """Compose natural language from structured reasoning.

    NOT a language model. Uses hierarchical templates:
      Level 1: Direct statement (1 sentence)
      Level 2: Statement + evidence (2-3 sentences)
      Level 3: Statement + derivation chain (paragraph)
      Level 4: Full explanation with intro/chain/conclusion

    The level is chosen based on reasoning depth — simple questions
    get simple answers, complex reasoning gets detailed explanations.
    """

    def compose_explanation(self, chain: DerivationChain,
                             causal: CausalNarrative | None,
                             question: str) -> str:
        """Compose a full explanation from derivation chain + causal trace.

        Structure:
          1. Opening: restate the core principle
          2. Derivation: walk through axiom chain step by step
          3. Causal context: if available, explain cause-effect
          4. Conclusion: direct answer to the "why/how"
          5. Evidence: list axioms used
        """

    def compose_comparison(self, result: ComparisonResult,
                            question: str) -> str:
        """Compose comparison text from structured diff."""

    def compose_general(self, context: KnowledgeContext,
                         chain: DerivationChain | None,
                         question: str) -> str:
        """Compose general knowledge response from axiom library."""

    def _select_level(self, chain: DerivationChain | None,
                       causal: CausalNarrative | None) -> int:
        """Choose template complexity level (1-4)."""
```

### 6. ReasoningDecoder (`reasoning/__init__.py`)

The orchestrator that replaces AugmentedSolver for EXPLAIN/COMPARE/GENERAL.

```python
class ReasoningDecoder:
    def __init__(self, library: AxiomLibrary, executor: SafeFormulaExecutor,
                 logic: LogicEngine, causal: CausalGraph | None,
                 k: int, l: int)

    def decode(self, parsed: ParsedQuestion, domain: str,
                facts: dict | None = None) -> QAResponse:
        """Generate a self-contained response using world model reasoning.

        Flow for EXPLAIN:
          1. KnowledgeSearch -> find relevant axioms + concepts
          2. ChainBuilder -> build derivation chain
          3. CausalTracer -> trace causal relationships
          4. Composer -> compose natural language explanation

        Flow for COMPARE:
          1. KnowledgeSearch -> find axioms for each item
          2. Comparator -> execute + structural diff
          3. Composer -> compose comparison text

        Flow for GENERAL:
          1. KnowledgeSearch -> broad axiom search
          2. ChainBuilder -> optional derivation if variables present
          3. Composer -> compose from knowledge context
        """
```

## Example Outputs

### EXPLAIN — "Why do heavier objects fall at the same rate?"

```
All objects in free fall accelerate at the same rate regardless of mass.

Derivation:
  1. Gravitational force on an object: W = m * g  (Weight)
  2. Newton's Second Law: F = m * a
  3. Setting equal: m * a = m * g
  4. Dividing both sides by mass: a = g
  5. Mass cancels — acceleration equals g for all objects.

The key insight is that mass appears on both sides of the equation:
it increases the gravitational pull (W = m*g) but equally increases
the resistance to acceleration (F = m*a). These two effects cancel
exactly, leaving acceleration = g = 9.8 m/s^2 for all masses.

Axioms used:
  - Weight: W = m * g
  - Newton's Second Law: F = m * a
```

### COMPARE — "Compare force and momentum"

```
Force and momentum are both fundamental concepts in dynamics that
depend on mass, but they describe different aspects of motion.

Force (F = m * a):
  - Measures instantaneous push/pull
  - Depends on mass and acceleration
  - Units: Newtons (kg*m/s^2)

Momentum (p = m * v):
  - Measures accumulated motion
  - Depends on mass and velocity
  - Units: kg*m/s

Key relationship: Force is the rate of change of momentum (F = dp/dt).
Applying a force over time changes an object's momentum (impulse J = F*dt).

Shared: both proportional to mass
Different: force uses acceleration (rate of change), momentum uses velocity

Axioms used:
  - Newton's Second Law: F = m * a
  - Momentum: p = m * v
  - Impulse: J = F * dt = dp
```

### GENERAL — "Tell me about entropy"

```
Entropy is a measure of disorder or the number of possible
microstates of a system (physics.thermo domain).

Key formulas:
  - Entropy Change: dS = dQ / T
    (entropy increases when heat is added at temperature T)
  - Boltzmann Entropy: S = k * ln(W)
    (entropy is proportional to the logarithm of microstates)
  - Clausius Inequality: dS >= dQ / T
    (entropy of an isolated system never decreases)

Related concepts:
  - Second Law of Thermodynamics: entropy of the universe always increases
  - Carnot Efficiency: eta = 1 - T_cold / T_hot
    (maximum efficiency limited by entropy considerations)

Axioms used:
  - Entropy Change: dS = dQ / T
  - Boltzmann Distribution: P(E) = exp(-E / (k_B * T)) / Z
  - Clausius Inequality: dS >= dQ / T
  - Carnot Efficiency: eta = 1 - T_cold / T_hot
```

## Integration with QAPipeline

The ReasoningDecoder replaces AugmentedSolver as the default handler
for non-computable questions. The LLM augmented solver becomes a
true last-resort fallback (e.g., for questions completely outside
the axiom library).

```python
# In QAPipeline.ask():
if direct_solver.can_solve(parsed, domain):
    return direct_solver.solve(parsed, domain)

# v0.4.0: use reasoning decoder instead of augmented solver
response = reasoning_decoder.decode(parsed, domain)
if response.confidence > 0.3:
    return response

# True last resort: augmented solver (LLM or template)
return augmented_solver.solve(parsed, domain)
```

## Key Design Decisions

1. **Not a language model.** The composer uses hierarchical templates
   with structured slots, not free-form generation. Quality comes from
   reasoning depth, not fluency.

2. **Derivation chains are the core innovation.** Connecting axioms
   through shared variables to derive new knowledge — this is how
   textbooks work, and it's fully deterministic.

3. **Variable cancellation detection** is what makes "why" answers
   work. Most physics "why" questions have the answer "because X
   cancels out in the equation."

4. **Complexity scales with reasoning depth.** Simple questions get
   1-sentence answers. Deep derivations get multi-paragraph explanations
   with step-by-step chains.

5. **Transparent — every claim is backed by a named axiom.** No
   hallucination possible because every statement traces to a formula.

## Test Strategy

| Test | What it validates |
|------|-------------------|
| `test_knowledge_search.py` | Multi-hop search, derivation path, variable tracing |
| `test_chain_builder.py` | Build chains, symbolic substitution, simplification |
| `test_causal_tracer.py` | Trace explanations, cancellation detection |
| `test_comparator.py` | Dual execute, structural diff, similarity/difference |
| `test_composer.py` | Level selection, explanation composition, comparison composition |
| `test_reasoning_decoder.py` | End-to-end: EXPLAIN/COMPARE/GENERAL all self-contained |
