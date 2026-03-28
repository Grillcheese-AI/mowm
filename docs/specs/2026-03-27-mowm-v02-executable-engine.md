# MoWM v0.2.0 Design Spec — Executable Axiom Engine

**Date:** 2026-03-27
**Author:** Nicolas Cloutier + Claude
**Status:** Draft
**Module:** `mowm/` (extends v0.1.0)
**Depends on:** v0.1.0 (base, axiom_library, world, router, pipeline, discovery, 14 domains)

## Overview

v0.1.0 stores axioms as block-code vectors with formula strings — but they
don't *compute*. v0.2.0 makes axioms executable: each formula can evaluate
inputs and produce outputs, enabling actual world simulation, causal graph
construction, and dual-logic reasoning.

This transforms MoWM from a symbolic routing system into a **computational
world engine** that can run physics, evaluate game theory, build causal
graphs, encode DNA, and synthesize harmonic phasors — all driven by the
same axiom library and expert routing infrastructure.

## What v0.2.0 Enables (from generalization docs)

- **Legal expert** evaluates `O(Action) <- Condition1 AND Condition2 AND NOT Exception`
- **Marketing expert** computes `P(Conversion) = sigmoid(w1*Attention + w2*Trust - w3*Friction)`
- **Physics expert** runs `F = m * a` and returns numerical force
- **Bio expert** encodes DNA k-mers into 10kD VSA space
- **Music expert** builds harmonic phasors via `x(t) = A * exp(i*(omega*t + phi))`
- **Causal expert** constructs signed causal graphs with polarity propagation
- **Identity expert** extracts personal axioms from conversational funneling
- **Ensemble expert** aggregates population vectors via `V_pop = sgn(sum(w_i * V_i))`

## Architecture

```
v0.1.0 (unchanged)                    v0.2.0 (new)
+------------------+                  +------------------+
| AxiomLibrary     |                  | AxiomExecutor    |
| (vectors + meta) |  -- extends -->  | (callable engine)|
+------------------+                  +------------------+
| World            |                  | CausalGraph      |
| (HYLA predict)   |  -- extends -->  | (signed digraph) |
+------------------+                  +------------------+
| Pipeline         |                  | LogicEngine      |
| (route+merge)    |  -- extends -->  | (deductive+prob) |
+------------------+                  +------------------+
| Discovery        |                  | SequenceEncoder  |
| (GP-SR)          |  -- extends -->  | (DNA, temporal)  |
+------------------+                  +------------------+
                                      | PhasorEngine     |
                                      | (harmonic bank)  |
                                      +------------------+
                                      | IdentityState    |
                                      | (persona extract)|
                                      +------------------+
                                      | DataLoader       |
                                      | (realms + JSONL) |
                                      +------------------+
                                      | SimEnvironment   |
                                      | (grid world)     |
                                      +------------------+
```

## Code Constraints (inherited from v0.1.0)

- Max 1000 lines per file
- GoF patterns with base classes in `base.py`
- All GPU ops via grilly (never raw numpy for VSA ops)
- Tests use k=4, l=32
- Line length 100 (ruff)
- Python >=3.12

## Module Structure (new files only)

```
mowm/
    # v0.1.0 files unchanged
    base.py                  # EXTEND: add BaseExecutor, BaseCausalGraph, BaseLogicEngine ABCs
    axiom_library.py         # EXTEND: Axiom gains .execute() via AxiomExecutor
    ...
    # v0.2.0 new files
    executor.py              # AxiomExecutor — parse formula strings into callables
    causal_graph.py          # CausalGraph — signed directed graph with propagation
    logic_engine.py          # LogicEngine — deductive (bool) + probabilistic (sigmoid)
    sequence_encoder.py      # SequenceEncoder — DNA k-mer, temporal event, text token
    phasor_engine.py         # PhasorEngine — harmonic phasor bank (from AURA phasor.py)
    identity_state.py        # IdentityState — persona extraction + axiom binding
    data_loader.py           # DataLoader — E:\RAW_TEXTS\realms connector
    sim_environment.py       # SimEnvironment — simple grid world for testing
    training.py              # Training loop — connect pipeline to trajectories
    tests/
        test_executor.py
        test_causal_graph.py
        test_logic_engine.py
        test_sequence_encoder.py
        test_phasor_engine.py
        test_identity_state.py
        test_data_loader.py
        test_sim_environment.py
        test_training.py
```

## Component Design

### 1. AxiomExecutor (`executor.py`)

Parses axiom formula strings into callable functions. Uses the same
operator grammar from the SR discovery loop, but in reverse — instead
of searching for a formula, it evaluates a known formula.

```python
class AxiomExecutor(ABC):
    """Base for executing axiom formulas on numerical inputs."""

    @abstractmethod
    def execute(self, formula_str: str, inputs: dict[str, float]) -> float | np.ndarray:
        """Evaluate a formula string with named variable inputs."""
        ...

class SafeFormulaExecutor(AxiomExecutor):
    """Execute formulas via AST parsing (no eval/exec).

    Parses formula strings like "F = m * a" into an AST, then evaluates
    with provided variable bindings. Supports all operators from the
    discovery loop grammar (+, -, *, /, sin, cos, exp, sqrt, square, log,
    AND, OR, NOT, IMPLIES).

    Uses the same ExprNode tree from discovery.py, extended with named
    variable resolution instead of positional x0, x1.
    """

    def __init__(self):
        self._cache: dict[str, ExprNode] = {}  # formula -> parsed tree

    def parse(self, formula_str: str) -> ExprNode:
        """Parse formula string into expression tree.

        Handles:
          - Infix: "F = m * a" -> extract RHS, parse to tree
          - Named vars: "m", "a", "v0", "theta" etc.
          - Constants: numeric literals
          - Functions: sin(x), sqrt(x), exp(x)
        """
        ...

    def execute(self, formula_str: str, inputs: dict[str, float]) -> float:
        """Evaluate formula with named variables.

        Example:
            executor.execute("F = m * a", {"m": 10.0, "a": 9.8})
            # returns 98.0
        """
        ...
```

The `Axiom` dataclass gains an optional `executor` field, and a convenience
`axiom.execute(inputs)` method that delegates to its executor.

### 2. CausalGraph (`causal_graph.py`)

Signed directed graph for causal reasoning. Each edge has polarity (+/-),
strength, and an optional axiom reference.

```python
class CausalNode:
    """A node in the causal graph."""
    name: str
    domain: str
    vector: np.ndarray          # block-code representation

class CausalEdge:
    """A signed causal link."""
    source: str
    target: str
    polarity: Literal["+", "-"]  # same-direction or opposite
    strength: float              # [0, 1]
    axiom: Axiom | None          # the axiom governing this link

class CausalGraph:
    """Signed directed causal graph with propagation.

    Supports:
      - add_node, add_edge, remove_node, remove_edge
      - propagate(node, delta) -> dict[str, float]
        Forward-propagate a change through the graph following edge polarity
      - counterfactual(node, value) -> dict[str, float]
        "What if node had value X?" via do-calculus approximation
      - find_paths(source, target) -> list of causal paths
      - feedback_loops() -> list of reinforcing/balancing loops
      - encode_to_vsa() -> np.ndarray
        Encode entire graph as a single block-code vector via
        bind(node1, bind(edge, node2)) bundled over all edges
    """
```

Reference: `grillcheese/backend/brain/temporal_knowledge.py` has causal
graph with nx.DiGraph. We adapt the pattern but keep it pure numpy
(no networkx dependency) with adjacency dict.

### 3. LogicEngine (`logic_engine.py`)

Dual-mode reasoning: deductive (boolean) and probabilistic (sigmoid).
Switches based on the axiom's domain prefix.

```python
class LogicMode(Enum):
    DEDUCTIVE = "deductive"       # boolean: True/False
    PROBABILISTIC = "probabilistic"  # sigmoid: [0, 1]

class LogicEngine:
    """Dual-mode logic evaluation.

    Deductive mode (logic.*, legal.*):
      - Evaluates propositions as boolean
      - Supports: AND, OR, NOT, IMPLIES, FORALL, EXISTS
      - Returns: True/False

    Probabilistic mode (economics.*, social.*, marketing.*):
      - Evaluates via sigmoid/softmax
      - Supports: weighted sums, sigmoid activation
      - Returns: float in [0, 1]

    Mode selection is automatic based on axiom domain prefix, or can
    be overridden manually.
    """

    def evaluate(self, axiom: Axiom, facts: dict[str, bool | float],
                 mode: LogicMode | None = None) -> bool | float:
        ...

    def check_obligation(self, conditions: list[str], facts: dict[str, bool],
                         exceptions: list[str] | None = None) -> bool:
        """Deontic logic: O(Action) <- Cond1 AND Cond2 AND NOT Exception"""
        ...

    def conversion_probability(self, factors: dict[str, float],
                                weights: dict[str, float]) -> float:
        """P(Conversion) = sigmoid(sum(w_i * factor_i))"""
        ...
```

Reference: `cubemind/docs/generalization/symbolic_logic_gate.md` has the
LegalLogicExpert and MarketingLogicExpert patterns.

### 4. SequenceEncoder (`sequence_encoder.py`)

Encodes sequences (DNA, temporal events, text tokens) into block-code
VSA space with positional binding.

```python
class SequenceEncoder:
    """Encode ordered sequences into block-code VSA vectors.

    Supports three modes:
      - DNA: k-mer fragmentation with permutation-based positioning
        V_seq = sgn(rho^0(V_A) + rho^1(V_C) + rho^2(V_G) + rho^3(V_T))
      - Temporal: event + timestamp binding with epoch vectors
        V_event = bind(time_vec, content_vec)
      - Text: token + position binding (like WorldEncoder but with order)
        V_text = sum(bind(token_i, pos_i))
    """

    def encode_dna(self, sequence: str, k: int = 3) -> np.ndarray:
        """Encode DNA nucleotide sequence via k-mer VSA.

        Splits into overlapping k-mers, each encoded as a bound
        sequence of nucleotide vectors with cyclic permutation for
        position. Bundles all k-mers into a single vector.
        """
        ...

    def encode_temporal(self, events: list[tuple[float, str]]) -> np.ndarray:
        """Encode time-stamped events with epoch binding.

        Each event = bind(timestamp_vec, content_vec).
        All events bundled with temporal weighting (recent = higher weight).
        """
        ...

    def encode_tokens(self, tokens: list[str]) -> np.ndarray:
        """Encode ordered token sequence with positional binding."""
        ...
```

Reference: `cubemind/docs/generalization/family_bio_kin_atomic.md` describes
the DNA k-mer encoding with permutation-based positioning.

### 5. PhasorEngine (`phasor_engine.py`)

Harmonic phasor bank for Music and signal processing domains. Adapted
from AURA_GENESIS `core/phasor.py` (PhasorBank with 384 harmonics).

```python
class PhasorEngine:
    """Harmonic phasor bank for frequency-domain VSA encoding.

    Each phasor = A * exp(i * (omega * t + phi)).
    A bank of H harmonics decomposes signals into frequency components.

    Used by:
      - Music domain: harmonic series, chord encoding, counterpoint
      - Colors domain: wavelength-frequency mapping
      - Ensemble domain: population "frequency" analysis

    Adapted from AURA_GENESIS/aura/core/phasor.py PhasorBank.
    """

    def __init__(self, n_harmonics: int = 64, rho: float = 0.985):
        ...

    def step(self, signal: float) -> np.ndarray:
        """Process one time step, return harmonic features."""
        ...

    def encode_chord(self, frequencies: list[float]) -> np.ndarray:
        """Encode a chord as superposition of harmonic phasors."""
        ...

    def encode_spectrum(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """FFT-based spectral encoding into block-code space."""
        ...

    def harmonic_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Consonance measure between two phasor encodings."""
        ...
```

### 6. IdentityState (`identity_state.py`)

Persona extraction from conversation. Builds a personal axiom set from
user statements, tracks emotional baseline, and maintains the "identity
vector" V_Me = sgn(V_Family + (V_Music * V_Math) + V_DNA).

```python
class IdentityState:
    """Personal world model — extracts axioms from conversation.

    Implements the "funneling conversations" pattern from identity_me.md:
      1. Chiseling Phase: extract core beliefs from casual input
      2. Axiom Binding: map beliefs to block-code identity vectors
      3. Consistency: ensure world remains congruent with identity

    Maintains:
      - Core beliefs as axioms (e.g., "Family is most important")
      - Emotional baseline (Plutchik 8-emotion model)
      - Trait stability scores
      - Lineage ledger (V_Me composition)
    """

    def __init__(self, k: int, l: int, library: AxiomLibrary):
        ...

    def process_statement(self, text: str) -> list[Axiom]:
        """Extract axioms from a natural language statement.

        "Family is the most important value" ->
          Axiom("family_priority", "identity.values", "family > all", ...)
        """
        ...

    def emotional_state(self) -> dict[str, float]:
        """Current Plutchik 8-emotion vector."""
        ...

    def identity_vector(self) -> np.ndarray:
        """Compose the full identity vector from all personal axioms."""
        ...

    def alignment_score(self, action_vec: np.ndarray) -> float:
        """How aligned is a proposed action with the identity?"""
        ...
```

Reference: `cubemind/docs/generalization/identity_me.md` and
`E:\RAW_TEXTS\realms\my_identity\emotions.jsonl` (10k+ Plutchik samples).

### 7. DataLoader (`data_loader.py`)

Connects to `E:\RAW_TEXTS\realms\` and other data sources. Loads
domain-specific training data as (input, output) pairs for the discovery
loop and pipeline training.

```python
class RealmDataLoader:
    """Load training data from E:\RAW_TEXTS\realms\ by domain.

    Maps realm directories to MoWM domains:
      quantum_mechanics/ -> physics, atomic
      my_identity/       -> identity
      finance/           -> economics
      history/           -> historical
      temporal_timelines_events/ -> historical
      instruction/       -> general training
      labeling/          -> domain classification

    Supports:
      - .txt files (full text, split into chunks)
      - .jsonl files (structured records)
      - .json files (temporal events, metadata)
    """

    def __init__(self, realms_path: str = "E:/RAW_TEXTS/realms"):
        ...

    def load_domain(self, domain: str, max_samples: int = 1000) -> list[dict]:
        """Load training samples for a specific domain."""
        ...

    def load_emotions(self) -> list[dict]:
        """Load Plutchik emotion samples from my_identity/emotions.jsonl."""
        ...

    def load_temporal_events(self) -> list[dict]:
        """Load chronological events from temporal_timelines_events/."""
        ...

    def domain_mapping(self) -> dict[str, list[str]]:
        """Return mapping of MoWM domains to realm directories."""
        ...
```

### 8. SimEnvironment (`sim_environment.py`)

Simple grid world for testing world model transitions end-to-end.
Validates that the pipeline can predict state changes from actions.

```python
class GridWorld:
    """Simple 2D grid world for testing MoWM transitions.

    States: (x, y) position on an NxN grid
    Actions: up, down, left, right
    Features: walls, terrain types (normal, ice, mud)

    Encodes states and actions as block-code vectors via WorldEncoder,
    then feeds them through the MoWM pipeline.
    """

    def __init__(self, size: int = 5, k: int = 4, l: int = 32):
        ...

    def reset(self) -> dict[str, str]:
        """Reset to starting position. Returns state dict."""
        ...

    def step(self, action: str) -> tuple[dict[str, str], float]:
        """Execute action, return (new_state, reward)."""
        ...

    def encode_state(self, state: dict) -> np.ndarray:
        """Encode grid state as block-code vector."""
        ...

    def generate_trajectories(self, n: int = 100) -> list[tuple]:
        """Generate random (state, action, next_state, reward) trajectories."""
        ...
```

### 9. Training Loop (`training.py`)

Connects the pipeline to real trajectories for learning. Computes the
full loss function from the v0.1.0 spec.

```python
class MoWMTrainer:
    """Training loop for MoWM pipeline.

    Loss = lambda_bind * L_bind
         + lambda_inv  * L_inv
         + lambda_ortho * L_ortho
         + lambda_div  * L_div
         + lambda_ent  * L_ent
         + lambda_cvl  * L_CVL

    Gradients through discrete selections via DisARM.
    """

    def __init__(self, pipeline: MoWMPipeline, lr: float = 1e-3):
        ...

    def train_step(self, trajectories: list[tuple]) -> dict[str, float]:
        """Single training step. Returns loss dict."""
        ...

    def train_epoch(self, data_loader: RealmDataLoader, domain: str) -> dict:
        """Train one epoch on a domain's data."""
        ...

    def evaluate(self, env: GridWorld, n_episodes: int = 10) -> dict:
        """Evaluate pipeline on grid world episodes."""
        ...
```

## Data Flow — Executable Pipeline

### v0.1.0 flow (symbolic only):
```
Input -> encode -> route -> HYLA predict -> merge -> cleanup -> output
```

### v0.2.0 flow (executable):
```
Input -> encode -> route -> expert world predicts:
  1. HYLA generates transition delta (as before)
  2. AxiomExecutor evaluates selected axiom formulas numerically
  3. LogicEngine applies deductive/probabilistic reasoning
  4. CausalGraph propagates effects through signed links
  5. Results merged: HYLA prediction + axiom computation + causal effects
-> merge -> cleanup -> CVL Q-value -> output
```

The key addition is step 2-4: alongside the HYLA-generated transition,
the executor runs the actual formulas, the logic engine evaluates
conditions, and the causal graph propagates consequences. These three
signals are merged with the HYLA prediction to produce a richer output.

## Test Strategy

Tests use k=4, l=32 as before.

| Test | What it validates |
|------|-------------------|
| `test_executor.py` | Parse "F = m * a", execute with inputs, edge cases |
| `test_causal_graph.py` | Add/remove nodes/edges, propagate, counterfactual, encode_to_vsa |
| `test_logic_engine.py` | Deductive (modus ponens), probabilistic (sigmoid), deontic obligation |
| `test_sequence_encoder.py` | DNA k-mer encoding, temporal events, token sequences |
| `test_phasor_engine.py` | Harmonic step, chord encoding, spectrum, consonance |
| `test_identity_state.py` | Statement -> axiom extraction, emotional state, alignment |
| `test_data_loader.py` | Load physics realm, load emotions JSONL, domain mapping |
| `test_sim_environment.py` | Grid world step, trajectory generation, encode state |
| `test_training.py` | Single train step, loss dict, evaluate on grid world |

## External References

- **AURA_GENESIS `core/phasor.py`** — PhasorBank (384 harmonics, leaky resonator)
- **AURA_GENESIS `training/causal_trainer.py`** — Causal graph with butterfly effect
- **AURA_GENESIS `emotion/plutchik.py`** — 8 base emotions with freq/amplitude/phase
- **GrillCheese `brain/temporal_knowledge.py`** — 100K events, causal chains
- **GrillCheese `symbolic/vsa_engine.py`** — bind/bundle/similarity wrapper
- **GrillCheese `plasticity/oja.py`** — Oja/Sanger with drift monitoring
- **Google Research `hypertransformer/`** — layerwise weight generation
- **aura-snn-rag `hippocampal.py`** — place/grid/time cells, episodic memory
- **E:\RAW_TEXTS\realms** — 5,998 training files across 35 domains
- **D:\google-research\** — (findings pending, will amend if relevant)

## Key Design Decisions

1. **SafeFormulaExecutor uses AST, not eval()** — no arbitrary code execution.
   Reuses ExprNode from discovery.py extended with named variable resolution.

2. **CausalGraph is pure numpy, no networkx** — adjacency dict with numpy
   arrays for propagation. Keeps the dependency footprint minimal.

3. **LogicEngine mode is auto-detected** from axiom domain prefix but
   overridable. No need for the user to specify deductive vs probabilistic.

4. **PhasorEngine adapted from AURA, not reimplemented** — same leaky
   resonator math, but using grilly ops where applicable and numpy fallback.

5. **DataLoader is path-configurable** — default E:\RAW_TEXTS\realms but
   any directory with the same structure works.

6. **SimEnvironment is intentionally simple** — 5x5 grid world is enough
   to validate the full pipeline end-to-end. Complex envs come later.

7. **Training uses numerical gradient descent** — same pattern as
   cubemind's CVL update_critic. No torch/jax dependency.

## Scope Check

This spec covers 9 new modules. Each is independent and testable in isolation.
The natural implementation order is:
1. executor (foundation — makes axioms callable)
2. causal_graph (depends on executor for edge axiom evaluation)
3. logic_engine (depends on executor for formula eval)
4. sequence_encoder (independent)
5. phasor_engine (independent)
6. identity_state (depends on axiom_library, executor)
7. data_loader (independent)
8. sim_environment (depends on pipeline)
9. training (depends on everything)
