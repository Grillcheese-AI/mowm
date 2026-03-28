# CLAUDE.md

## Project Overview

MoWM (Mixture of World Models) is a general-purpose, self-organizing, recursive mixture of
expert world models with a living axiom library. Each expert selects axioms from a shared
Global World and constructs block-code transition functions via HYLA hypernetworks. A discovery
loop finds new axioms via symbolic regression when existing experts fail.

**License:** BSL-1.1 | **Python:** >=3.12 | **Package manager:** uv

## Commands

```bash
# Install (editable, with cubemind and grilly as sibling repos)
uv venv && uv pip install -e ".[dev]"

# Lint
uv run ruff check mowm/ tests/

# Tests (quick, stop on first failure)
uv run pytest tests/ -v -q -x --tb=short

# Single test file
uv run pytest tests/test_axiom_library.py -v
```

## Architecture

The pipeline flows: **Input -> HYLA Encoder -> DSelect-k Router -> Expert Worlds (recursive Z-axis) -> Bind & Mixture -> Cleanup -> CVL -> HYLA Decoder -> Output**

The Global World holds ~350 pre-seeded axioms across 14 domains, plus open-ended discovered axioms.

### Key Modules

- **`base.py`** — All ABCs: BaseWorld, BaseRouter, BaseDomain, MergeStrategy, SearchStrategy, CleanupStrategy, DiscoveryObserver, WorldFactory
- **`axiom_library.py`** — Global World axiom storage, selection (cosine similarity), composition (VSA bind)
- **`world.py`** — Recursive World class with HYLA hypernetworks, sub-world spawning, Oja consolidation
- **`router.py`** — DSelect-k gating over dynamic expert pool, spawn/prune logic
- **`pipeline.py`** — End-to-end orchestrator: encode -> route -> predict -> cleanup -> CVL -> decode
- **`discovery.py`** — Universal 3-step discovery loop: semantic mapping, symbolic regression, axiom distillation
- **`domains/`** — 14 domain modules, each exports a `seed(library)` function via `@register_domain`

### Dependencies

Imports from `cubemind` (BlockCodes, HYLA, DSelectKGate, ContrastiveValueEstimator, DisARM,
WorldEncoder, WorldManager) and `grilly` (GPU bridge). Both are editable sibling repos.

## Critical Conventions

- **Max 1000 lines per file.** Split by responsibility if approaching limit.
- **GoF design patterns.** All major components have abstract base classes in `base.py` so implementations can be swapped (Strategy, Abstract Factory, Composite, Template Method, Observer, Registry).
- **Always use grilly GPU ops, not raw numpy** for VSA operations. The 3-level fallback in BlockCodes handles this automatically.
- **Line length: 100** (ruff config in `pyproject.toml`).
- **Tests use small dimensions** (`k=4, l=32`) to avoid OOM.
- **onnx is excluded** from dependencies (CVE-2026-28500).
