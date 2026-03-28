"""MoWM CLI — ask questions from the command line.

Usage:
    uv run python -m mowm.cli "What is F when m=10 and a=9.8?"
    uv run python -m mowm.cli --interactive
    uv run python -m mowm.cli --domain physics "Calculate KE when m=2 and v=3"
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.domains import seed_all
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa import QAPipeline, QAResponse


def _build_pipeline(
    k: int = 4,
    l: int = 32,  # noqa: E741
    domains: list[str] | None = None,
    codebook_path: str | None = None,
    llm_endpoint: str | None = None,
    api_key: str | None = None,
) -> QAPipeline:
    """Build the full QA pipeline with all domains seeded."""
    lib = AxiomLibrary(k=k, l=l)

    # Import and seed domains
    all_domains = [
        "physics", "logic", "economics", "social", "linguistics",
        "causal", "atomic", "bio", "music", "identity", "colors",
        "language", "historical", "ensemble",
    ]
    for d in (domains or all_domains):
        try:
            __import__(f"mowm.domains.{d}")
        except ImportError:
            pass
    seed_all(lib, domains=domains)

    executor = SafeFormulaExecutor()
    logic = LogicEngine()

    # Build a basic causal graph with common physics nodes
    causal = CausalGraph(k=k, l=l)
    bc = lib.bc
    for name in ("force", "acceleration", "velocity", "position",
                  "mass", "energy", "momentum", "pressure",
                  "temperature", "entropy"):
        causal.add_node(name, "physics", bc.random_discrete(
            seed=hash(name) % (2**31),
        ))
    causal.add_edge("force", "acceleration", "+", 0.9)
    causal.add_edge("acceleration", "velocity", "+", 0.8)
    causal.add_edge("velocity", "position", "+", 0.7)
    causal.add_edge("force", "momentum", "+", 0.85)
    causal.add_edge("mass", "momentum", "+", 0.8)
    causal.add_edge("energy", "temperature", "+", 0.6)
    causal.add_edge("temperature", "entropy", "+", 0.5)
    causal.add_edge("pressure", "force", "+", 0.7)

    return QAPipeline(
        library=lib,
        executor=executor,
        logic=logic,
        causal=causal,
        k=k,
        l=l,
        codebook_path=codebook_path,
        llm_endpoint=llm_endpoint,
        api_key=api_key,
    )


def _print_response(response: QAResponse, verbose: bool = False) -> None:
    """Print a QA response to stdout."""
    print(f"\n{response.answer}")
    print(f"\n  [{response.mode}] confidence={response.confidence:.2f} "
          f"domain={response.domain}")

    if verbose:
        if response.computed_values:
            print(f"  computed: {response.computed_values}")
        if response.evidence:
            print(f"  evidence:")
            for e in response.evidence:
                print(f"    - {e}")
        if response.reasoning_trace:
            print(f"  trace:")
            for t in response.reasoning_trace:
                print(f"    > {t}")


def _interactive(pipeline: QAPipeline, verbose: bool = False) -> None:
    """Run interactive Q&A loop."""
    print("MoWM Q&A — type a question, 'quit' to exit\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        t0 = time.time()
        response = pipeline.ask(question)
        elapsed = time.time() - t0

        _print_response(response, verbose)
        print(f"  ({elapsed:.2f}s)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoWM Q&A — ask questions answered by world model reasoning",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run interactive Q&A loop",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show evidence, trace, and computed values",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Restrict to specific domain (e.g., physics)",
    )
    parser.add_argument(
        "--codebook",
        default=None,
        help="Path to NV-Embed codebook .npz",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="LLM endpoint URL (e.g., https://openrouter.ai/api/v1/chat/completions)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for LLM endpoint",
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="VSA block count (default: 4, production: 80)",
    )
    parser.add_argument(
        "--l", type=int, default=32,
        help="VSA block length (default: 32, production: 128)",
    )

    args = parser.parse_args()

    # Build pipeline
    domains = [args.domain] if args.domain else None
    print("Loading MoWM...", end=" ", flush=True)
    t0 = time.time()
    pipeline = _build_pipeline(
        k=args.k,
        l=args.l,
        domains=domains,
        codebook_path=args.codebook,
        llm_endpoint=args.llm,
        api_key=args.api_key,
    )
    print(f"ready ({time.time() - t0:.1f}s, "
          f"{len(pipeline._lib)} axioms)\n")

    if args.interactive or args.question is None:
        _interactive(pipeline, args.verbose)
    else:
        response = pipeline.ask(args.question)
        _print_response(response, args.verbose)


if __name__ == "__main__":
    main()
