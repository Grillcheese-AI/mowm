# mowm/web.py
"""MoWM Web Interface — Gradio-based Q&A with visual reasoning trace.

Usage:
    uv run python -m mowm.web
    uv run python -m mowm.web --port 7860
    uv run python -m mowm.web --share  # public URL via Gradio
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from mowm.axiom_library import AxiomLibrary
from mowm.causal_graph import CausalGraph
from mowm.domains import seed_all
from mowm.executor import SafeFormulaExecutor
from mowm.logic_engine import LogicEngine
from mowm.qa import QAPipeline, QAResponse

# Optional imports — graceful degradation
try:
    from mowm.reasoning import ReasoningDecoder
    HAS_REASONING = True
except ImportError:
    HAS_REASONING = False

try:
    from mowm.vocal_tract import VocalTract
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


def _build_pipeline(
    k: int = 4,
    l: int = 32,  # noqa: E741
    codebook_path: str | None = None,
) -> dict[str, Any]:
    """Build the full MoWM stack."""
    lib = AxiomLibrary(k=k, l=l)

    # Import and seed all domains
    all_domains = [
        "physics", "logic", "economics", "social", "linguistics",
        "causal", "atomic", "bio", "music", "identity", "colors",
        "language", "historical", "ensemble", "security",
    ]
    for d in all_domains:
        try:
            __import__(f"mowm.domains.{d}")
        except ImportError:
            pass
    seed_all(lib)

    executor = SafeFormulaExecutor()
    logic = LogicEngine()

    # Build causal graph with common nodes
    causal = CausalGraph(k=k, l=l)
    bc = lib.bc
    nodes = [
        "force", "acceleration", "velocity", "position",
        "mass", "energy", "momentum", "pressure",
        "temperature", "entropy", "price", "demand",
        "supply", "inflation", "interest_rate",
    ]
    for name in nodes:
        causal.add_node(name, "general", bc.random_discrete(
            seed=hash(name) % (2**31),
        ))
    edges = [
        ("force", "acceleration", "+", 0.9),
        ("acceleration", "velocity", "+", 0.8),
        ("velocity", "position", "+", 0.7),
        ("force", "momentum", "+", 0.85),
        ("mass", "momentum", "+", 0.8),
        ("energy", "temperature", "+", 0.6),
        ("temperature", "entropy", "+", 0.5),
        ("pressure", "force", "+", 0.7),
        ("demand", "price", "+", 0.8),
        ("supply", "price", "-", 0.8),
        ("inflation", "interest_rate", "+", 0.7),
        ("interest_rate", "demand", "-", 0.6),
    ]
    for src, tgt, pol, strength in edges:
        causal.add_edge(src, tgt, pol, strength)

    # Load NV-Embed codebook if available
    seq_encoder = None
    codebook_stats = {}
    if codebook_path:
        try:
            from mowm.sequence_encoder import SequenceEncoder
            seq_encoder = SequenceEncoder.from_codebook(
                codebook_path, k=k, l=l,
            )
            cb = np.load(codebook_path, allow_pickle=True)
            codebook_stats = {
                "embeddings": int(cb["embeddings"].shape[0]),
                "latent_dim": int(cb["latent_dim"]),
                "domains": list(cb["domain_names"]),
                "has_projection": "projection_matrix" in cb,
            }
            print(f"  Codebook: {codebook_stats['embeddings']} embeddings, "
                  f"{len(codebook_stats['domains'])} domains")
        except Exception as e:
            print(f"  Codebook load failed: {e}")

    qa = QAPipeline(
        library=lib, executor=executor, logic=logic,
        causal=causal, k=k, l=l,
        codebook_path=codebook_path,
    )

    return {
        "library": lib,
        "executor": executor,
        "logic": logic,
        "causal": causal,
        "qa": qa,
        "seq_encoder": seq_encoder,
        "codebook_stats": codebook_stats,
        "k": k,
        "l": l,
    }


def _format_response(response: QAResponse, elapsed: float) -> str:
    """Format QAResponse as rich markdown for the web UI."""
    lines: list[str] = []

    # Answer
    lines.append(f"### Answer\n\n{response.answer}\n")

    # Metadata bar
    mode_emoji = "🔢" if response.mode == "direct" else "🧠"
    conf_bar = "█" * int(response.confidence * 10) + "░" * (10 - int(response.confidence * 10))
    lines.append(
        f"**Mode:** {mode_emoji} `{response.mode}` "
        f"| **Domain:** `{response.domain}` "
        f"| **Confidence:** `{conf_bar}` {response.confidence:.0%} "
        f"| **Time:** `{elapsed:.2f}s`\n"
    )

    # Computed values
    if response.computed_values:
        lines.append("### Computed Values\n")
        for var, val in response.computed_values.items():
            if isinstance(val, float):
                lines.append(f"- **{var}** = `{val:.6g}`")
            else:
                lines.append(f"- **{var}** = `{val}`")
        lines.append("")

    # Evidence
    if response.evidence:
        lines.append("### Evidence (Axioms Used)\n")
        for e in response.evidence:
            lines.append(f"- 📐 {e}")
        lines.append("")

    # Reasoning trace
    if response.reasoning_trace:
        lines.append("<details><summary>🔍 Reasoning Trace</summary>\n")
        for step in response.reasoning_trace:
            lines.append(f"- {step}")
        lines.append("\n</details>")

    return "\n".join(lines)


def _format_library_stats(lib: AxiomLibrary) -> str:
    """Format library statistics as markdown."""
    domains = lib.domains()
    lines = ["### Axiom Library\n"]
    lines.append(f"**Total axioms:** {len(lib)}\n")
    lines.append("| Domain | Count |")
    lines.append("|--------|-------|")
    for domain, axioms in sorted(domains.items()):
        lines.append(f"| {domain} | {len(axioms)} |")
    return "\n".join(lines)


def _format_causal_graph(causal: CausalGraph) -> str:
    """Format causal graph info as markdown."""
    lines = ["### Causal Graph\n"]
    lines.append(f"**Nodes:** {causal.node_count} | **Edges:** {causal.edge_count}\n")
    for edge in causal.edges[:20]:
        arrow = "→+" if edge.polarity == "+" else "→−"
        lines.append(f"- `{edge.source}` {arrow} `{edge.target}` ({edge.strength:.1f})")
    if causal.edge_count > 20:
        lines.append(f"\n*... and {causal.edge_count - 20} more edges*")
    return "\n".join(lines)


def create_app(stack: dict[str, Any]):
    """Create the Gradio app."""
    import gradio as gr

    qa: QAPipeline = stack["qa"]
    lib: AxiomLibrary = stack["library"]
    causal: CausalGraph = stack["causal"]
    executor: SafeFormulaExecutor = stack["executor"]

    def explain_routing(question: str) -> str:
        """Show expert bid breakdown for a question."""
        if not question.strip():
            return "*Enter a question to see expert bids.*"
        return qa._classifier.explain_routing(question)

    def ask_question(question: str, facts_str: str) -> str:
        if not question.strip():
            return "*Please enter a question.*"

        # Parse optional facts (key=value pairs)
        facts = None
        if facts_str.strip():
            facts = {}
            for pair in facts_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    k = k.strip()
                    v = v.strip().lower()
                    facts[k] = v in ("true", "1", "yes")

        t0 = time.time()
        response = qa.ask(question, facts=facts)
        elapsed = time.time() - t0

        return _format_response(response, elapsed)

    def compute_formula(formula: str, variables_str: str) -> str:
        if not formula.strip():
            return "*Enter a formula (e.g., `F = m * a`)*"

        variables: dict[str, float] = {}
        for pair in variables_str.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    variables[k.strip()] = float(v.strip())
                except ValueError:
                    pass

        try:
            result = executor.execute(formula, variables)
            # Try solve_for if we have a target
            lines = [f"### Result\n\n**{formula}**\n"]
            if variables:
                var_str = ", ".join(f"{k}={v}" for k, v in variables.items())
                lines.append(f"With: {var_str}\n")
            lines.append(f"**= {result:.6g}**\n")

            # Try symbolic rearrangements
            if "=" in formula:
                lhs = formula.split("=")[0].strip()
                try:
                    rearranged = executor.solve_for(formula, lhs)
                    lines.append(f"\nSymPy rearrangement: `{rearranged}`")
                except Exception:
                    pass

            return "\n".join(lines)

        except Exception as e:
            return f"**Error:** {e}"

    codebook_stats = stack.get("codebook_stats", {})

    def get_stats() -> str:
        parts = [_format_library_stats(lib)]
        if codebook_stats:
            parts.append("\n\n---\n\n### NV-Embed Codebook\n")
            parts.append(
                f"**Embeddings:** {codebook_stats.get('embeddings', 0):,}\n"
                f"**Latent dim:** {codebook_stats.get('latent_dim', 0)}\n"
                f"**Projection matrix:** "
                f"{'loaded' if codebook_stats.get('has_projection') else 'not loaded'}\n"
                f"**Domains:** {', '.join(codebook_stats.get('domains', []))}"
            )
        parts.append("\n\n---\n\n" + _format_causal_graph(causal))
        return "".join(parts)

    # Build the Gradio interface
    with gr.Blocks(
        title="MoWM — Mixture of World Models",
    ) as app:
        embed_count = codebook_stats.get("embeddings", 0)
        embed_str = f" | {embed_count:,} NV-Embed vectors" if embed_count else ""
        gr.Markdown(
            "# 🌍 MoWM — Mixture of World Models\n"
            "**Self-organizing neuro-symbolic reasoning engine** "
            f"| {len(lib)} axioms across "
            f"{len(lib.domains())} domains"
            f"{embed_str}\n\n"
            "*Ask questions. Get deterministic, traceable answers "
            "backed by named axioms. No hallucination possible.*"
        )

        with gr.Tabs():
            # Tab 1: Q&A
            with gr.Tab("💬 Ask MoWM"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Question",
                            placeholder=(
                                "What is F when m=10 and a=9.8?\n"
                                "Why do heavier objects fall at the same rate?\n"
                                "Compare force and momentum\n"
                                "What happens if force increases by 10?\n"
                                "Is the obligation valid if signed_contract and paid_deposit?"
                            ),
                            lines=3,
                        )
                        facts_input = gr.Textbox(
                            label="Facts (optional, for logic evaluation)",
                            placeholder="signed_contract=true, paid_deposit=false",
                            lines=1,
                        )
                        ask_btn = gr.Button("Ask", variant="primary")
                    with gr.Column(scale=3):
                        answer_output = gr.Markdown(
                            label="Answer",
                            value="*Ask a question to get started.*",
                        )

                ask_btn.click(
                    ask_question,
                    inputs=[question_input, facts_input],
                    outputs=answer_output,
                )
                question_input.submit(
                    ask_question,
                    inputs=[question_input, facts_input],
                    outputs=answer_output,
                )

                # Example questions
                gr.Examples(
                    examples=[
                        ["What is F when m=10 and a=9.8?", ""],
                        ["What is KE when m=2 and v=3?", ""],
                        ["What is V when I=5 and R=10?", ""],
                        ["What happens if force increases by 10?", ""],
                        ["Is the obligation valid if signed_contract and paid_deposit?",
                         "signed_contract=true, paid_deposit=true"],
                        ["Is the obligation valid if signed_contract and paid_deposit?",
                         "signed_contract=true, paid_deposit=false"],
                        ["Why do heavier objects fall at the same rate?", ""],
                        ["Compare force and momentum", ""],
                        ["Explain Newton's Second Law", ""],
                        ["Tell me about entropy", ""],
                    ],
                    inputs=[question_input, facts_input],
                )

            # Tab 2: Formula Calculator
            with gr.Tab("🔢 Formula Calculator"):
                with gr.Row():
                    with gr.Column():
                        formula_input = gr.Textbox(
                            label="Formula",
                            placeholder="F = m * a",
                            lines=1,
                        )
                        vars_input = gr.Textbox(
                            label="Variables (comma-separated)",
                            placeholder="m=10, a=9.8",
                            lines=1,
                        )
                        calc_btn = gr.Button("Compute", variant="primary")
                    with gr.Column():
                        calc_output = gr.Markdown(
                            label="Result",
                            value="*Enter a formula and variables.*",
                        )

                calc_btn.click(
                    compute_formula,
                    inputs=[formula_input, vars_input],
                    outputs=calc_output,
                )

                gr.Examples(
                    examples=[
                        ["F = m * a", "m=10, a=9.8"],
                        ["KE = 0.5 * m * v^2", "m=2, v=3"],
                        ["V = I * R", "I=5, R=10"],
                        ["E = m * c^2", "m=1, c=299792458"],
                        ["F = m * a", "F=98, a=9.8"],
                    ],
                    inputs=[formula_input, vars_input],
                )

            # Tab 3: Knowledge Base
            # Tab 4: Expert Routing
            with gr.Tab("🎯 Expert Routing"):
                gr.Markdown(
                    "### Expert Bidding — See How Domains Compete\n\n"
                    "*No hardcoded routing. Each domain's axiom set bids "
                    "on the question. Highest bid wins.*"
                )
                route_input = gr.Textbox(
                    label="Question",
                    placeholder="What is superposition in quantum mechanics?",
                    lines=2,
                )
                route_btn = gr.Button("Show Expert Bids", variant="primary")
                route_output = gr.Markdown(
                    value="*Enter a question to see expert bids.*",
                )
                route_btn.click(
                    explain_routing,
                    inputs=route_input,
                    outputs=route_output,
                )
                route_input.submit(
                    explain_routing,
                    inputs=route_input,
                    outputs=route_output,
                )
                gr.Examples(
                    examples=[
                        "What is the force when m=10 and a=9.8?",
                        "Explain quantum superposition",
                        "What is superposition in circuit analysis?",
                        "How does DNA encode genetic information?",
                        "What is the Nash equilibrium?",
                        "Explain modus ponens",
                        "How does RSA encryption work?",
                        "What is the harmonic series?",
                    ],
                    inputs=route_input,
                )

            # Tab 5: Knowledge Base
            with gr.Tab("📚 Knowledge Base"):
                stats_output = gr.Markdown(value=get_stats())
                refresh_btn = gr.Button("Refresh Stats")
                refresh_btn.click(
                    lambda: get_stats(),
                    outputs=stats_output,
                )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoWM Web Interface",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve on (default: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--codebook", type=str, default=None,
        help="Path to NV-Embed codebook .npz (e.g., mowm_nvembed_codebook.npz)",
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

    print("Building MoWM stack...", end=" ", flush=True)
    t0 = time.time()
    stack = _build_pipeline(k=args.k, l=args.l, codebook_path=args.codebook)
    print(f"done ({time.time() - t0:.1f}s, "
          f"{len(stack['library'])} axioms)")

    app = create_app(stack)
    app.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
