# mowm/reasoning/__init__.py
"""ReasoningDecoder — self-contained reasoning for EXPLAIN/COMPARE/GENERAL.

Orchestrates KnowledgeSearch, ChainBuilder, CausalTracer, Comparator, and
Composer to produce natural language answers without any external LLM.

Public API:
    ReasoningDecoder -- the main orchestrator
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary
    from mowm.causal_graph import CausalGraph
    from mowm.executor import SafeFormulaExecutor
    from mowm.logic_engine import LogicEngine

from mowm.qa.direct_solver import QAResponse
from mowm.qa.parser import Intent, ParsedQuestion
from mowm.reasoning.causal_tracer import CausalTracer
from mowm.reasoning.chain_builder import ChainBuilder
from mowm.reasoning.comparator import Comparator
from mowm.reasoning.composer import Composer
from mowm.reasoning.knowledge_search import KnowledgeSearch


class ReasoningDecoder:
    """Self-contained reasoning decoder for EXPLAIN/COMPARE/GENERAL.

    Replaces AugmentedSolver as the primary handler for open-ended
    questions. The LLM fallback stays as a true last resort.

    Args:
        library: AxiomLibrary for axiom lookup.
        executor: SafeFormulaExecutor for formula evaluation.
        logic: LogicEngine for logic evaluation.
        causal: Optional CausalGraph for causal tracing.
        k: Number of VSA blocks.
        l: VSA block length.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
        logic: LogicEngine,
        causal: CausalGraph | None = None,
        k: int = 80,
        l: int = 128,  # noqa: E741
    ) -> None:
        self._lib = library
        self._executor = executor
        self._logic = logic
        self._causal = causal
        self._k = k
        self._l = l

        self._search = KnowledgeSearch(library=library, k=k, l=l)
        self._chain = ChainBuilder(executor=executor)
        self._tracer = CausalTracer()
        self._comparator = Comparator(library=library, executor=executor)
        self._composer = Composer()

        # Build adjacency from CausalGraph if provided
        self._adjacency = self._build_adjacency(causal)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decode(
        self,
        parsed: ParsedQuestion,
        domain: str,
        facts: dict[str, Any] | None = None,
    ) -> QAResponse:
        """Generate a self-contained response using world model reasoning.

        Routes to the appropriate flow based on intent:
          EXPLAIN -> knowledge search + chain + causal trace + compose
          COMPARE -> knowledge search + comparator + compose
          GENERAL -> knowledge search + optional chain + compose

        Args:
            parsed: Output from QuestionParser.parse().
            domain: Top-level domain string.
            facts: Optional variable/fact bindings.

        Returns:
            QAResponse with answer, confidence, evidence, etc.
        """
        facts = facts or {}
        trace: list[str] = []

        if parsed.intent == Intent.EXPLAIN:
            return self._decode_explain(parsed, domain, trace)
        if parsed.intent == Intent.COMPARE:
            return self._decode_compare(parsed, domain, trace)
        return self._decode_general(parsed, domain, trace)

    # ------------------------------------------------------------------
    # EXPLAIN flow
    # ------------------------------------------------------------------

    def _decode_explain(
        self,
        parsed: ParsedQuestion,
        domain: str,
        trace: list[str],
    ) -> QAResponse:
        """EXPLAIN flow: search -> chain -> causal -> compose."""
        trace.append(f"EXPLAIN intent for domain={domain}")

        # 1. Knowledge search
        ctx = self._search.search(
            keywords=parsed.keywords,
            domain=domain,
            target_var=parsed.target,
        )
        trace.append(
            f"Found {len(ctx.primary_axioms)} primary, "
            f"{len(ctx.supporting_axioms)} supporting axioms"
        )

        # 2. Build explanation chain
        chain = self._chain.build_explanation_chain(
            context=ctx,
            keywords=parsed.keywords,
        )
        trace.append(f"Built chain with {len(chain.steps)} steps")

        # 3. Causal trace (if adjacency available)
        causal_narrative = None
        if self._adjacency:
            causal_narrative = self._tracer.trace_explanation(
                parsed.keywords, adjacency=self._adjacency,
            )
            if causal_narrative and causal_narrative.cause:
                trace.append(
                    f"Causal trace from '{causal_narrative.cause}' "
                    f"with {len(causal_narrative.effects)} effects"
                )
            else:
                causal_narrative = None

        # 4. Compose
        answer = self._composer.compose_explanation(
            chain=chain,
            causal=causal_narrative,
            question=parsed.raw,
        )

        confidence = self._estimate_confidence(ctx, chain)
        evidence = list(chain.axioms_used)

        return QAResponse(
            answer=answer,
            mode="EXPLAIN",
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # COMPARE flow
    # ------------------------------------------------------------------

    def _decode_compare(
        self,
        parsed: ParsedQuestion,
        domain: str,
        trace: list[str],
    ) -> QAResponse:
        """COMPARE flow: extract items -> comparator -> compose."""
        trace.append(f"COMPARE intent for domain={domain}")

        # Extract item names from keywords
        item_names = self._extract_comparison_items(parsed)
        trace.append(f"Comparison items: {item_names}")

        # Run comparison
        result = self._comparator.compare(
            item_names=item_names,
            domain=domain,
            variables=parsed.variables or None,
        )
        trace.append(
            f"Found {len(result.similarities)} similarities, "
            f"{len(result.differences)} differences"
        )

        # Compose
        answer = self._composer.compose_comparison(
            result=result,
            question=parsed.raw,
        )

        evidence = [
            item["name"]
            for item in result.items
        ]
        confidence = 0.7 if len(result.items) >= 2 else 0.4

        return QAResponse(
            answer=answer,
            mode="COMPARE",
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # GENERAL flow
    # ------------------------------------------------------------------

    def _decode_general(
        self,
        parsed: ParsedQuestion,
        domain: str,
        trace: list[str],
    ) -> QAResponse:
        """GENERAL flow: search -> optional chain -> compose."""
        trace.append(f"GENERAL intent for domain={domain}")

        # 1. Knowledge search
        ctx = self._search.search(
            keywords=parsed.keywords,
            domain=domain,
        )
        trace.append(
            f"Found {len(ctx.primary_axioms)} primary, "
            f"{len(ctx.supporting_axioms)} supporting axioms"
        )

        # 2. Optional chain if variables present
        chain = None
        if parsed.variables:
            chain = self._chain.build_chain(
                context=ctx,
                variables=parsed.variables,
                target=parsed.target,
            )
            trace.append(f"Built chain with {len(chain.steps)} steps")

        # 3. Compose
        answer = self._composer.compose_general(
            context=ctx,
            chain=chain,
            question=parsed.raw,
        )

        confidence = self._estimate_confidence(ctx, chain)
        evidence = [ax.name for ax in ctx.primary_axioms]

        return QAResponse(
            answer=answer,
            mode="GENERAL",
            confidence=confidence,
            domain=domain,
            evidence=evidence,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_adjacency(
        self, causal: CausalGraph | None,
    ) -> dict | None:
        """Convert CausalGraph to SemanticEdge adjacency dict for CausalTracer."""
        if causal is None:
            return None
        from mowm.reasoning.edge_types import EdgeType, SemanticEdge

        adj: dict[str, list[SemanticEdge]] = {}
        for src, edges in causal._adj.items():
            semantic_edges = []
            for e in edges:
                etype = (
                    EdgeType.INCREASES if e.polarity == "+"
                    else EdgeType.DECREASES
                )
                sem = SemanticEdge(
                    source_id=src,
                    target_id=e.target,
                    edge_type=etype,
                    weight=e.strength,
                )
                semantic_edges.append(sem)
            if semantic_edges:
                adj[src] = semantic_edges
        return adj if adj else None

    def _extract_comparison_items(
        self,
        parsed: ParsedQuestion,
    ) -> list[str]:
        """Extract item names for comparison from parsed question.

        Strategy: find axiom names that match keywords via name tokens
        or formula tokens, then deduplicate and return up to 4 items.
        Falls back to keyword pairs if fewer than 2 axioms matched.
        """
        all_axioms = self._lib.all_axioms()
        kw_set = {kw.lower() for kw in parsed.keywords}

        matched: list[str] = []
        for ax in all_axioms:
            # Match on axiom name tokens
            ax_name_tokens = {
                t.lower()
                for t in re.findall(r"[A-Za-z_]\w*", ax.name)
            }
            # Also match on formula tokens
            ax_formula_tokens = {
                t.lower()
                for t in re.findall(r"[A-Za-z_]\w*", ax.formula_str)
            }
            all_tokens = ax_name_tokens | ax_formula_tokens
            if kw_set & all_tokens:
                matched.append(ax.name)

        if len(matched) >= 2:
            return matched[:4]  # Max 4 items

        # Fallback: use the keywords themselves as item names
        # Filter out stop-words and comparison verbs
        _SKIP = {"compare", "and", "vs", "versus", "between", "difference"}
        items = [kw for kw in parsed.keywords if kw.lower() not in _SKIP]
        if len(items) >= 2:
            return items[:4]

        return parsed.keywords[:2] if parsed.keywords else []

    def _estimate_confidence(
        self,
        ctx,
        chain=None,
    ) -> float:
        """Estimate response confidence from context and chain quality."""
        score = 0.0

        # Primary axioms found -> base confidence
        if ctx.primary_axioms:
            score += 0.4
        if len(ctx.primary_axioms) >= 3:
            score += 0.1

        # Chain steps -> higher confidence
        if chain and chain.steps:
            score += 0.2
            if len(chain.steps) >= 2:
                score += 0.1

        # Domain chain -> domain relevance
        if ctx.domain_chain:
            score += 0.1

        return min(score, 0.95)


__all__ = ["ReasoningDecoder"]
