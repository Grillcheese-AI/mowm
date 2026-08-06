"""MoWM Q&A Pipeline — question answering with world model reasoning.

Orchestrates: classify domain -> parse question -> direct solve -> augmented fallback.

Public API:
    QAPipeline — End-to-end Q&A orchestrator
    QAResponse  — Structured answer dataclass (re-exported from direct_solver)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mowm.causal_graph import CausalGraph
    from mowm.logic_engine import LogicEngine

from mowm.axiom_library import AxiomLibrary
from mowm.executor import SafeFormulaExecutor
from mowm.qa.augmented_solver import AugmentedSolver
from mowm.qa.classifier import DomainClassifier
from mowm.qa.expert_bid_classifier import ExpertBidClassifier
from mowm.qa.direct_solver import DirectSolver, QAResponse
from mowm.qa.formatter import ResponseFormatter
from mowm.qa.factual_lookup import FactualLookup
from mowm.qa.parser import QuestionParser

# Factual question patterns (when/where/who)
import re as _re
_FACTUAL_RE = _re.compile(
    r"^\s*(?:when|where|who|what year|what date)\b",
    _re.IGNORECASE,
)
# "what happened" is too broad — only trigger if a year follows
_WHAT_HAPPENED_RE = _re.compile(
    r"^\s*what\s+happened\s+(?:in\s+)?(\d{3,4})\b",
    _re.IGNORECASE,
)


class QAPipeline:
    """End-to-end Q&A pipeline: classify -> parse -> direct solve -> augmented fallback.

    Args:
        library: AxiomLibrary instance used for domain classification and axiom lookup.
        executor: SafeFormulaExecutor for formula evaluation.
        logic: LogicEngine for EVALUATE intent handling.
        causal: Optional CausalGraph for CAUSAL intent handling.
        k: VSA block count (used to build AxiomLibrary if not provided separately).
        l: VSA block length.
        codebook_path: Unused reserved parameter for future codebook loading.
        llm_endpoint: Optional OpenRouter-compatible endpoint URL for augmented answers.
        api_key: Optional bearer token for the LLM endpoint.
    """

    def __init__(
        self,
        library: AxiomLibrary,
        executor: SafeFormulaExecutor,
        logic: "LogicEngine",
        causal: "CausalGraph | None" = None,
        k: int = 80,
        l: int = 128,  # noqa: E741
        codebook_path: str | None = None,
        llm_endpoint: str | None = None,
        api_key: str | None = None,
        factual_data_paths: list[str] | None = None,
    ) -> None:
        self._lib = library
        self._executor = executor
        self._logic = logic
        self._causal = causal
        self._fmt = ResponseFormatter()
        self._parser = QuestionParser()
        # Expert-bidding classifier: domains compete based on axiom content
        # No hardcoded routing tables — competence derived from axioms
        self._classifier = ExpertBidClassifier(library)

        # Build a null causal graph if none provided (DirectSolver requires one)
        if self._causal is None:
            from mowm.causal_graph import CausalGraph
            self._causal = CausalGraph(k=library.k, l=library.l)

        self._direct = DirectSolver(
            library=library,
            executor=executor,
            logic_engine=logic,
            causal_graph=self._causal,
            formatter=self._fmt,
        )
        self._augmented = AugmentedSolver(
            library=library,
            executor=executor,
            llm_endpoint=llm_endpoint,
            api_key=api_key,
        )

        # Factual lookup for when/where/who questions
        default_fact_paths = [
            "D:/grillcheese_training_data/temporal/historical",
        ]
        self._factual = FactualLookup(
            data_paths=factual_data_paths or default_fact_paths,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, question: str, facts: dict[str, Any] | None = None) -> QAResponse:
        """Answer a single question using the full pipeline.

        Flow:
            1. Classify domain via DomainClassifier.
            2. Parse question into ParsedQuestion via QuestionParser.
            3. Attempt DirectSolver (COMPUTE / EVALUATE / CAUSAL).
            4. Fall back to AugmentedSolver (LLM or template).

        Args:
            question: Natural-language question string.
            facts: Optional boolean / numeric fact bindings for EVALUATE intent.

        Returns:
            QAResponse with answer, mode, confidence, domain, evidence, etc.
        """
        if not question or not question.strip():
            return QAResponse(
                answer="No question provided.",
                mode="NONE",
                confidence=0.0,
                domain="",
                reasoning_trace=["Empty question string received."],
            )

        facts = facts or {}

        # 1. Classify domain
        domain = self._classifier.top_domain(question)
        if not domain:
            domain = "general"

        # 2. Parse question
        parsed = self._parser.parse(question)

        # 2.5 Try factual lookup for when/where/who questions
        # Also handle "what happened in YEAR" with year-specific search
        _what_happened = _WHAT_HAPPENED_RE.match(question)
        is_factual = _FACTUAL_RE.match(question) or _what_happened
        if is_factual and self._factual.count > 0:
            fact_results = self._factual.search(question, top_n=3)
            if fact_results:
                best = fact_results[0]
                answer_lines = [best.answer]

                # Try to find a year: from entry, from question, or from answer text
                year = best.year
                if not year:
                    year = self._factual._extract_year(best.question)
                if not year:
                    year = self._factual._extract_year(best.answer)
                if year:
                    answer_lines.insert(0, f"**{year} AD**\n")
                if len(fact_results) > 1:
                    answer_lines.append("\n**Related:**")
                    for fr in fact_results[1:]:
                        snippet = fr.answer[:150]
                        answer_lines.append(f"- {snippet}...")

                return QAResponse(
                    answer="\n".join(answer_lines),
                    mode="FACTUAL",
                    confidence=0.75,
                    domain=domain,
                    evidence=[
                        f"Factual lookup ({self._factual.count} entries)",
                    ],
                    reasoning_trace=[
                        f"Factual question detected (when/where/who).",
                        f"Searched {self._factual.count} historical entries.",
                        f"Top match: {best.question[:80]}",
                    ],
                )

        # 3. Try direct solve
        if self._direct.can_solve(parsed, domain):
            resp = self._direct.solve(parsed, domain, facts=facts)
            if resp.confidence > 0.0:
                return resp

        # 4. Augmented fallback
        return self._augmented.solve(parsed, domain, facts=facts)

    def ask_batch(
        self,
        questions: list[str],
        facts: dict[str, Any] | None = None,
    ) -> list[QAResponse]:
        """Answer a list of questions, applying the same facts to each.

        Args:
            questions: List of natural-language question strings.
            facts: Optional shared fact bindings applied to every question.

        Returns:
            List of QAResponse objects in the same order as *questions*.
        """
        return [self.ask(q, facts=facts) for q in questions]


__all__ = ["QAPipeline", "QAResponse"]
