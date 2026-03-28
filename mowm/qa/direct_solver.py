# mowm/qa/direct_solver.py
"""DirectSolver — deterministic Q&A solver that bypasses LLM for structured intents.

Handles COMPUTE, EVALUATE, and CAUSAL intents by delegating to domain subsystems.
Returns can_solve=False for EXPLAIN, COMPARE, and GENERAL intents.

QAResponse is defined here (not in __init__.py) to avoid circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary
    from mowm.causal_graph import CausalGraph
    from mowm.executor import SafeFormulaExecutor
    from mowm.logic_engine import LogicEngine
    from mowm.qa.formatter import ResponseFormatter
    from mowm.qa.parser import ParsedQuestion

from mowm.qa.parser import Intent


@dataclass
class QAResponse:
    """Structured response from DirectSolver.solve().

    Attributes:
        answer: Human-readable answer string.
        mode: Intent string that produced the answer (e.g. "COMPUTE").
        confidence: Scalar confidence in [0.0, 1.0].
        domain: Domain used to resolve the answer.
        evidence: List of supporting evidence strings (axiom names, facts, etc.).
        computed_values: Mapping of variable names to their computed values.
        reasoning_trace: Step-by-step reasoning log.
    """

    answer: str = ""
    mode: str = ""
    confidence: float = 0.0
    domain: str = ""
    evidence: list[str] = field(default_factory=list)
    computed_values: dict[str, Any] = field(default_factory=dict)
    reasoning_trace: list[str] = field(default_factory=list)


class DirectSolver:
    """Compute answers without an LLM for structured, deterministic intents.

    Args:
        library: AxiomLibrary for axiom lookup.
        executor: SafeFormulaExecutor for formula evaluation.
        logic_engine: LogicEngine for obligation / formula evaluation.
        causal_graph: CausalGraph for causal propagation.
        formatter: ResponseFormatter for building the answer string.
    """

    def __init__(
        self,
        library: "AxiomLibrary",
        executor: "SafeFormulaExecutor",
        logic_engine: "LogicEngine",
        causal_graph: "CausalGraph",
        formatter: "ResponseFormatter",
    ) -> None:
        self._lib = library
        self._executor = executor
        self._logic = logic_engine
        self._graph = causal_graph
        self._fmt = formatter

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def can_solve(self, parsed: "ParsedQuestion", domain: str) -> bool:
        """Return True if DirectSolver can handle this intent without an LLM.

        COMPUTE, EVALUATE, CAUSAL → potentially solvable.
        EXPLAIN, COMPARE, GENERAL → always False.

        Args:
            parsed: Output from QuestionParser.parse().
            domain: Top-level domain string (e.g. "physics").

        Returns:
            bool
        """
        if parsed.intent == Intent.COMPUTE:
            return self._can_solve_compute(parsed, domain)
        if parsed.intent == Intent.EVALUATE:
            return True
        if parsed.intent == Intent.CAUSAL:
            return True
        return False

    def solve(
        self,
        parsed: "ParsedQuestion",
        domain: str,
        facts: dict[str, Any] | None = None,
    ) -> QAResponse:
        """Attempt to solve the question deterministically.

        Args:
            parsed: Output from QuestionParser.parse().
            domain: Top-level domain string.
            facts: Boolean / float fact bindings for EVALUATE intent.

        Returns:
            QAResponse — check response.confidence > 0 for a successful solve.
        """
        facts = facts or {}
        if parsed.intent == Intent.COMPUTE:
            return self._solve_compute(parsed, domain)
        if parsed.intent == Intent.EVALUATE:
            return self._solve_evaluate(parsed, domain, facts)
        if parsed.intent == Intent.CAUSAL:
            return self._solve_causal(parsed, domain)
        # EXPLAIN / COMPARE / GENERAL — cannot solve
        return QAResponse(
            answer="This question requires an LLM to answer.",
            mode=parsed.intent.value,
            confidence=0.0,
            domain=domain,
            reasoning_trace=["Intent is not directly solvable by DirectSolver."],
        )

    # ------------------------------------------------------------------
    # COMPUTE
    # ------------------------------------------------------------------

    def _find_axiom_for_target(self, target: str, domain: str):
        """Return the best axiom whose formula contains *target* as a variable."""
        candidates = [
            ax for ax in self._lib.all_axioms()
            if ax.domain.split(".")[0] == domain and target in ax.formula_str
        ]
        if not candidates:
            # Fallback: search all domains
            candidates = [
                ax for ax in self._lib.all_axioms()
                if target in ax.formula_str
            ]
        return candidates[0] if candidates else None

    def _can_solve_compute(self, parsed: "ParsedQuestion", domain: str) -> bool:
        if parsed.target is None:
            return False
        axiom = self._find_axiom_for_target(parsed.target, domain)
        return axiom is not None

    def _solve_compute(self, parsed: "ParsedQuestion", domain: str) -> QAResponse:
        trace: list[str] = []

        if parsed.target is None:
            return QAResponse(
                answer="No target variable specified.",
                mode=Intent.COMPUTE.value,
                confidence=0.0,
                domain=domain,
                reasoning_trace=["No target variable found in question."],
            )

        axiom = self._find_axiom_for_target(parsed.target, domain)
        if axiom is None:
            return QAResponse(
                answer=f"No axiom found for target variable '{parsed.target}'.",
                mode=Intent.COMPUTE.value,
                confidence=0.0,
                domain=domain,
                reasoning_trace=[f"Searched library for axiom with target '{parsed.target}': not found."],
            )

        trace.append(f"Matched axiom: {axiom.name!r} ({axiom.formula_str})")

        # Check all required variables are provided
        # Validate formula is parseable before executing
        try:
            self._executor.parse(axiom.formula_str)
        except Exception as exc:
            return QAResponse(
                answer=f"Formula parse error: {exc}",
                mode=Intent.COMPUTE.value,
                confidence=0.0,
                domain=domain,
                evidence=[axiom.name],
                reasoning_trace=[f"Parse error: {exc}"],
            )

        try:
            result = self._executor.execute(axiom.formula_str, parsed.variables)
        except Exception as exc:
            return QAResponse(
                answer=f"Execution error: {exc}",
                mode=Intent.COMPUTE.value,
                confidence=0.0,
                domain=domain,
                evidence=[axiom.name],
                reasoning_trace=[f"Execution error: {exc}"],
            )

        trace.append(f"Computed {parsed.target} = {result}")

        answer = self._fmt.format_compute(
            axiom_name=axiom.name,
            formula=axiom.formula_str,
            variables=parsed.variables,
            result=result,
        )

        computed = dict(parsed.variables)
        computed[parsed.target] = result

        return QAResponse(
            answer=answer,
            mode=Intent.COMPUTE.value,
            confidence=0.9,
            domain=axiom.domain.split(".")[0],
            evidence=[axiom.name],
            computed_values=computed,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------------

    def _solve_evaluate(
        self,
        parsed: "ParsedQuestion",
        domain: str,
        facts: dict[str, Any],
    ) -> QAResponse:
        trace: list[str] = []

        # If conditions were extracted, use check_obligation
        if parsed.conditions:
            trace.append(f"Checking obligation for conditions: {parsed.conditions}")
            bool_facts = {k: bool(v) for k, v in facts.items()}
            try:
                result = self._logic.check_obligation(parsed.conditions, bool_facts)
            except Exception as exc:
                return QAResponse(
                    answer=f"Logic error: {exc}",
                    mode=Intent.EVALUATE.value,
                    confidence=0.0,
                    domain=domain,
                    reasoning_trace=[f"check_obligation raised: {exc}"],
                )
            trace.append(f"Obligation holds: {result}")
            answer = self._fmt.format_evaluate(parsed.conditions, result)
            return QAResponse(
                answer=answer,
                mode=Intent.EVALUATE.value,
                confidence=0.85,
                domain=domain,
                computed_values={"holds": result},
                reasoning_trace=trace,
            )

        # Fallback: try to evaluate as a logic formula using keywords
        # Look for an axiom formula that uses evaluate-style logic
        logic_axioms = [
            ax for ax in self._lib.all_axioms()
            if ax.domain.split(".")[0] == domain
        ]
        if logic_axioms and facts:
            axiom = logic_axioms[0]
            trace.append(f"Evaluating logic formula: {axiom.formula_str!r}")
            try:
                bool_facts_typed: dict[str, bool | float] = {
                    k: bool(v) if isinstance(v, (bool, int)) else float(v)
                    for k, v in facts.items()
                }
                result_val = self._logic.evaluate(axiom.formula_str, bool_facts_typed)
                holds = bool(result_val) if isinstance(result_val, bool) else result_val > 0.5
            except Exception as exc:
                return QAResponse(
                    answer=f"Logic evaluation error: {exc}",
                    mode=Intent.EVALUATE.value,
                    confidence=0.0,
                    domain=domain,
                    reasoning_trace=[f"evaluate raised: {exc}"],
                )
            trace.append(f"Formula result: {result_val}")
            answer = self._fmt.format_evaluate([], holds)
            return QAResponse(
                answer=answer,
                mode=Intent.EVALUATE.value,
                confidence=0.75,
                domain=domain,
                evidence=[axiom.name],
                computed_values={"result": result_val},
                reasoning_trace=trace,
            )

        # Minimal evaluation — no conditions, no facts
        trace.append("No conditions or facts provided; defaulting to False.")
        answer = self._fmt.format_evaluate([], False)
        return QAResponse(
            answer=answer,
            mode=Intent.EVALUATE.value,
            confidence=0.5,
            domain=domain,
            reasoning_trace=trace,
        )

    # ------------------------------------------------------------------
    # CAUSAL
    # ------------------------------------------------------------------

    def _solve_causal(self, parsed: "ParsedQuestion", domain: str) -> QAResponse:
        trace: list[str] = []

        node = parsed.causal_node
        delta = parsed.causal_delta

        if node is None or delta is None:
            return QAResponse(
                answer="Could not extract causal node or delta from question.",
                mode=Intent.CAUSAL.value,
                confidence=0.0,
                domain=domain,
                reasoning_trace=["causal_node or causal_delta is None."],
            )

        trace.append(f"Propagating delta={delta} from node='{node}'")

        try:
            effects = self._graph.propagate(node, delta)
        except KeyError:
            return QAResponse(
                answer=f"Node '{node}' is not in the causal graph.",
                mode=Intent.CAUSAL.value,
                confidence=0.0,
                domain=domain,
                reasoning_trace=[f"KeyError: node '{node}' not found in graph."],
            )

        trace.append(f"Downstream effects: {effects}")

        effect_strs = [f"{n}: {v:+.4g}" for n, v in effects.items()]
        answer = self._fmt.format_causal(node=node, delta=delta, effects=effect_strs)

        return QAResponse(
            answer=answer,
            mode=Intent.CAUSAL.value,
            confidence=0.9,
            domain=domain,
            computed_values=dict(effects),
            reasoning_trace=trace,
        )
