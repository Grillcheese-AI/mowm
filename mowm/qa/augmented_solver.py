# mowm/qa/augmented_solver.py
"""AugmentedSolver — LLM-assisted Q&A with template fallback.

Packages MoWM reasoning context (domain, axioms, computed evidence) for LLM
generation via OpenRouter-compatible HTTP API.  Falls back to a deterministic
template response when no LLM endpoint is configured or when httpx is absent.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False

if TYPE_CHECKING:
    from mowm.axiom_library import AxiomLibrary
    from mowm.executor import SafeFormulaExecutor
    from mowm.qa.parser import ParsedQuestion

from mowm.qa.direct_solver import QAResponse


class AugmentedSolver:
    """Package MoWM reasoning context for LLM generation with template fallback.

    Args:
        library: AxiomLibrary for axiom lookup and keyword matching.
        executor: SafeFormulaExecutor for computing supporting evidence values.
        llm_endpoint: Optional OpenRouter-compatible HTTP endpoint URL.
        api_key: Optional bearer token for the LLM endpoint.
        max_axioms: Maximum number of relevant axioms to include in context (default 3).
        timeout: HTTP request timeout in seconds (default 10.0).
    """

    def __init__(
        self,
        library: "AxiomLibrary",
        executor: "SafeFormulaExecutor",
        llm_endpoint: str | None = None,
        api_key: str | None = None,
        max_axioms: int = 3,
        timeout: float = 10.0,
    ) -> None:
        self._lib = library
        self._executor = executor
        self._llm_endpoint = llm_endpoint
        self._api_key = api_key
        self._max_axioms = max_axioms
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        parsed: "ParsedQuestion",
        domain: str,
        facts: dict[str, Any] | None = None,
    ) -> QAResponse:
        """Generate an answer using LLM context or template fallback.

        Args:
            parsed: Output from QuestionParser.parse().
            domain: Top-level domain string (e.g. "physics").
            facts: Optional additional fact bindings.

        Returns:
            QAResponse with the generated answer.
        """
        facts = facts or {}
        context = self._build_context(parsed, domain, facts)

        if self._llm_endpoint and _HTTPX_AVAILABLE:
            return self._call_llm(parsed, context, domain)

        return self._template_fallback(parsed, context, domain)

    # ------------------------------------------------------------------
    # Axiom relevance scoring
    # ------------------------------------------------------------------

    def find_relevant_axioms(self, parsed: "ParsedQuestion", domain: str) -> list:
        """Return the most relevant axioms by keyword overlap scoring.

        Args:
            parsed: ParsedQuestion with .keywords and .raw fields.
            domain: Domain to preference (top-domain prefix match).

        Returns:
            List of Axiom objects sorted by descending relevance score,
            capped at self._max_axioms.
        """
        question_keywords = {kw.lower() for kw in parsed.keywords}
        if parsed.raw:
            # also include raw tokens
            import re
            question_keywords.update(
                tok.lower() for tok in re.findall(r"[A-Za-z_]\w*", parsed.raw)
            )

        scored: list[tuple[float, Any]] = []
        for axiom in self._lib.all_axioms():
            import re as _re
            axiom_tokens = {
                tok.lower()
                for tok in _re.findall(r"[A-Za-z_]\w*", axiom.name + " " + axiom.formula_str)
            }
            overlap = len(question_keywords & axiom_tokens)
            # Strong domain bonus — prefer axioms from the classified domain
            domain_bonus = 3.0 if axiom.domain.split(".")[0] == domain else 0.0
            score = float(overlap) + domain_bonus
            scored.append((score, axiom))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ax for _, ax in scored[: self._max_axioms]]

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_context(
        self,
        parsed: "ParsedQuestion",
        domain: str,
        facts: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a structured context dict for the LLM prompt or template.

        Returns a dict with keys:
            domain, axioms, computed_values, question
        """
        relevant = self.find_relevant_axioms(parsed, domain)

        # Execute computable axioms to gather supporting evidence
        computed: dict[str, Any] = {}
        for axiom in relevant:
            if parsed.variables:
                try:
                    result = self._executor.execute(axiom.formula_str, parsed.variables)
                    computed[axiom.name] = result
                except Exception:
                    pass

        return {
            "domain": domain,
            "axioms": [
                {"name": ax.name, "formula": ax.formula_str, "domain": ax.domain}
                for ax in relevant
            ],
            "computed_values": computed,
            "question": parsed.raw,
        }

    # ------------------------------------------------------------------
    # LLM call (OpenRouter format)
    # ------------------------------------------------------------------

    def _build_prompt(self, parsed: "ParsedQuestion", context: dict[str, Any]) -> str:
        """Build a structured system+user prompt string from context."""
        lines: list[str] = [
            f"Domain: {context['domain']}",
        ]
        if context["axioms"]:
            lines.append("Relevant axioms:")
            for ax in context["axioms"]:
                lines.append(f"  - {ax['name']}: {ax['formula']}")
        if context["computed_values"]:
            lines.append("Computed values:")
            for name, val in context["computed_values"].items():
                lines.append(f"  - {name}: {val}")
        lines.append(f"\nQuestion: {context['question']}")
        return "\n".join(lines)

    def _call_llm(
        self,
        parsed: "ParsedQuestion",
        context: dict[str, Any],
        domain: str,
    ) -> QAResponse:
        """POST to LLM endpoint with OpenRouter-compatible payload.

        Returns:
            QAResponse from LLM; falls back to template on any HTTP error.
        """
        prompt = self._build_prompt(parsed, context)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a world-model reasoning assistant. "
                        "Answer questions using the provided axioms and computed values."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = httpx.post(
                self._llm_endpoint,
                headers=headers,
                content=json.dumps(payload),
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return QAResponse(
                answer=answer,
                mode="LLM",
                confidence=0.7,
                domain=domain,
                evidence=[ax["name"] for ax in context["axioms"]],
                computed_values=context["computed_values"],
                reasoning_trace=[f"LLM response from {self._llm_endpoint}"],
            )
        except Exception as exc:
            # Fall back to template on any HTTP / parsing error
            resp = self._template_fallback(parsed, context, domain)
            resp.reasoning_trace.append(f"LLM call failed ({exc}); used template fallback.")
            return resp

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    def _template_fallback(
        self,
        parsed: "ParsedQuestion",
        context: dict[str, Any],
        domain: str,
    ) -> QAResponse:
        """Generate a deterministic template-based answer from context."""
        lines: list[str] = [f"[{domain.upper()} — template answer]"]
        lines.append(f"Question: {context['question']}")

        if context["axioms"]:
            lines.append("Relevant axioms:")
            for ax in context["axioms"]:
                lines.append(f"  - {ax['name']}: {ax['formula']}")

        if context["computed_values"]:
            lines.append("Computed supporting values:")
            for name, val in context["computed_values"].items():
                lines.append(f"  - {name}: {val}")

        if not context["axioms"] and not context["computed_values"]:
            lines.append("No specific axioms found for this question in the world model.")

        answer = "\n".join(lines)
        trace = ["Used template fallback (no LLM endpoint configured or httpx unavailable)."]

        # Dynamic confidence based on match quality
        n_axioms = len(context["axioms"])
        n_computed = len(context["computed_values"])
        # Check if axioms are from the right domain
        in_domain = sum(
            1 for ax in context["axioms"]
            if ax.get("domain", "").split(".")[0] == domain
        )

        confidence = 0.30  # base
        confidence += min(0.20, n_axioms * 0.05)  # more axioms = more confident
        confidence += min(0.15, n_computed * 0.10)  # computed values = stronger
        confidence += min(0.15, in_domain * 0.05)   # in-domain axioms = better match
        confidence = min(0.85, confidence)  # cap below direct solver

        return QAResponse(
            answer=answer,
            mode="TEMPLATE",
            confidence=round(confidence, 2),
            domain=domain,
            evidence=[ax["name"] for ax in context["axioms"]],
            computed_values=context["computed_values"],
            reasoning_trace=trace,
        )
