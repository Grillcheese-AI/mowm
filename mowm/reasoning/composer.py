# mowm/reasoning/composer.py
"""Composer — template-based natural language generation from reasoning.

NOT a language model. Uses hierarchical templates:
  Level 1: Direct statement (1 sentence)
  Level 2: Statement + evidence (2-3 sentences)
  Level 3: Statement + derivation chain (paragraph)
  Level 4: Full explanation with intro/chain/causal/conclusion

Quality comes from reasoning depth, not fluency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mowm.reasoning.causal_tracer import CausalNarrative
    from mowm.reasoning.chain_builder import DerivationChain
    from mowm.reasoning.comparator import ComparisonResult
    from mowm.reasoning.knowledge_search import KnowledgeContext


class Composer:
    """Compose natural language from structured reasoning results.

    Template-based generation with complexity scaling:
      Level 1: Direct statement (1 sentence)
      Level 2: Statement + evidence (2-3 sentences)
      Level 3: Statement + derivation chain (paragraph)
      Level 4: Full explanation with intro/chain/causal/conclusion
    """

    # ------------------------------------------------------------------
    # compose_explanation
    # ------------------------------------------------------------------

    def compose_explanation(
        self,
        chain: DerivationChain,
        causal: CausalNarrative | None,
        question: str,
    ) -> str:
        """Compose a full explanation from derivation chain + causal trace.

        Structure (Level 4):
          1. Opening: restate the core principle
          2. Derivation: walk through axiom chain step by step
          3. Causal context: if available, explain cause-effect
          4. Conclusion: direct answer to the "why/how"
          5. Evidence: list axioms used

        Args:
            chain: DerivationChain from ChainBuilder.
            causal: Optional CausalNarrative from CausalTracer.
            question: Original question text.

        Returns:
            Composed natural language explanation string.
        """
        level = self._select_level(chain, causal)

        if level == 1:
            return self._compose_level1(chain, question)
        if level == 2:
            return self._compose_level2(chain, question)
        if level == 3:
            return self._compose_level3(chain, question)
        return self._compose_level4(chain, causal, question)

    # ------------------------------------------------------------------
    # compose_comparison
    # ------------------------------------------------------------------

    def compose_comparison(
        self,
        result: ComparisonResult,
        question: str,
    ) -> str:
        """Compose comparison text from structured diff.

        Args:
            result: ComparisonResult from Comparator.
            question: Original question text.

        Returns:
            Composed comparison text.
        """
        if not result.items:
            return "No items available for comparison."

        lines: list[str] = []

        # Header
        names = [item["name"] for item in result.items]
        if len(names) == 2:
            lines.append(f"{names[0]} and {names[1]} are compared below.")
        elif len(names) > 2:
            lines.append(
                f"{', '.join(names[:-1])}, and {names[-1]} "
                f"are compared below."
            )
        else:
            lines.append(f"Analysis of {names[0]}:")

        lines.append("")

        # Item details
        for item in result.items:
            name = item.get("name", "?")
            formula = item.get("formula")
            value = item.get("value")
            if formula:
                line = f"{name}: {formula}"
                if value is not None:
                    line += f" = {value}"
                lines.append(line)
            else:
                lines.append(f"{name}: no formula found")

        # Similarities
        if result.similarities:
            lines.append("")
            lines.append("Similarities:")
            for sim in result.similarities:
                lines.append(f"  - {sim}")

        # Differences
        if result.differences:
            lines.append("")
            lines.append("Differences:")
            for diff in result.differences:
                lines.append(f"  - {diff}")

        # Structural diff
        if result.structural_diff:
            lines.append("")
            lines.append(f"Structural: {result.structural_diff}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # compose_general
    # ------------------------------------------------------------------

    def compose_general(
        self,
        context: KnowledgeContext,
        chain: DerivationChain | None,
        question: str,
    ) -> str:
        """Compose general knowledge response from axiom library.

        Args:
            context: KnowledgeContext from KnowledgeSearch.
            chain: Optional DerivationChain if variables were present.
            question: Original question text.

        Returns:
            Composed general knowledge text.
        """
        lines: list[str] = []

        # Intro from concepts
        if context.concepts:
            if context.domain_chain:
                domain_str = ", ".join(context.domain_chain[:3])
                lines.append(f"{context.concepts[0]} ({domain_str}).")
            else:
                lines.append(f"{context.concepts[0]}.")

        # Key formulas from primary axioms
        if context.primary_axioms:
            lines.append("")
            lines.append("Key formulas:")
            for ax in context.primary_axioms:
                lines.append(f"  - {ax.name}: {ax.formula_str}")

        # Supporting axioms
        if context.supporting_axioms:
            lines.append("")
            lines.append("Related concepts:")
            for ax in context.supporting_axioms[:5]:
                lines.append(f"  - {ax.name}: {ax.formula_str}")

        # Derivation chain if present
        if chain and chain.steps:
            lines.append("")
            lines.append("Derivation:")
            for i, step in enumerate(chain.steps, 1):
                lines.append(f"  {i}. {step.explanation}")
            if chain.conclusion:
                lines.append(f"  => {chain.conclusion}")

        # Variables mentioned
        if context.variables_mentioned:
            sorted_vars = sorted(context.variables_mentioned)
            lines.append("")
            lines.append(f"Variables: {', '.join(sorted_vars[:15])}")

        # Axioms used
        all_axioms = context.primary_axioms + context.supporting_axioms
        if all_axioms:
            lines.append("")
            lines.append("Axioms used:")
            seen: set[str] = set()
            for ax in all_axioms:
                if ax.name not in seen:
                    seen.add(ax.name)
                    lines.append(f"  - {ax.name}")

        if not lines:
            return (
                "No specific information found in the world model "
                "for this question."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Level selection
    # ------------------------------------------------------------------

    def _select_level(
        self,
        chain: DerivationChain | None,
        causal: CausalNarrative | None,
    ) -> int:
        """Choose template complexity level (1-4).

        Level 1: No chain, no causal -> direct statement
        Level 2: Short chain (1 step), no causal
        Level 3: Multi-step chain, no causal
        Level 4: Chain + causal narrative
        """
        if chain is None or not chain.steps:
            return 1
        if causal is not None and len(chain.steps) >= 1:
            return 4
        if len(chain.steps) >= 2:
            return 3
        return 2

    # ------------------------------------------------------------------
    # Level templates
    # ------------------------------------------------------------------

    def _compose_level1(
        self, chain: DerivationChain, question: str,
    ) -> str:
        """Level 1: direct statement."""
        if chain.conclusion:
            return chain.conclusion
        return f"Based on the available axioms: {question}"

    def _compose_level2(
        self, chain: DerivationChain, question: str,
    ) -> str:
        """Level 2: statement + evidence."""
        lines: list[str] = []
        if chain.conclusion:
            lines.append(chain.conclusion)
        if chain.steps:
            step = chain.steps[0]
            lines.append(
                f"This follows from {step.axiom_name}: "
                f"{step.formula}."
            )
        if chain.axioms_used:
            lines.append(
                f"Axiom used: {', '.join(chain.axioms_used)}."
            )
        return "\n".join(lines) if lines else "No derivation available."

    def _compose_level3(
        self, chain: DerivationChain, question: str,
    ) -> str:
        """Level 3: statement + derivation chain."""
        lines: list[str] = []

        if chain.conclusion:
            lines.append(chain.conclusion)

        lines.append("")
        lines.append("Derivation:")
        for i, step in enumerate(chain.steps, 1):
            lines.append(f"  {i}. {step.explanation}")

        if chain.axioms_used:
            lines.append("")
            lines.append("Axioms used:")
            for name in chain.axioms_used:
                lines.append(f"  - {name}")

        return "\n".join(lines)

    def _compose_level4(
        self,
        chain: DerivationChain,
        causal: CausalNarrative | None,
        question: str,
    ) -> str:
        """Level 4: full explanation with intro/chain/causal/conclusion."""
        lines: list[str] = []

        # 1. Opening
        if chain.conclusion:
            lines.append(chain.conclusion)
        lines.append("")

        # 2. Derivation chain
        lines.append("Derivation:")
        for i, step in enumerate(chain.steps, 1):
            lines.append(f"  {i}. {step.explanation}")

        # 3. Causal context
        if causal:
            lines.append("")
            if causal.cancellations:
                for a, b in causal.cancellations:
                    lines.append(
                        f"The key insight is that {a} and {b} "
                        f"cancel each other out."
                    )
            if causal.net_effect:
                lines.append(f"Net effect: {causal.net_effect}")
            if causal.effects:
                lines.append("Causal path:")
                for target, direction, strength in causal.effects:
                    lines.append(
                        f"  - {causal.cause} -> {target} "
                        f"({direction}, strength {strength:.2f})"
                    )

        # 4. Conclusion
        if chain.conclusion:
            lines.append("")
            lines.append(f"Therefore: {chain.conclusion}")

        # 5. Evidence
        if chain.axioms_used:
            lines.append("")
            lines.append("Axioms used:")
            for name in chain.axioms_used:
                lines.append(f"  - {name}")

        return "\n".join(lines)
