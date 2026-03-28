# mowm/reasoning/chain_builder.py
"""ChainBuilder — build derivation chains by connecting axioms through shared variables."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mowm.executor import SafeFormulaExecutor
from mowm.reasoning.edge_types import SemanticEdge
from mowm.reasoning.knowledge_search import KnowledgeContext, _extract_formula_vars

if TYPE_CHECKING:
    pass  # no TYPE_CHECKING imports needed


@dataclass
class DerivationStep:
    """A single step in a derivation chain."""

    axiom_name: str
    formula: str
    substitutions: dict[str, float | str] = field(default_factory=dict)
    result_var: str = ""
    result_value: float | str = ""
    explanation: str = ""
    edge: SemanticEdge | None = None


@dataclass
class DerivationChain:
    """A complete derivation from axioms to conclusion."""

    steps: list[DerivationStep] = field(default_factory=list)
    conclusion: str = ""
    variables_used: set[str] = field(default_factory=set)
    axioms_used: list[str] = field(default_factory=list)


class ChainBuilder:
    """Build derivation chains from axiom knowledge contexts.

    Connects axioms through shared variables, executing formulas
    at each step to propagate computed values forward.
    """

    def __init__(self, executor: SafeFormulaExecutor) -> None:
        self._executor = executor

    def build_chain(
        self,
        context: KnowledgeContext,
        variables: dict[str, float],
        target: str | None = None,
    ) -> DerivationChain:
        """Build a numeric derivation chain from knowledge context.

        Executes axioms in order, propagating computed values.
        """
        chain = DerivationChain()
        current_vars = dict(variables)
        used_axioms: set[str] = set()

        all_axioms = context.primary_axioms + context.supporting_axioms
        max_rounds = min(len(all_axioms), 10)

        for _ in range(max_rounds):
            progress = False
            for ax in all_axioms:
                if ax.name in used_axioms:
                    continue

                lhs = self._extract_lhs(ax.formula_str)
                if not lhs:
                    continue
                if target and lhs == target and lhs in current_vars:
                    continue

                try:
                    result = self._executor.execute(ax.formula_str, current_vars)
                except Exception:
                    continue

                if result == 0.0 and lhs not in current_vars:
                    formula_vars = _extract_formula_vars(ax.formula_str)
                    rhs_vars = formula_vars - {lhs}
                    if not rhs_vars.issubset(set(current_vars.keys())):
                        continue

                step = DerivationStep(
                    axiom_name=ax.name,
                    formula=ax.formula_str,
                    substitutions=dict(current_vars),
                    result_var=lhs,
                    result_value=result,
                    explanation=f"From {ax.name}: {lhs} = {result}",
                )
                chain.steps.append(step)
                chain.axioms_used.append(ax.name)
                current_vars[lhs] = result
                used_axioms.add(ax.name)
                progress = True

                if target and lhs == target:
                    chain.conclusion = f"{target} = {result}"
                    break

            if not progress or (target and target in current_vars):
                break

        chain.variables_used = set(current_vars.keys())
        if not chain.conclusion and chain.steps:
            last = chain.steps[-1]
            chain.conclusion = f"{last.result_var} = {last.result_value}"
        return chain

    def build_explanation_chain(
        self,
        context: KnowledgeContext,
        keywords: list[str],
    ) -> DerivationChain:
        """Build a chain optimized for explanation (symbolic, no numeric values).

        Shows how axioms connect conceptually via shared variables.
        """
        chain = DerivationChain()
        all_axioms = context.primary_axioms + context.supporting_axioms
        if not all_axioms:
            return chain

        used: set[str] = set()

        for ax in all_axioms[:5]:
            if ax.name in used:
                continue
            formula_vars = _extract_formula_vars(ax.formula_str)
            lhs = self._extract_lhs(ax.formula_str) or ""

            step = DerivationStep(
                axiom_name=ax.name,
                formula=ax.formula_str,
                result_var=lhs,
                result_value=(
                    ax.formula_str.split("=", 1)[-1].strip()
                    if "=" in ax.formula_str
                    else ax.formula_str
                ),
                explanation=f"By {ax.name}: {ax.formula_str}",
            )
            chain.steps.append(step)
            chain.axioms_used.append(ax.name)
            chain.variables_used.update(formula_vars)
            used.add(ax.name)

        if chain.steps:
            chain.conclusion = f"Connected through: {', '.join(sorted(chain.variables_used)[:5])}"
        return chain

    def simplify_chain(self, chain: DerivationChain) -> DerivationChain:
        """Remove redundant steps where result_value is 0 and no downstream use."""
        if len(chain.steps) <= 1:
            return chain

        used_vars: set[str] = set()
        for step in chain.steps:
            formula_vars = _extract_formula_vars(step.formula)
            used_vars.update(formula_vars)

        simplified_steps = []
        for step in chain.steps:
            if step.result_value == 0.0 and step.result_var not in used_vars:
                continue
            simplified_steps.append(step)

        return DerivationChain(
            steps=simplified_steps or chain.steps[:1],
            conclusion=chain.conclusion,
            variables_used=chain.variables_used,
            axioms_used=[s.axiom_name for s in (simplified_steps or chain.steps[:1])],
        )

    @staticmethod
    def _extract_lhs(formula: str) -> str | None:
        if "=" not in formula or "==" in formula:
            return None
        lhs = formula.split("=", 1)[0].strip()
        if re.match(r"^[a-zA-Z_]\w*$", lhs):
            return lhs
        return None
