# mowm/reasoning/comparator.py
"""Comparator — structural diff and dual execution for COMPARE intent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mowm.executor import SafeFormulaExecutor
from mowm.reasoning.knowledge_search import _extract_formula_vars

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary


@dataclass
class ComparisonResult:
    """Result of comparing two or more concepts."""

    items: list[dict] = field(default_factory=list)
    similarities: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    structural_diff: str = ""


class Comparator:
    """Compare concepts by finding axioms, shared/unique variables, and executing both."""

    def __init__(self, library: AxiomLibrary, executor: SafeFormulaExecutor) -> None:
        self._lib = library
        self._executor = executor

    def compare(
        self,
        item_names: list[str],
        domain: str,
        variables: dict[str, float] | None = None,
    ) -> ComparisonResult:
        """Compare two or more concepts by axiom structure."""
        result = ComparisonResult()
        axioms = self._find_axioms(item_names, domain)

        if len(axioms) < 2:
            result.structural_diff = "Not enough axioms found for comparison."
            for ax in axioms:
                result.items.append({
                    "name": ax.name,
                    "formula": ax.formula_str,
                    "domain": ax.domain,
                    "value": None,
                })
            return result

        all_var_sets = []
        for ax in axioms:
            vars_set = _extract_formula_vars(ax.formula_str)
            all_var_sets.append(vars_set)

            value = None
            if variables:
                try:
                    value = self._executor.execute(ax.formula_str, variables)
                except Exception:
                    pass

            result.items.append({
                "name": ax.name,
                "formula": ax.formula_str,
                "domain": ax.domain,
                "value": value,
            })

        if len(all_var_sets) >= 2:
            shared = all_var_sets[0]
            for vs in all_var_sets[1:]:
                shared = shared & vs
            if shared:
                result.similarities.append(f"both depend on {', '.join(sorted(shared))}")

            for i, ax in enumerate(axioms):
                unique = all_var_sets[i] - shared if shared else all_var_sets[i]
                if unique:
                    others = set()
                    for j, vs in enumerate(all_var_sets):
                        if j != i:
                            others |= vs
                    truly_unique = unique - others
                    if truly_unique:
                        result.differences.append(
                            f"{ax.name} uniquely uses {', '.join(sorted(truly_unique))}"
                        )

            if axioms[0].domain == axioms[1].domain:
                result.similarities.append(f"same domain ({axioms[0].domain.split('.')[0]})")
            else:
                result.differences.append(
                    f"different domains: {axioms[0].domain} vs {axioms[1].domain}"
                )

            result.structural_diff = self._analyze_structure(
                axioms, all_var_sets, shared if shared else set()
            )

        return result

    def _find_axioms(self, names: list[str], domain: str) -> list[Axiom]:
        all_axioms = self._lib.all_axioms()
        found = []
        for name in names:
            name_lower = name.lower()
            for ax in all_axioms:
                if name_lower in ax.name.lower() or name_lower in ax.formula_str.lower():
                    if ax not in found:
                        found.append(ax)
                        break
        return found

    def _analyze_structure(self, axioms, var_sets, shared) -> str:
        if not shared:
            return "No shared mathematical foundation."

        names = [ax.name for ax in axioms]
        formulas = [ax.formula_str for ax in axioms]

        has_squared = any("^2" in f or "square" in f for f in formulas)
        has_linear = any("^2" not in f and "square" not in f for f in formulas)

        if has_squared and has_linear:
            sq_name = (
                names[0] if "^2" in formulas[0] or "square" in formulas[0] else names[1]
            )
            lin_name = names[1] if sq_name == names[0] else names[0]
            return (
                f"{sq_name} scales quadratically while {lin_name} scales linearly "
                f"with shared variables."
            )

        return f"Both share variables {', '.join(sorted(shared))} but differ in structure."
