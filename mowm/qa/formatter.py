# mowm/qa/formatter.py
"""ResponseFormatter — pure formatting helpers for Q&A pipeline responses.

No QA pipeline dependencies; can be used standalone.
"""
from __future__ import annotations

from typing import Any


class ResponseFormatter:
    """Format structured Q&A results into human-readable strings."""

    # ------------------------------------------------------------------
    # format_compute
    # ------------------------------------------------------------------

    def format_compute(
        self,
        axiom_name: str,
        formula: str,
        variables: dict[str, Any],
        result: float | int,
        unit: str = "",
    ) -> str:
        """Format a compute result.

        Args:
            axiom_name: Name of the axiom used (e.g. "Newton's Second Law").
            formula: Formula string (e.g. "F = m * a").
            variables: Mapping of variable names to their values.
            result: Computed numeric result.
            unit: Optional physical unit (e.g. "N", "m/s^2").

        Returns:
            Multi-line formatted string.
        """
        lines: list[str] = []
        lines.append(f"Axiom: {axiom_name}")
        lines.append(f"Formula: {formula}")
        if variables:
            lines.append("Variables:")
            for name, value in variables.items():
                lines.append(f"  {name} = {value}")
        unit_str = f" {unit}" if unit else ""
        lines.append(f"Result: {result}{unit_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # format_evaluate
    # ------------------------------------------------------------------

    def format_evaluate(
        self,
        conditions: list[str],
        result: bool,
        exceptions: list[str] | None = None,
    ) -> str:
        """Format a logical evaluation result.

        Args:
            conditions: List of condition strings evaluated.
            result: Whether all conditions hold.
            exceptions: Optional list of conditions that failed / exceptions raised.

        Returns:
            Multi-line formatted string.
        """
        lines: list[str] = []
        verdict = "holds" if result else "does not hold"
        lines.append(f"Evaluation: {verdict}")
        if conditions:
            lines.append("Conditions:")
            for cond in conditions:
                lines.append(f"  - {cond}")
        if exceptions:
            lines.append("Exceptions:")
            for exc in exceptions:
                lines.append(f"  - {exc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # format_causal
    # ------------------------------------------------------------------

    def format_causal(
        self,
        node: str,
        delta: float | int,
        effects: list[str] | None = None,
    ) -> str:
        """Format a causal chain description.

        Args:
            node: The causal node / variable name.
            delta: Change in the node's value (positive or negative).
            effects: List of downstream effect descriptions.

        Returns:
            Multi-line formatted string.
        """
        lines: list[str] = []
        direction = "increases" if delta >= 0 else "decreases"
        lines.append(f"Causal: {node} {direction} by {abs(delta)}")
        if effects:
            lines.append("Downstream effects:")
            for effect in effects:
                lines.append(f"  - {effect}")
        else:
            lines.append("No downstream effects")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # format_explain
    # ------------------------------------------------------------------

    def format_explain(
        self,
        axiom_name: str,
        formula: str,
        domain: str,
        context: str = "",
    ) -> str:
        """Format an axiom explanation.

        Args:
            axiom_name: Name of the axiom being explained.
            formula: Formula or rule string.
            domain: Domain the axiom belongs to (e.g. "physics").
            context: Optional additional context / narrative text.

        Returns:
            Multi-line formatted string.
        """
        lines: list[str] = []
        lines.append(f"Axiom: {axiom_name}")
        lines.append(f"Formula: {formula}")
        lines.append(f"Domain: {domain}")
        if context:
            lines.append(f"Context: {context}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # format_compare
    # ------------------------------------------------------------------

    def format_compare(
        self,
        items: list[dict[str, Any]],
    ) -> str:
        """Format a comparison of multiple items.

        Each item dict should have at minimum a "name" key.  Optional keys are
        "value" and "unit".

        Args:
            items: List of item dicts to compare.

        Returns:
            Multi-line formatted string; "No items to compare" when empty.
        """
        if not items:
            return "No items to compare"
        lines: list[str] = ["Comparison:"]
        for item in items:
            name = item.get("name", "?")
            value = item.get("value", "")
            unit = item.get("unit", "")
            unit_str = f" {unit}" if unit else ""
            if value != "":
                lines.append(f"  {name}: {value}{unit_str}")
            else:
                lines.append(f"  {name}")
        return "\n".join(lines)
