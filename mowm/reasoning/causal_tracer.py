# mowm/reasoning/causal_tracer.py
"""CausalTracer — trace causal paths and detect variable cancellations."""
from __future__ import annotations

from dataclasses import dataclass, field

from mowm.reasoning.edge_types import EDGE_TO_SIGN, EffectSign, SemanticEdge


@dataclass
class CausalNarrative:
    """Result of causal tracing for an explanation."""

    cause: str = ""
    effects: list[tuple[str, str, float]] = field(default_factory=list)
    cancellations: list[tuple[str, str]] = field(default_factory=list)
    net_effect: str = ""
    graph_path: list[str] = field(default_factory=list)


class CausalTracer:
    """Trace causal paths and detect variable cancellations.

    Uses EffectSign arithmetic: multiply signs along each path.
    POSITIVE * NEGATIVE = NEGATIVE. If both positive and negative
    paths exist to the same target -> cancellation detected.
    """

    def __init__(self, adjacency: dict[str, list[SemanticEdge]] | None = None) -> None:
        self._adj = adjacency or {}

    def set_adjacency(self, adj: dict[str, list[SemanticEdge]]) -> None:
        self._adj = adj

    def _get_edge_effect(self, edge: SemanticEdge) -> EffectSign:
        return EDGE_TO_SIGN.get(edge.edge_type, EffectSign.NEUTRAL)

    def trace_paths(
        self,
        source: str,
        target: str,
    ) -> list[tuple[list[str], EffectSign]]:
        """Find all paths from source to target with net effect sign."""
        paths: list[tuple[list[str], EffectSign]] = []
        stack: list[tuple[str, list[str], EffectSign]] = [
            (source, [source], EffectSign.POSITIVE)
        ]

        while stack:
            current, path, net_effect = stack.pop()
            if current == target and len(path) > 1:
                paths.append((path, net_effect))
                continue
            if len(path) > 8:
                continue

            for edge in self._adj.get(current, []):
                if edge.target_id not in path:
                    edge_effect = self._get_edge_effect(edge)
                    if edge_effect == EffectSign.NEUTRAL:
                        new_effect = net_effect
                    else:
                        new_val = net_effect.value * edge_effect.value
                        new_effect = EffectSign(new_val) if new_val != 0 else EffectSign.NEUTRAL
                    stack.append((edge.target_id, path + [edge.target_id], new_effect))

        return paths

    def find_cancellation(self, source: str, target: str) -> tuple[str, str] | None:
        """Detect if source cancels out when affecting target via opposing paths."""
        paths = self.trace_paths(source, target)
        has_pos = any(eff == EffectSign.POSITIVE for _, eff in paths)
        has_neg = any(eff == EffectSign.NEGATIVE for _, eff in paths)
        if has_pos and has_neg:
            return (source, target)
        return None

    def trace_explanation(
        self,
        keywords: list[str],
        adjacency: dict[str, list[SemanticEdge]] | None = None,
    ) -> CausalNarrative:
        """Trace causal relationships between keyword concepts."""
        adj = adjacency or self._adj
        if not adj or not keywords:
            return CausalNarrative()

        narrative = CausalNarrative()
        kw_lower = [k.lower() for k in keywords]
        nodes_in_graph = set(adj.keys())
        for edges in adj.values():
            for e in edges:
                nodes_in_graph.add(e.target_id)

        relevant = [k for k in kw_lower if k in nodes_in_graph]
        if not relevant:
            return narrative

        narrative.cause = relevant[0]
        narrative.graph_path = relevant

        for node in relevant:
            for edge in adj.get(node, []):
                effect = self._get_edge_effect(edge)
                direction = "increases" if effect == EffectSign.POSITIVE else (
                    "decreases" if effect == EffectSign.NEGATIVE else "affects"
                )
                narrative.effects.append((edge.target_id, direction, edge.weight))

        # Use the resolved adj for cancellation detection (not self._adj)
        saved = self._adj
        self._adj = adj
        try:
            for i, src in enumerate(relevant):
                for tgt in relevant[i + 1:]:
                    cancel = self.find_cancellation(src, tgt)
                    if cancel:
                        narrative.cancellations.append(cancel)
        finally:
            self._adj = saved

        if narrative.cancellations:
            narrative.net_effect = "cancellation detected"
        elif narrative.effects:
            narrative.net_effect = "net effect propagated"
        else:
            narrative.net_effect = "no causal effects found"

        return narrative
