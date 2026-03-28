# tests/test_causal_tracer.py
"""Tests for CausalTracer — EffectSign arithmetic and cancellation detection."""
from __future__ import annotations

import pytest

from mowm.reasoning.causal_tracer import CausalNarrative, CausalTracer
from mowm.reasoning.edge_types import EdgeType, EffectSign, SemanticEdge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_edge(src: str, tgt: str, etype: EdgeType, weight: float = 1.0) -> SemanticEdge:
    return SemanticEdge(source_id=src, target_id=tgt, edge_type=etype, weight=weight)


def make_adj(*edges: SemanticEdge) -> dict[str, list[SemanticEdge]]:
    adj: dict[str, list[SemanticEdge]] = {}
    for e in edges:
        adj.setdefault(e.source_id, []).append(e)
    return adj


# ---------------------------------------------------------------------------
# Tests: trace_paths
# ---------------------------------------------------------------------------


def test_positive_path_direct(tracer_empty):
    """Direct INCREASES edge: source -[+]-> target."""
    adj = make_adj(make_edge("force", "acceleration", EdgeType.INCREASES))
    tracer_empty.set_adjacency(adj)
    paths = tracer_empty.trace_paths("force", "acceleration")
    assert len(paths) == 1
    path, sign = paths[0]
    assert path == ["force", "acceleration"]
    assert sign == EffectSign.POSITIVE


def test_negative_path_direct(tracer_empty):
    """Direct DECREASES edge: source -[-]-> target."""
    adj = make_adj(make_edge("friction", "velocity", EdgeType.DECREASES))
    tracer_empty.set_adjacency(adj)
    paths = tracer_empty.trace_paths("friction", "velocity")
    assert len(paths) == 1
    _, sign = paths[0]
    assert sign == EffectSign.NEGATIVE


def test_sign_multiplication_pos_times_neg(tracer_empty):
    """Chained path: A -[+]-> B -[-]-> C gives net NEGATIVE."""
    adj = make_adj(
        make_edge("A", "B", EdgeType.INCREASES),
        make_edge("B", "C", EdgeType.DECREASES),
    )
    tracer_empty.set_adjacency(adj)
    paths = tracer_empty.trace_paths("A", "C")
    assert len(paths) == 1
    _, sign = paths[0]
    assert sign == EffectSign.NEGATIVE


def test_no_path_returns_empty(tracer_empty):
    """No path between disconnected nodes returns empty list."""
    adj = make_adj(make_edge("X", "Y", EdgeType.CAUSES))
    tracer_empty.set_adjacency(adj)
    paths = tracer_empty.trace_paths("X", "Z")
    assert paths == []


def test_empty_graph_returns_empty_paths(tracer_empty):
    """Empty adjacency returns empty paths."""
    paths = tracer_empty.trace_paths("a", "b")
    assert paths == []


# ---------------------------------------------------------------------------
# Tests: find_cancellation
# ---------------------------------------------------------------------------


def test_cancellation_detected_opposing_paths(tracer_empty):
    """mass -> acceleration via force (positive) and via inertia (negative) = cancellation."""
    adj = make_adj(
        make_edge("mass", "force", EdgeType.PROPORTIONAL_TO),
        make_edge("force", "acceleration", EdgeType.INCREASES),
        make_edge("mass", "inertia", EdgeType.PROPORTIONAL_TO),
        make_edge("inertia", "acceleration", EdgeType.DECREASES),
    )
    tracer_empty.set_adjacency(adj)
    result = tracer_empty.find_cancellation("mass", "acceleration")
    assert result == ("mass", "acceleration")


def test_no_cancellation_single_path(tracer_empty):
    """Single-sign path: no cancellation."""
    adj = make_adj(
        make_edge("heat", "temperature", EdgeType.INCREASES),
        make_edge("temperature", "pressure", EdgeType.PROPORTIONAL_TO),
    )
    tracer_empty.set_adjacency(adj)
    result = tracer_empty.find_cancellation("heat", "pressure")
    assert result is None


def test_no_cancellation_no_path(tracer_empty):
    """Disconnected nodes: find_cancellation returns None."""
    adj = make_adj(make_edge("A", "B", EdgeType.CAUSES))
    tracer_empty.set_adjacency(adj)
    result = tracer_empty.find_cancellation("A", "Z")
    assert result is None


# ---------------------------------------------------------------------------
# Tests: trace_explanation
# ---------------------------------------------------------------------------


def test_trace_explanation_basic(tracer_empty):
    """Keywords present in graph return a narrative with effects."""
    adj = make_adj(
        make_edge("force", "acceleration", EdgeType.INCREASES),
        make_edge("force", "momentum", EdgeType.PROPORTIONAL_TO),
    )
    narrative = tracer_empty.trace_explanation(["force", "acceleration"], adjacency=adj)
    assert narrative.cause == "force"
    assert any(t[0] == "acceleration" for t in narrative.effects)
    assert narrative.net_effect == "net effect propagated"


def test_trace_explanation_empty_keywords():
    """Empty keywords returns empty narrative."""
    tracer = CausalTracer()
    narrative = tracer.trace_explanation([])
    assert narrative.cause == ""
    assert narrative.effects == []


def test_trace_explanation_cancellation_narrative(tracer_empty):
    """Keywords with cancellation paths surface in narrative.cancellations."""
    adj = make_adj(
        make_edge("mass", "force", EdgeType.PROPORTIONAL_TO),
        make_edge("force", "accel", EdgeType.INCREASES),
        make_edge("mass", "inertia", EdgeType.PROPORTIONAL_TO),
        make_edge("inertia", "accel", EdgeType.DECREASES),
    )
    narrative = tracer_empty.trace_explanation(["mass", "accel"], adjacency=adj)
    assert narrative.net_effect == "cancellation detected"
    assert len(narrative.cancellations) >= 1


def test_trace_explanation_keywords_not_in_graph(tracer_empty):
    """Keywords absent from graph return empty narrative (no crash)."""
    adj = make_adj(make_edge("X", "Y", EdgeType.CAUSES))
    narrative = tracer_empty.trace_explanation(["unknown_concept"], adjacency=adj)
    assert narrative.cause == ""
    assert narrative.effects == []


def test_feedback_loop_no_infinite_loop(tracer_empty):
    """Circular graph does not cause infinite loop (path length limit)."""
    adj = make_adj(
        make_edge("A", "B", EdgeType.INCREASES),
        make_edge("B", "C", EdgeType.INCREASES),
        make_edge("C", "A", EdgeType.INCREASES),
    )
    tracer_empty.set_adjacency(adj)
    # Should terminate; result may have 0 or 1 path depending on cycle handling
    paths = tracer_empty.trace_paths("A", "A")
    assert isinstance(paths, list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracer_empty() -> CausalTracer:
    return CausalTracer()
