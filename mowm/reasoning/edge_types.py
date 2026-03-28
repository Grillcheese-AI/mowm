# mowm/reasoning/edge_types.py
"""Semantic edge types for the axiom knowledge graph.

Defines typed relationships between axiom/concept nodes (EdgeType),
mathematical sign for causal path tracing (EffectSign), and the
SemanticEdge dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(Enum):
    """Semantic relationship types between axiom/concept nodes."""

    INCREASES = "INCREASES"
    DECREASES = "DECREASES"
    CAUSES = "CAUSES"
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    SPECIAL_CASE_OF = "SPECIAL_CASE_OF"
    PROPORTIONAL_TO = "PROPORTIONAL_TO"
    INVERSELY_PROPORTIONAL = "INVERSELY_PROPORTIONAL"
    DERIVATIVE_OF = "DERIVATIVE_OF"
    INTEGRAL_OF = "INTEGRAL_OF"
    SHARES_VARIABLE = "SHARES_VARIABLE"
    SIMILAR_TO_VEC = "SIMILAR_TO_VEC"


class EffectSign(Enum):
    """Mathematical sign for causal path tracing."""

    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0


EDGE_TO_SIGN: dict[EdgeType, EffectSign] = {
    EdgeType.INCREASES: EffectSign.POSITIVE,
    EdgeType.PROPORTIONAL_TO: EffectSign.POSITIVE,
    EdgeType.CAUSES: EffectSign.POSITIVE,
    EdgeType.DECREASES: EffectSign.NEGATIVE,
    EdgeType.INVERSELY_PROPORTIONAL: EffectSign.NEGATIVE,
}


@dataclass
class SemanticEdge:
    """A typed, weighted edge in the axiom knowledge graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
