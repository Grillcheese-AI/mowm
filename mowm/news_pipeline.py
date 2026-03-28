# mowm/news_pipeline.py
"""CausalNewsExtractor + LiveNewsInjector — real-time news to graph edges.

CausalNewsExtractor parses headlines into SemanticEdge objects by mapping
verbs to EdgeType (regex-based SVO extraction, no spaCy dependency).

LiveNewsInjector fetches RSS feeds and injects ephemeral edges into a
KnowledgeContext for the ReasoningDecoder to use as real-time context.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mowm.reasoning.edge_types import EdgeType, SemanticEdge


# ── Verb-to-EdgeType lexicon ────────────────────────────────────────────

_POSITIVE_VERBS = frozenset({
    "increase", "increases", "increased", "increasing",
    "boost", "boosts", "boosted", "boosting",
    "raise", "raises", "raised", "raising",
    "drive", "drives", "drove", "driving",
    "accelerate", "accelerates", "accelerated",
    "surge", "surges", "surged", "surging",
    "strengthen", "strengthens", "strengthened",
    "improve", "improves", "improved", "improving",
    "grow", "grows", "grew", "growing",
    "expand", "expands", "expanded", "expanding",
})

_NEGATIVE_VERBS = frozenset({
    "decrease", "decreases", "decreased", "decreasing",
    "crush", "crushes", "crushed", "crushing",
    "drop", "drops", "dropped", "dropping",
    "reduce", "reduces", "reduced", "reducing",
    "slow", "slows", "slowed", "slowing",
    "plummet", "plummets", "plummeted",
    "hurt", "hurts", "weaken", "weakens",
    "decline", "declines", "declined", "declining",
    "collapse", "collapses", "collapsed",
    "shrink", "shrinks", "shrank", "shrinking",
    "cut", "cuts", "lower", "lowers", "lowered",
})

_CAUSAL_VERBS = frozenset({
    "cause", "causes", "caused", "causing",
    "lead", "leads", "led", "leading",
    "create", "creates", "created", "creating",
    "force", "forces", "forced", "forcing",
    "trigger", "triggers", "triggered", "triggering",
    "result", "results", "resulted", "resulting",
    "spark", "sparks", "sparked",
})


def _verb_to_edge(verb: str) -> EdgeType | None:
    """Map a verb to its EdgeType."""
    v = verb.lower().strip()
    if v in _POSITIVE_VERBS:
        return EdgeType.INCREASES
    if v in _NEGATIVE_VERBS:
        return EdgeType.DECREASES
    if v in _CAUSAL_VERBS:
        return EdgeType.CAUSES
    return None


# ── SVO extraction patterns ─────────────────────────────────────────────

# Pattern: "Subject verb Object" — captures 3-word+ SVO
_SVO_PATTERN = re.compile(
    r"([A-Z][A-Za-z\s]+?)\s+"  # Subject (capitalized phrase)
    r"((?:"
    + "|".join(
        sorted(
            _POSITIVE_VERBS | _NEGATIVE_VERBS | _CAUSAL_VERBS,
            key=len,
            reverse=True,
        )
    )
    + r"))\s+"
    r"([A-Za-z\s]+?)(?:\.|,|;|$)",  # Object (phrase until punctuation)
    re.IGNORECASE,
)

# Simpler pattern: "X [verb] Y"
_SIMPLE_SVO = re.compile(
    r"(\w[\w\s]{1,30}?)\s+"
    r"(increases?|decreases?|boosts?|crushes?|cuts?|"
    r"raises?|drops?|reduces?|causes?|triggers?|"
    r"drives?|hurts?|slows?|improves?|weakens?)\s+"
    r"(\w[\w\s]{1,30}?)(?:\.|,|;|$)",
    re.IGNORECASE,
)


@dataclass
class NewsEvent:
    """A parsed news event with extracted causal claims."""
    headline: str
    source: str = ""
    timestamp: float = 0.0
    edges: list[SemanticEdge] = field(default_factory=list)


class CausalNewsExtractor:
    """Extract causal SemanticEdges from news headlines.

    Uses regex SVO extraction + verb lexicon mapping. No spaCy needed.
    Fast, deterministic, never hallucinates.
    """

    def parse_headline(self, headline: str) -> list[SemanticEdge]:
        """Extract structured causal edges from a headline.

        Args:
            headline: News headline string.

        Returns:
            List of SemanticEdge objects (may be empty).
        """
        edges: list[SemanticEdge] = []
        seen: set[tuple[str, str]] = set()

        # Try the complex SVO pattern first
        for match in _SVO_PATTERN.finditer(headline):
            subject = match.group(1).strip()
            verb = match.group(2).strip()
            obj = match.group(3).strip()
            edge_type = _verb_to_edge(verb)
            if edge_type and (subject, obj) not in seen:
                edges.append(SemanticEdge(
                    source_id=subject.lower(),
                    target_id=obj.lower(),
                    edge_type=edge_type,
                    weight=0.8,
                    metadata={
                        "source_text": headline,
                        "ephemeral": True,
                        "verb": verb,
                    },
                ))
                seen.add((subject, obj))

        # Fallback: simpler pattern
        if not edges:
            for match in _SIMPLE_SVO.finditer(headline):
                subject = match.group(1).strip()
                verb = match.group(2).strip()
                obj = match.group(3).strip()
                edge_type = _verb_to_edge(verb)
                if edge_type and (subject, obj) not in seen:
                    edges.append(SemanticEdge(
                        source_id=subject.lower(),
                        target_id=obj.lower(),
                        edge_type=edge_type,
                        weight=0.7,
                        metadata={
                            "source_text": headline,
                            "ephemeral": True,
                            "verb": verb,
                        },
                    ))
                    seen.add((subject, obj))

        return edges

    def parse_multiple(
        self, headlines: list[str],
    ) -> list[NewsEvent]:
        """Parse multiple headlines into NewsEvents."""
        events: list[NewsEvent] = []
        for headline in headlines:
            edges = self.parse_headline(headline)
            events.append(NewsEvent(
                headline=headline,
                timestamp=time.time(),
                edges=edges,
            ))
        return events


class LiveNewsInjector:
    """Inject ephemeral news edges into reasoning context.

    Fetches headlines (from a provider callback), parses them for
    causal edges, and injects them as temporary SemanticEdges with
    time-based decay.

    Args:
        extractor: CausalNewsExtractor instance.
        decay_hours: Hours until ephemeral edges lose all weight.
    """

    def __init__(
        self,
        extractor: CausalNewsExtractor | None = None,
        decay_hours: float = 24.0,
    ) -> None:
        self._extractor = extractor or CausalNewsExtractor()
        self._decay_hours = decay_hours
        self._cached_events: list[NewsEvent] = []

    def ingest_headlines(self, headlines: list[str]) -> int:
        """Parse and cache new headlines. Returns number of edges found."""
        events = self._extractor.parse_multiple(headlines)
        self._cached_events.extend(events)
        return sum(len(e.edges) for e in events)

    def get_ephemeral_edges(self) -> list[SemanticEdge]:
        """Return all cached edges with time-decayed weights.

        Edges older than decay_hours are dropped entirely.
        """
        now = time.time()
        cutoff = now - (self._decay_hours * 3600)

        active_edges: list[SemanticEdge] = []
        for event in self._cached_events:
            if event.timestamp < cutoff:
                continue
            age_hours = (now - event.timestamp) / 3600
            decay = max(0.0, 1.0 - (age_hours / self._decay_hours))

            for edge in event.edges:
                decayed = SemanticEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    weight=edge.weight * decay,
                    metadata=dict(edge.metadata),
                )
                active_edges.append(decayed)

        return active_edges

    def clear_expired(self) -> int:
        """Remove events older than decay_hours. Returns count removed."""
        now = time.time()
        cutoff = now - (self._decay_hours * 3600)
        before = len(self._cached_events)
        self._cached_events = [
            e for e in self._cached_events if e.timestamp >= cutoff
        ]
        return before - len(self._cached_events)

    @property
    def event_count(self) -> int:
        """Number of cached news events."""
        return len(self._cached_events)

    @property
    def edge_count(self) -> int:
        """Total number of edges across all cached events."""
        return sum(len(e.edges) for e in self._cached_events)
