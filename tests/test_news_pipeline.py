# tests/test_news_pipeline.py
"""Tests for CausalNewsExtractor + LiveNewsInjector."""
from __future__ import annotations

import time

import pytest

from mowm.news_pipeline import (
    CausalNewsExtractor,
    LiveNewsInjector,
    NewsEvent,
)
from mowm.reasoning.edge_types import EdgeType


class TestCausalNewsExtractor:
    def setup_method(self):
        self.ext = CausalNewsExtractor()

    def test_positive_verb(self):
        edges = self.ext.parse_headline(
            "Rate hikes boost bank profits.",
        )
        assert len(edges) >= 1
        assert edges[0].edge_type == EdgeType.INCREASES

    def test_negative_verb(self):
        edges = self.ext.parse_headline(
            "Tariffs crush tech stocks.",
        )
        assert len(edges) >= 1
        assert edges[0].edge_type == EdgeType.DECREASES

    def test_causal_verb(self):
        edges = self.ext.parse_headline(
            "Earthquake triggers landslide.",
        )
        assert len(edges) >= 1
        assert edges[0].edge_type == EdgeType.CAUSES

    def test_no_causal_verb(self):
        edges = self.ext.parse_headline(
            "The weather is nice today.",
        )
        assert edges == []

    def test_source_and_target_extracted(self):
        edges = self.ext.parse_headline(
            "Inflation increases consumer prices.",
        )
        assert len(edges) >= 1
        assert "inflation" in edges[0].source_id.lower()
        assert "consumer" in edges[0].target_id.lower() or \
               "prices" in edges[0].target_id.lower()

    def test_ephemeral_metadata(self):
        edges = self.ext.parse_headline(
            "Oil prices drop sharply.",
        )
        if edges:
            assert edges[0].metadata.get("ephemeral") is True
            assert "source_text" in edges[0].metadata

    def test_weight_below_one(self):
        edges = self.ext.parse_headline(
            "Interest rates reduce borrowing.",
        )
        if edges:
            assert edges[0].weight <= 0.9

    def test_parse_multiple(self):
        headlines = [
            "Fed raises interest rates.",
            "Tech stocks decline sharply.",
            "The weather is sunny.",
        ]
        events = self.ext.parse_multiple(headlines)
        assert len(events) == 3
        assert all(isinstance(e, NewsEvent) for e in events)
        # At least first two should have edges
        total_edges = sum(len(e.edges) for e in events)
        assert total_edges >= 1


class TestLiveNewsInjector:
    def setup_method(self):
        self.injector = LiveNewsInjector(decay_hours=1.0)

    def test_ingest_returns_edge_count(self):
        count = self.injector.ingest_headlines([
            "Inflation increases food prices.",
        ])
        assert count >= 1

    def test_event_count(self):
        self.injector.ingest_headlines([
            "Oil prices surge.",
            "Stocks decline.",
        ])
        assert self.injector.event_count == 2

    def test_edge_count(self):
        self.injector.ingest_headlines([
            "Inflation increases prices.",
        ])
        assert self.injector.edge_count >= 1

    def test_get_ephemeral_edges(self):
        self.injector.ingest_headlines([
            "Rates boost savings.",
        ])
        edges = self.injector.get_ephemeral_edges()
        assert len(edges) >= 1
        assert all(e.weight > 0 for e in edges)

    def test_edges_have_decayed_weight(self):
        self.injector.ingest_headlines([
            "Rates increase returns.",
        ])
        edges = self.injector.get_ephemeral_edges()
        if edges:
            # Just ingested — weight should be close to original
            assert edges[0].weight > 0.5

    def test_clear_expired(self):
        # Inject with a very short decay
        inj = LiveNewsInjector(decay_hours=0.0001)  # ~0.36 seconds
        inj.ingest_headlines(["Rates drop."])
        # Manually backdate the event
        if inj._cached_events:
            inj._cached_events[0].timestamp = time.time() - 3600
        removed = inj.clear_expired()
        assert removed == 1
        assert inj.event_count == 0

    def test_empty_ingest(self):
        count = self.injector.ingest_headlines([])
        assert count == 0
        assert self.injector.event_count == 0
