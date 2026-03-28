# tests/test_base.py
"""Tests for MoWM abstract base classes."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.base import (
    BaseDomain,
    BaseWorld,
    BaseRouter,
    CleanupStrategy,
    DiscoveryObserver,
    MergeStrategy,
    SearchStrategy,
    WorldFactory,
)


class TestABCsNotInstantiable:
    def test_base_world(self):
        with pytest.raises(TypeError):
            BaseWorld()

    def test_base_router(self):
        with pytest.raises(TypeError):
            BaseRouter()

    def test_base_domain(self):
        with pytest.raises(TypeError):
            BaseDomain()

    def test_merge_strategy(self):
        with pytest.raises(TypeError):
            MergeStrategy()

    def test_search_strategy(self):
        with pytest.raises(TypeError):
            SearchStrategy()

    def test_cleanup_strategy(self):
        with pytest.raises(TypeError):
            CleanupStrategy()

    def test_world_factory(self):
        with pytest.raises(TypeError):
            WorldFactory()

    def test_discovery_observer(self):
        with pytest.raises(TypeError):
            DiscoveryObserver()


class TestConcreteSubclasses:
    def test_merge_strategy_subclass(self):
        class MyMerge(MergeStrategy):
            def merge(self, deltas, weights):
                return deltas[0]

        m = MyMerge()
        dummy = np.zeros((4, 32), dtype=np.float32)
        result = m.merge([dummy, dummy], [0.5, 0.5])
        assert result.shape == (4, 32)

    def test_cleanup_strategy_subclass(self):
        class MyCleanup(CleanupStrategy):
            def cleanup(self, vec):
                return vec

        c = MyCleanup()
        dummy = np.zeros((4, 32), dtype=np.float32)
        assert c.cleanup(dummy).shape == (4, 32)

    def test_search_strategy_subclass(self):
        class MySearch(SearchStrategy):
            def search(self, inputs, outputs, operators, max_complexity):
                return "x + 1", 0.01

        s = MySearch()
        formula, error = s.search(
            np.array([[1.0]]), np.array([2.0]), ["+"], 10
        )
        assert formula == "x + 1"

    def test_discovery_observer_subclass(self):
        class MyObs(DiscoveryObserver):
            def __init__(self):
                self.notified = False

            def on_axiom_discovered(self, axiom):
                self.notified = True

        obs = MyObs()
        obs.on_axiom_discovered(None)
        assert obs.notified

    def test_base_domain_template_method(self):
        class MyDomain(BaseDomain):
            def __init__(self):
                self.registered = False

            @property
            def name(self) -> str:
                return "test"

            @property
            def operators(self) -> list[str]:
                return ["+"]

            def _register_axioms(self, library):
                self.registered = True

        d = MyDomain()
        d.seed(library=None)
        assert d.registered
