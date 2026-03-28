# tests/test_discovery.py
"""Tests for DiscoveryLoop — symbolic regression and axiom distillation."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.discovery import DiscoveryLoop, GeneticProgrammingSR

K, L = 4, 32


class TestGeneticProgrammingSR:
    def test_discovers_linear(self):
        sr = GeneticProgrammingSR(population_size=50, generations=30, seed=42)
        x = np.linspace(0, 5, 20).reshape(-1, 1)
        y = 2 * x.ravel() + 1
        formula, mse = sr.search(inputs=x, outputs=y, operators=["+", "*"], max_complexity=6)
        assert isinstance(formula, str)
        assert isinstance(mse, float)
        assert mse < 5.0

    def test_discovers_quadratic(self):
        sr = GeneticProgrammingSR(population_size=80, generations=40, seed=123)
        x = np.linspace(-3, 3, 25).reshape(-1, 1)
        y = x.ravel() ** 2
        formula, mse = sr.search(inputs=x, outputs=y, operators=["+", "*", "square"], max_complexity=8)
        assert isinstance(formula, str)
        assert mse < 10.0

    def test_returns_best_of_population(self):
        sr = GeneticProgrammingSR(population_size=20, generations=10, seed=42)
        x = np.array([[1], [2], [3]], dtype=np.float64)
        y = np.array([2, 4, 6], dtype=np.float64)
        formula, mse = sr.search(x, y, ["+", "*"], 5)
        assert formula is not None


class TestDiscoveryLoop:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)
        for i in range(5):
            ax = self.lib.make_axiom(f"base_{i}", "test", f"x + {i}", ["+"])
            self.lib.register(ax)

    def test_check_residual(self):
        loop = DiscoveryLoop(self.lib, k=K, l=L)
        bc = self.lib.bc
        a = bc.random_discrete(seed=1)
        b = bc.random_discrete(seed=2)
        residual = loop.check_residual(a, b)
        assert isinstance(residual, float)
        assert residual >= 0.0

    def test_check_residual_identical(self):
        loop = DiscoveryLoop(self.lib, k=K, l=L)
        bc = self.lib.bc
        a = bc.random_discrete(seed=1)
        residual = loop.check_residual(a, a)
        assert residual == pytest.approx(0.0, abs=1e-6)

    def test_discover_returns_axiom_or_none(self):
        loop = DiscoveryLoop(self.lib, k=K, l=L, sr_operators=["+", "*"], max_complexity=6)
        bc = self.lib.bc
        inputs = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
        outputs = np.array([3, 5, 7, 9, 11], dtype=np.float64)
        context = bc.random_discrete(seed=42)
        result = loop.discover(inputs, outputs, context)
        if result is not None:
            assert result.vector.shape == (K, L)
            assert result.domain == "discovered"

    def test_discover_registers_in_library(self):
        loop = DiscoveryLoop(self.lib, k=K, l=L, sr_operators=["+", "*"], max_complexity=6)
        initial_count = len(self.lib)
        inputs = np.array([[1], [2], [3], [4]], dtype=np.float64)
        outputs = np.array([2, 4, 6, 8], dtype=np.float64)
        context = self.lib.bc.random_discrete(seed=42)
        result = loop.discover(inputs, outputs, context)
        if result is not None:
            assert len(self.lib) == initial_count + 1

    def test_observer_notified(self):
        from mowm.base import DiscoveryObserver

        class MockObserver(DiscoveryObserver):
            def __init__(self):
                self.was_notified = False
                self.axiom = None

            def on_axiom_discovered(self, axiom):
                self.was_notified = True
                self.axiom = axiom

        obs = MockObserver()
        loop = DiscoveryLoop(
            self.lib, k=K, l=L,
            sr_operators=["+", "*"],
            max_complexity=6,
            observers=[obs],
        )
        inputs = np.array([[1], [2], [3], [4]], dtype=np.float64)
        outputs = np.array([2, 4, 6, 8], dtype=np.float64)
        context = self.lib.bc.random_discrete(seed=42)
        result = loop.discover(inputs, outputs, context)
        if result is not None:
            assert obs.was_notified
            assert obs.axiom is result

    def test_cross_domain_analogy(self):
        loop = DiscoveryLoop(self.lib, k=K, l=L)
        bc = self.lib.bc
        domain_a = bc.random_discrete(seed=100)
        domain_b = bc.random_discrete(seed=200)
        anomaly = bc.random_discrete(seed=300)
        analogy = loop.cross_domain_analogy(anomaly, domain_a, domain_b)
        assert analogy.shape == (K, L)
        sim = bc.similarity(analogy, anomaly)
        assert sim < 0.9
