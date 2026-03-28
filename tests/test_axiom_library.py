# tests/test_axiom_library.py
"""Tests for axiom library — global world axiom storage and selection."""
from __future__ import annotations

import numpy as np

from mowm.axiom_library import Axiom, AxiomLibrary

K, L = 4, 32  # small dims for testing


class TestAxiom:
    def test_axiom_creation(self):
        vec = np.zeros((K, L), dtype=np.float32)
        vec[:, 0] = 1.0
        ax = Axiom(
            name="test_axiom",
            domain="test.basic",
            formula_str="x + 1",
            vector=vec,
            operators=["+"],
        )
        assert ax.name == "test_axiom"
        assert ax.domain == "test.basic"
        assert ax.vector.shape == (K, L)
        assert ax.operators == ["+"]


class TestAxiomLibrary:
    def setup_method(self):
        self.lib = AxiomLibrary(k=K, l=L)

    def test_register_returns_id(self):
        vec = self.lib.bc.random_discrete(seed=1)
        ax = Axiom("a1", "test", "x", vec, ["+"])
        aid = self.lib.register(ax)
        assert isinstance(aid, int)
        assert aid >= 0

    def test_register_increments_id(self):
        vec1 = self.lib.bc.random_discrete(seed=1)
        vec2 = self.lib.bc.random_discrete(seed=2)
        a1 = Axiom("a1", "test", "x", vec1, ["+"])
        a2 = Axiom("a2", "test", "y", vec2, ["+"])
        id1 = self.lib.register(a1)
        id2 = self.lib.register(a2)
        assert id2 == id1 + 1

    def test_select_returns_similar(self):
        vec1 = self.lib.bc.random_discrete(seed=10)
        ax1 = Axiom("target", "test", "x", vec1, ["+"])
        self.lib.register(ax1)
        for i in range(5):
            v = self.lib.bc.random_discrete(seed=100 + i)
            self.lib.register(Axiom(f"d{i}", "other", "y", v, ["*"]))
        results = self.lib.select(vec1, top_n=3)
        assert len(results) <= 3
        assert results[0].name == "target"

    def test_compose_binds_axioms(self):
        vec1 = self.lib.bc.random_discrete(seed=1)
        vec2 = self.lib.bc.random_discrete(seed=2)
        ax1 = Axiom("a1", "test", "x", vec1, ["+"])
        ax2 = Axiom("a2", "test", "y", vec2, ["+"])
        composed = self.lib.compose([ax1, ax2])
        assert composed.shape == (K, L)
        sim1 = self.lib.bc.similarity(composed, vec1)
        sim2 = self.lib.bc.similarity(composed, vec2)
        assert sim1 < 0.9 or sim2 < 0.9

    def test_remove(self):
        vec = self.lib.bc.random_discrete(seed=1)
        ax = Axiom("removable", "test", "x", vec, ["+"])
        aid = self.lib.register(ax)
        self.lib.remove(aid)
        results = self.lib.select(vec, top_n=10)
        names = [a.name for a in results]
        assert "removable" not in names

    def test_domains_dict(self):
        vec1 = self.lib.bc.random_discrete(seed=1)
        vec2 = self.lib.bc.random_discrete(seed=2)
        self.lib.register(Axiom("a1", "physics", "F=ma", vec1, ["+", "*"]))
        self.lib.register(Axiom("a2", "logic", "A->B", vec2, ["AND"]))
        d = self.lib.domains()
        assert "physics" in d
        assert "logic" in d
        assert len(d["physics"]) == 1
        assert len(d["logic"]) == 1

    def test_len(self):
        assert len(self.lib) == 0
        vec = self.lib.bc.random_discrete(seed=1)
        self.lib.register(Axiom("a", "t", "x", vec, []))
        assert len(self.lib) == 1

    def test_make_axiom_from_text(self):
        ax = self.lib.make_axiom(
            name="Newton2",
            domain="physics.dynamics",
            formula_str="F = m * a",
            operators=["+", "*"],
        )
        assert ax.vector.shape == (K, L)
        assert ax.name == "Newton2"
        ax2 = self.lib.make_axiom(
            name="Newton2",
            domain="physics.dynamics",
            formula_str="F = m * a",
            operators=["+", "*"],
        )
        sim = self.lib.bc.similarity(ax.vector, ax2.vector)
        assert sim > 0.99

    def test_select_empty_library(self):
        vec = self.lib.bc.random_discrete(seed=1)
        results = self.lib.select(vec, top_n=5)
        assert results == []
