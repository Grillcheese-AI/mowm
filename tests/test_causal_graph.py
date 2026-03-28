# tests/test_causal_graph.py
"""Tests for CausalGraph — signed directed graph with propagation."""
from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes

from mowm.causal_graph import CausalEdge, CausalGraph, CausalNode

K, L = 4, 32


class TestCausalNode:
    def test_creation(self):
        vec = np.zeros((K, L), dtype=np.float32)
        node = CausalNode(name="force", domain="physics", vector=vec)
        assert node.name == "force"
        assert node.domain == "physics"
        assert node.vector.shape == (K, L)


class TestCausalEdge:
    def test_creation_positive(self):
        edge = CausalEdge(source="force", target="acceleration", polarity="+", strength=0.9)
        assert edge.polarity == "+"
        assert edge.strength == 0.9

    def test_creation_negative(self):
        edge = CausalEdge(source="friction", target="velocity", polarity="-", strength=0.7)
        assert edge.polarity == "-"


class TestCausalGraph:
    def setup_method(self):
        self.bc = BlockCodes(k=K, l=L)
        self.g = CausalGraph(k=K, l=L)

    def test_add_node(self):
        self.g.add_node("force", "physics", self.bc.random_discrete(seed=1))
        assert "force" in self.g.nodes

    def test_add_edge(self):
        self.g.add_node("force", "physics", self.bc.random_discrete(seed=1))
        self.g.add_node("accel", "physics", self.bc.random_discrete(seed=2))
        self.g.add_edge("force", "accel", "+", 0.9)
        assert len(self.g.edges) == 1

    def test_add_edge_missing_node_raises(self):
        self.g.add_node("force", "physics", self.bc.random_discrete(seed=1))
        with pytest.raises(KeyError):
            self.g.add_edge("force", "nonexistent", "+", 0.5)

    def test_remove_node(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "+", 0.5)
        self.g.remove_node("a")
        assert "a" not in self.g.nodes
        assert len(self.g.edges) == 0

    def test_remove_edge(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "+", 0.5)
        self.g.remove_edge("a", "b")
        assert len(self.g.edges) == 0

    def test_propagate_positive(self):
        for i, n in enumerate(["a", "b", "c"]):
            self.g.add_node(n, "test", self.bc.random_discrete(seed=i))
        self.g.add_edge("a", "b", "+", 0.8)
        self.g.add_edge("b", "c", "+", 0.6)
        result = self.g.propagate("a", 1.0)
        assert result["b"] > 0
        assert result["c"] > 0

    def test_propagate_negative(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "-", 0.8)
        result = self.g.propagate("a", 1.0)
        assert result["b"] < 0

    def test_propagate_unknown_node_raises(self):
        with pytest.raises(KeyError):
            self.g.propagate("nonexistent", 1.0)

    def test_counterfactual(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "+", 1.0)
        result = self.g.counterfactual("a", 5.0)
        assert result["a"] == pytest.approx(5.0)
        assert "b" in result

    def test_find_paths_direct(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "+", 0.5)
        paths = self.g.find_paths("a", "b")
        assert len(paths) == 1
        assert paths[0] == ["a", "b"]

    def test_find_paths_indirect(self):
        for i, n in enumerate(["a", "b", "c"]):
            self.g.add_node(n, "test", self.bc.random_discrete(seed=i))
        self.g.add_edge("a", "b", "+", 0.5)
        self.g.add_edge("b", "c", "+", 0.5)
        paths = self.g.find_paths("a", "c")
        assert ["a", "b", "c"] in paths

    def test_find_paths_no_path(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        assert self.g.find_paths("a", "b") == []

    def test_feedback_loops_none(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "+", 0.5)
        assert self.g.feedback_loops() == []

    def test_feedback_loops_found(self):
        for i, n in enumerate(["a", "b", "c"]):
            self.g.add_node(n, "test", self.bc.random_discrete(seed=i))
        self.g.add_edge("a", "b", "+", 0.5)
        self.g.add_edge("b", "c", "+", 0.5)
        self.g.add_edge("c", "a", "+", 0.5)
        assert len(self.g.feedback_loops()) >= 1

    def test_encode_to_vsa(self):
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "+", 0.5)
        vec = self.g.encode_to_vsa()
        assert vec.shape == (K, L)
        assert vec.dtype == np.float32

    def test_encode_empty_graph(self):
        assert self.g.encode_to_vsa().shape == (K, L)

    def test_node_count(self):
        assert self.g.node_count == 0
        self.g.add_node("x", "test", self.bc.random_discrete(seed=1))
        assert self.g.node_count == 1

    def test_edge_count(self):
        assert self.g.edge_count == 0
        self.g.add_node("a", "test", self.bc.random_discrete(seed=1))
        self.g.add_node("b", "test", self.bc.random_discrete(seed=2))
        self.g.add_edge("a", "b", "+", 0.5)
        assert self.g.edge_count == 1
