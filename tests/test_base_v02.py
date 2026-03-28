# tests/test_base_v02.py
"""Tests for v0.2.0 ABC extensions in base.py."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.base import BaseCausalGraph, BaseExecutor, BaseLogicEngine

K, L = 4, 32


class ConcreteExecutor(BaseExecutor):
    def execute(self, formula_str: str, inputs: dict[str, float]) -> float:
        return 42.0

    def parse(self, formula_str: str):
        return None


class ConcreteCausalGraph(BaseCausalGraph):
    def add_node(self, name: str, domain: str, vector: np.ndarray) -> None:
        pass

    def add_edge(
        self, source: str, target: str, polarity: str, strength: float
    ) -> None:
        pass

    def remove_node(self, name: str) -> None:
        pass

    def remove_edge(self, source: str, target: str) -> None:
        pass

    def propagate(self, node: str, delta: float) -> dict[str, float]:
        return {}

    def counterfactual(self, node: str, value: float) -> dict[str, float]:
        return {}

    def find_paths(self, source: str, target: str) -> list[list[str]]:
        return []

    def feedback_loops(self) -> list[list[str]]:
        return []

    def encode_to_vsa(self) -> np.ndarray:
        return np.zeros((K, L), dtype=np.float32)


class ConcreteLogicEngine(BaseLogicEngine):
    def evaluate(
        self, formula_str: str, facts: dict[str, bool | float], mode: str | None = None
    ) -> bool | float:
        return True

    def check_obligation(
        self,
        conditions: list[str],
        facts: dict[str, bool],
        exceptions: list[str] | None = None,
    ) -> bool:
        return True


class TestBaseExecutor:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseExecutor()

    def test_concrete_executor(self):
        ex = ConcreteExecutor()
        result = ex.execute("F = m * a", {"m": 10.0, "a": 9.8})
        assert result == 42.0


class TestBaseCausalGraph:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseCausalGraph()

    def test_concrete_causal_graph(self):
        g = ConcreteCausalGraph()
        vec = np.zeros((K, L), dtype=np.float32)
        g.add_node("x", "test", vec)
        g.add_edge("x", "y", "+", 0.8)
        result = g.propagate("x", 1.0)
        assert isinstance(result, dict)


class TestBaseLogicEngine:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLogicEngine()

    def test_concrete_logic_engine(self):
        le = ConcreteLogicEngine()
        result = le.evaluate("A AND B", {"A": True, "B": True})
        assert result is True
