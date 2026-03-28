# mowm/discovery.py
"""DiscoveryLoop — universal 3-step discovery via symbolic regression.

When prediction residuals are high, the loop:
  1. Semantic mapping: find nearest axioms, cross-domain analogy
  2. Symbolic hypothesis: genetic programming SR over operator grammar
  3. Axiom distillation: encode discovered formula as block-code, register

The SR implementation uses a simple genetic programming loop over an
operator grammar (no PySR dependency). Population = expression trees
evaluated against training data.
"""
from __future__ import annotations

import operator as op_module
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from cubemind.ops.block_codes import BlockCodes

from mowm.base import DiscoveryObserver, SearchStrategy

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary


_SAFE_OPS: dict[str, Callable] = {
    "+": op_module.add,
    "-": op_module.sub,
    "*": op_module.mul,
    "/": lambda a, b: np.where(np.abs(b) > 1e-10, a / np.where(np.abs(b) > 1e-10, b, 1.0), a),
    "sin": lambda a, _=None: np.sin(a),
    "cos": lambda a, _=None: np.cos(a),
    "sqrt": lambda a, _=None: np.sqrt(np.abs(a)),
    "square": lambda a, _=None: a ** 2,
    "exp": lambda a, _=None: np.clip(np.exp(np.clip(a, -10, 10)), -1e6, 1e6),
    "log": lambda a, _=None: np.log(np.abs(a) + 1e-10),
    "AND": lambda a, b: np.where((a > 0.5) & (b > 0.5), 1.0, 0.0),
    "OR": lambda a, b: np.where((a > 0.5) | (b > 0.5), 1.0, 0.0),
    "NOT": lambda a, _=None: np.where(a > 0.5, 0.0, 1.0),
    "IMPLIES": lambda a, b: np.where((a > 0.5) & (b <= 0.5), 0.0, 1.0),
}

_UNARY_OPS = {"sin", "cos", "sqrt", "square", "exp", "log", "NOT"}
_BINARY_OPS = {"+", "-", "*", "/", "AND", "OR", "IMPLIES"}


@dataclass
class ExprNode:
    """A node in an expression tree."""

    op: str | None = None
    value: float | None = None
    var_idx: int | None = None
    left: ExprNode | None = None
    right: ExprNode | None = None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.op is None:
            if self.var_idx is not None:
                idx = min(self.var_idx, x.shape[1] - 1)
                return x[:, idx].astype(np.float64)
            return np.full(x.shape[0], self.value or 0.0, dtype=np.float64)

        fn = _SAFE_OPS.get(self.op)
        if fn is None:
            return np.zeros(x.shape[0], dtype=np.float64)

        left_val = self.left.evaluate(x) if self.left else np.zeros(
            x.shape[0], dtype=np.float64
        )

        if self.op in _UNARY_OPS:
            try:
                result = fn(left_val)
            except Exception:
                result = np.zeros_like(left_val)
        else:
            right_val = self.right.evaluate(x) if self.right else np.zeros(
                x.shape[0], dtype=np.float64
            )
            try:
                result = fn(left_val, right_val)
            except Exception:
                result = np.zeros_like(left_val)

        return np.clip(np.nan_to_num(result, nan=0.0), -1e6, 1e6)

    def complexity(self) -> int:
        count = 1
        if self.left:
            count += self.left.complexity()
        if self.right:
            count += self.right.complexity()
        return count

    def to_string(self) -> str:
        if self.op is None:
            if self.var_idx is not None:
                return f"x{self.var_idx}"
            return f"{self.value:.4g}" if self.value is not None else "0"

        if self.op in _UNARY_OPS:
            left_str = self.left.to_string() if self.left else "0"
            return f"{self.op}({left_str})"

        left_str = self.left.to_string() if self.left else "0"
        right_str = self.right.to_string() if self.right else "0"
        return f"({left_str} {self.op} {right_str})"


class GeneticProgrammingSR(SearchStrategy):
    """Symbolic regression via genetic programming over expression trees."""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        tournament_size: int = 5,
        crossover_rate: float = 0.7,
        max_tree_depth: int = 4,
        complexity_penalty: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.max_tree_depth = max_tree_depth
        self.complexity_penalty = complexity_penalty
        self._rng = np.random.default_rng(seed)

    def _random_tree(
        self, operators: list[str], n_features: int,
        max_depth: int, depth: int = 0,
    ) -> ExprNode:
        if depth >= max_depth or (depth > 0 and self._rng.random() < 0.3):
            if self._rng.random() < 0.6:
                return ExprNode(var_idx=int(self._rng.integers(0, n_features)))
            return ExprNode(value=float(self._rng.uniform(-5, 5)))

        op = operators[int(self._rng.integers(0, len(operators)))]
        left = self._random_tree(operators, n_features, max_depth, depth + 1)

        if op in _UNARY_OPS:
            return ExprNode(op=op, left=left)

        right = self._random_tree(operators, n_features, max_depth, depth + 1)
        return ExprNode(op=op, left=left, right=right)

    def _fitness(
        self, tree: ExprNode, inputs: np.ndarray,
        outputs: np.ndarray, max_complexity: int,
    ) -> float:
        comp = tree.complexity()
        if comp > max_complexity:
            return 1e9

        try:
            predicted = tree.evaluate(inputs)
            mse = float(np.mean((predicted - outputs) ** 2))
        except Exception:
            return 1e9

        if not np.isfinite(mse):
            return 1e9

        return mse + self.complexity_penalty * comp

    def _tournament_select(
        self, population: list[ExprNode], fitnesses: list[float],
    ) -> ExprNode:
        indices = self._rng.integers(0, len(population), size=self.tournament_size)
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitnesses[int(idx)] < fitnesses[int(best_idx)]:
                best_idx = idx
        return population[int(best_idx)]

    def _mutate(
        self, tree: ExprNode, operators: list[str], n_features: int,
    ) -> ExprNode:
        if self._rng.random() < 0.3 or tree.op is None:
            return self._random_tree(operators, n_features, 3)

        new_tree = ExprNode(op=tree.op, value=tree.value, var_idx=tree.var_idx)
        if tree.left:
            if self._rng.random() < 0.5:
                new_tree.left = self._mutate(tree.left, operators, n_features)
            else:
                new_tree.left = tree.left
        if tree.right:
            if self._rng.random() < 0.5:
                new_tree.right = self._mutate(tree.right, operators, n_features)
            else:
                new_tree.right = tree.right
        return new_tree

    def _crossover(self, parent1: ExprNode, parent2: ExprNode) -> ExprNode:
        if self._rng.random() < 0.5 and parent1.left:
            return ExprNode(
                op=parent1.op, value=parent1.value, var_idx=parent1.var_idx,
                left=parent2 if self._rng.random() < 0.3 else parent1.left,
                right=parent1.right,
            )
        if parent1.right:
            return ExprNode(
                op=parent1.op, value=parent1.value, var_idx=parent1.var_idx,
                left=parent1.left,
                right=parent2 if self._rng.random() < 0.3 else parent1.right,
            )
        return parent2

    def search(
        self, inputs: np.ndarray, outputs: np.ndarray,
        operators: list[str], max_complexity: int,
    ) -> tuple[str, float]:
        inputs = np.atleast_2d(inputs).astype(np.float64)
        outputs = np.asarray(outputs, dtype=np.float64).ravel()
        n_features = inputs.shape[1]

        valid_ops = [op for op in operators if op in _SAFE_OPS]
        if not valid_ops:
            valid_ops = ["+", "*"]

        population = [
            self._random_tree(valid_ops, n_features, self.max_tree_depth)
            for _ in range(self.population_size)
        ]

        best_tree = population[0]
        best_fitness = self._fitness(best_tree, inputs, outputs, max_complexity)

        for gen in range(self.generations):
            fitnesses = [
                self._fitness(t, inputs, outputs, max_complexity)
                for t in population
            ]

            for i, f in enumerate(fitnesses):
                if f < best_fitness:
                    best_fitness = f
                    best_tree = population[i]

            new_pop: list[ExprNode] = [best_tree]
            while len(new_pop) < self.population_size:
                if self._rng.random() < self.crossover_rate:
                    p1 = self._tournament_select(population, fitnesses)
                    p2 = self._tournament_select(population, fitnesses)
                    child = self._crossover(p1, p2)
                else:
                    parent = self._tournament_select(population, fitnesses)
                    child = self._mutate(parent, valid_ops, n_features)
                new_pop.append(child)

            population = new_pop

        try:
            final_pred = best_tree.evaluate(inputs)
            mse = float(np.mean((final_pred - outputs) ** 2))
        except Exception:
            mse = float("inf")

        return best_tree.to_string(), mse


class DiscoveryLoop:
    """Universal 3-step discovery loop."""

    def __init__(
        self,
        library: "AxiomLibrary",
        k: int,
        l: int,  # noqa: E741
        sr_operators: list[str] | None = None,
        logic_operators: list[str] | None = None,
        max_complexity: int = 10,
        sr_population: int = 100,
        sr_generations: int = 50,
        observers: list[DiscoveryObserver] | None = None,
        seed: int = 42,
    ) -> None:
        self.library = library
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)

        self.sr_operators = sr_operators or [
            "+", "-", "*", "/", "sin", "cos", "exp", "sqrt", "square", "log",
        ]
        self.logic_operators = logic_operators or ["AND", "OR", "NOT", "IMPLIES"]
        self.max_complexity = max_complexity
        self._observers = observers or []

        self._sr = GeneticProgrammingSR(
            population_size=sr_population,
            generations=sr_generations,
            seed=seed,
        )

    def check_residual(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        diff = actual.astype(np.float64) - predicted.astype(np.float64)
        return float(np.linalg.norm(diff))

    def cross_domain_analogy(
        self, anomaly: np.ndarray, domain_a: np.ndarray, domain_b: np.ndarray,
    ) -> np.ndarray:
        unbound = self.bc.unbind(anomaly, domain_a)
        return self.bc.bind(unbound, domain_b)

    def discover(
        self, inputs: np.ndarray, outputs: np.ndarray, context_vec: np.ndarray,
    ) -> "Axiom | None":
        nearest = self.library.select(context_vec, top_n=3)
        extra_ops: set[str] = set()
        for ax in nearest:
            extra_ops.update(ax.operators)

        all_ops = list(set(self.sr_operators) | set(self.logic_operators) | extra_ops)
        valid_ops = [op for op in all_ops if op in _SAFE_OPS]
        if not valid_ops:
            valid_ops = ["+", "*"]

        formula_str, mse = self._sr.search(
            inputs=inputs, outputs=outputs,
            operators=valid_ops, max_complexity=self.max_complexity,
        )

        if mse > 1e6 or not np.isfinite(mse):
            return None

        new_axiom = self.library.make_axiom(
            name=f"discovered_{formula_str[:40]}",
            domain="discovered",
            formula_str=formula_str,
            operators=valid_ops,
        )
        self.library.register(new_axiom)

        for obs in self._observers:
            obs.on_axiom_discovered(new_axiom)

        return new_axiom
