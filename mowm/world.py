# mowm/world.py
"""World — recursive expert world model with HYLA hypernetworks.

Each world at depth z contains one or more HYLAs that generate transition
deltas. Deltas are merged via a MergeStrategy and applied to the state
via block-code binding. Sub-worlds at z+1 handle residuals.

Implements the Composite pattern: worlds are tree nodes with optional children.
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from cubemind.execution.hyla import HYLA
from cubemind.ops.block_codes import BlockCodes

from mowm.base import BaseWorld, CleanupStrategy, MergeStrategy

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom

_world_id_counter = itertools.count(start=1000)


class BindComposeMerge(MergeStrategy):
    """Merge deltas via weighted bundling (algebraic composition)."""

    def __init__(self, bc: BlockCodes) -> None:
        self.bc = bc

    def merge(
        self,
        deltas: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        if len(deltas) == 1:
            return deltas[0]

        weighted = [w * d for w, d in zip(weights, deltas)]
        stacked = np.stack(weighted, axis=0)
        merged = stacked.sum(axis=0).astype(np.float32)

        block_sums = merged.sum(axis=-1, keepdims=True)
        block_sums = np.where(block_sums == 0, 1.0, block_sums)
        return (merged / block_sums).astype(np.float32)


class WeightedAvgMerge(MergeStrategy):
    """Merge deltas via weighted average (simple linear combination)."""

    def merge(
        self,
        deltas: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        if len(deltas) == 1:
            return deltas[0]

        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0
        normalized_weights = [w / total_weight for w in weights]
        result = sum(w * d for w, d in zip(normalized_weights, deltas))
        return result.astype(np.float32)


class ArgmaxCleanup(CleanupStrategy):
    """Per-block argmax cleanup — produces one-hot block codes."""

    def cleanup(self, vec: np.ndarray) -> np.ndarray:
        k, l_block = vec.shape
        clean = np.zeros_like(vec)
        for block_idx in range(k):
            max_idx = np.argmax(vec[block_idx])
            clean[block_idx, max_idx] = 1.0
        return clean


class World(BaseWorld):
    """A single recursive expert world model at depth z.

    Contains one or more HYLAs that generate transition deltas from an
    input (state_vec || action_vec). Deltas are merged via MergeStrategy
    and applied to the state via block-code binding.

    Args:
        world_id: Unique identifier.
        k: Number of blocks.
        l: Block length.
        n_hylas: Number of HYLA modules per world.
        z_depth: Current depth in the recursive tree.
        z_max: Maximum allowed recursion depth.
        tau: Confidence threshold for spawning sub-worlds.
        axioms: List of axioms this world operates with.
        seed: Random seed for reproducibility.
        merge_strategy: Strategy for merging HYLA deltas.
        cleanup_strategy: Strategy for cleaning up predictions.
    """

    def __init__(
        self,
        world_id: int,
        k: int,
        l: int,  # noqa: E741
        n_hylas: int,
        z_depth: int,
        z_max: int,
        tau: float,
        axioms: list[Axiom],
        seed: int = 42,
        merge_strategy: MergeStrategy | None = None,
        cleanup_strategy: CleanupStrategy | None = None,
    ) -> None:
        self.world_id = world_id
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.n_hylas = n_hylas
        self.z_depth = z_depth
        self.z_max = z_max
        self.tau = tau
        self.axioms = list(axioms)
        self._seed = seed

        self.bc = BlockCodes(k=k, l=l)
        self._merge = merge_strategy or BindComposeMerge(self.bc)
        self._cleanup = cleanup_strategy or ArgmaxCleanup()

        d_input = 2 * self.d_vsa
        d_hidden = min(256, self.d_vsa)
        self.hylas: list[HYLA] = []
        for i in range(n_hylas):
            hyla = HYLA(
                d_vsa=d_input,
                d_hidden=d_hidden,
                d_out=self.d_vsa,
                k=k,
                l=l * 2,
                seed=seed + i,
            )
            self.hylas.append(hyla)

        if axioms:
            self._axiom_vec = axioms[0].vector.copy()
            for ax in axioms[1:]:
                self._axiom_vec = self.bc.bind(self._axiom_vec, ax.vector)
        else:
            self._axiom_vec = self.bc.random_discrete(seed=seed + 1000)

        self.subworlds: list[World] = []
        self._obs_count: int = 0
        self._last_selected_step: int = -1

    @property
    def obs_count(self) -> int:
        return self._obs_count

    @property
    def axiom_vec(self) -> np.ndarray:
        """The world's bound-axiom signature (a bind-chain over the world's axioms)."""
        return self._axiom_vec

    @property
    def axiom_centroid(self) -> np.ndarray:
        """Normalized mean of this world's individual axiom vectors.

        The similarity-preserving BUNDLED signature: cosine to the centroid
        reflects resemblance to the world's axioms, unlike ``axiom_vec`` (a
        bind-chain, quasi-orthogonal to its own parts, unusable for routing —
        the H-F2 M1 finding). Centroid routing beat max-over-individual-axioms
        0.680 vs 0.589 macro at ~40x fewer comparisons in CubbyLLM's
        exp_m3_open_set screen. Recomputed per access (worlds hold few
        axioms); an empty-axiom world falls back to its normalized random
        signature so the property is always defined.
        """
        if not self.axioms:
            v = self._axiom_vec.astype(np.float64).ravel()
            return (v / (np.linalg.norm(v) + 1e-12)).reshape(
                self._axiom_vec.shape).astype(np.float32)
        M = np.stack([np.asarray(a.vector, dtype=np.float64).ravel()
                      for a in self.axioms])
        c = M.mean(axis=0)
        return (c / (np.linalg.norm(c) + 1e-12)).reshape(
            self.axioms[0].vector.shape).astype(np.float32)

    @property
    def last_selected_step(self) -> int:
        return self._last_selected_step

    def mark_selected(self, step: int) -> None:
        self._last_selected_step = step

    def predict(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        self._obs_count += 1
        self._last_selected_step = self._obs_count

        input_flat = np.concatenate(
            [state_vec.ravel(), action_vec.ravel()]
        ).astype(np.float32)

        axiom_flat = np.concatenate(
            [self._axiom_vec.ravel(), self._axiom_vec.ravel()]
        ).astype(np.float32)
        conditioned_input = input_flat + axiom_flat

        deltas: list[np.ndarray] = []
        for hyla in self.hylas:
            delta_flat = hyla.forward(conditioned_input, conditioned_input)
            delta = delta_flat[:self.d_vsa].reshape(self.k, self.l)
            deltas.append(delta)

        weights = [1.0 / len(deltas)] * len(deltas)
        merged_delta = self._merge.merge(deltas, weights)
        predicted = self.bc.bind(state_vec, merged_delta)

        block_maxes = np.max(predicted, axis=-1)
        confidence = float(np.mean(block_maxes))
        confidence = min(max(confidence, 0.0), 1.0)

        return predicted.astype(np.float32), confidence

    def spawn_subworld(self, axioms: list[Axiom]) -> World:
        if self.z_depth >= self.z_max:
            raise ValueError(
                f"Cannot spawn sub-world: already at z_max={self.z_max}"
            )
        child_id = next(_world_id_counter)
        child = World(
            world_id=child_id,
            k=self.k,
            l=self.l,
            n_hylas=self.n_hylas,
            z_depth=self.z_depth + 1,
            z_max=self.z_max,
            tau=self.tau,
            axioms=axioms,
            seed=self._seed + child_id,
            merge_strategy=self._merge,
            cleanup_strategy=self._cleanup,
        )
        self.subworlds.append(child)
        return child

    def consolidate(self, other: World) -> None:
        alpha = 0.1
        total_obs = max(self._obs_count + other._obs_count, 1)
        weight_other = other._obs_count / total_obs

        # Oja-merge every current HYLA weight matrix. HYLA's old monolithic W_H
        # output matrix (53GB at d_vsa=10240) was factored into the low-rank
        # W_A + W_B pair; merge those, not the removed W_H (see
        # cubemind/execution/hyla.py).
        for h_self, h_other in zip(self.hylas, other.hylas):
            for attr in ("W_h", "W_A", "W_B"):
                w_self = np.asarray(getattr(h_self, attr), dtype=np.float32)
                w_other = np.asarray(getattr(h_other, attr), dtype=np.float32)
                merged = (
                    w_self + alpha * weight_other * (w_other - w_self)
                ).astype(np.float32)
                setattr(h_self, attr, merged)

        self._axiom_vec = self.bc.bundle(
            [self._axiom_vec, other._axiom_vec], normalize=True
        )
        self._obs_count += other._obs_count
