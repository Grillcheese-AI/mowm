# tests/test_world.py
"""Tests for World class — recursive expert world with HYLA hypernetworks."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.axiom_library import Axiom, AxiomLibrary
from mowm.world import World, BindComposeMerge, ArgmaxCleanup

K, L = 4, 32
D_VSA = K * L


def _make_lib() -> AxiomLibrary:
    return AxiomLibrary(k=K, l=L)


def _make_axioms(lib: AxiomLibrary, n: int = 3) -> list[Axiom]:
    axioms = []
    for i in range(n):
        ax = lib.make_axiom(
            name=f"test_axiom_{i}",
            domain="test",
            formula_str=f"x + {i}",
            operators=["+"],
        )
        axioms.append(ax)
    return axioms


class TestWorld:
    def setup_method(self):
        self.lib = _make_lib()
        self.axioms = _make_axioms(self.lib, 3)

    def test_world_creation(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        assert w.world_id == 0
        assert w.z_depth == 0
        assert w.obs_count == 0

    def test_predict_returns_shape_and_confidence(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=2,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        pred, conf = w.predict(state, action)
        assert pred.shape == (K, L)
        assert pred.dtype == np.float32
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_predict_increments_obs_count(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        w.predict(state, action)
        assert w.obs_count == 1
        w.predict(state, action)
        assert w.obs_count == 2

    def test_predict_updates_last_selected_step(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        w.predict(state, action)
        assert w.last_selected_step >= 0

    def test_spawn_subworld(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        child = w.spawn_subworld(self.axioms[:1])
        assert child.z_depth == 1
        assert child.world_id != w.world_id

    def test_spawn_respects_z_max(self):
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=2, z_max=2, tau=0.5, axioms=self.axioms,
        )
        with pytest.raises(ValueError, match="z_max"):
            w.spawn_subworld(self.axioms[:1])

    def test_consolidate_absorbs_other(self):
        w1 = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        w2 = World(
            world_id=1, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms[:1],
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        w1.predict(state, action)
        w2.predict(state, action)
        w1.consolidate(w2)
        pred, conf = w1.predict(state, action)
        assert pred.shape == (K, L)

    def test_deterministic_predictions(self):
        w1 = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms, seed=42,
        )
        w2 = World(
            world_id=1, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms, seed=42,
        )
        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        p1, c1 = w1.predict(state, action)
        p2, c2 = w2.predict(state, action)
        np.testing.assert_array_equal(p1, p2)


class TestBindComposeMerge:
    def test_merge_shape(self):
        bc = AxiomLibrary(k=K, l=L).bc
        d1 = bc.random_discrete(seed=1)
        d2 = bc.random_discrete(seed=2)
        merger = BindComposeMerge(bc)
        result = merger.merge([d1, d2], [0.5, 0.5])
        assert result.shape == (K, L)

    def test_single_delta(self):
        bc = AxiomLibrary(k=K, l=L).bc
        d1 = bc.random_discrete(seed=1)
        merger = BindComposeMerge(bc)
        result = merger.merge([d1], [1.0])
        np.testing.assert_array_equal(result, d1)


class TestArgmaxCleanup:
    def test_cleanup_produces_one_hot(self):
        cleanup = ArgmaxCleanup()
        noisy = np.random.default_rng(42).random((K, L)).astype(np.float32)
        clean = cleanup.cleanup(noisy)
        assert clean.shape == (K, L)
        for block_idx in range(K):
            assert np.sum(clean[block_idx]) == pytest.approx(1.0)
            assert np.max(clean[block_idx]) == pytest.approx(1.0)
