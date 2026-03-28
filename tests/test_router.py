# tests/test_router.py
"""Tests for MoWMRouter — dynamic expert routing with spawn/prune."""
from __future__ import annotations

import pytest

from mowm.axiom_library import AxiomLibrary
from mowm.router import MoWMRouter
from mowm.world import World

K, L = 4, 32


def _make_lib() -> AxiomLibrary:
    return AxiomLibrary(k=K, l=L)


def _make_axioms(lib, n=3):
    return [
        lib.make_axiom(f"ax_{i}", "test", f"x+{i}", ["+"])
        for i in range(n)
    ]


class TestMoWMRouter:
    def setup_method(self):
        self.lib = _make_lib()
        self.axioms = _make_axioms(self.lib, 3)

    def test_creation(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        assert router.num_active_worlds == 0

    def test_add_world(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        router.add_world(w)
        assert router.num_active_worlds == 1

    def test_route_returns_worlds_and_weights(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        for i in range(3):
            w = World(
                world_id=i, k=K, l=L, n_hylas=1,
                z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
                seed=i * 10,
            )
            router.add_world(w)

        bc = self.lib.bc
        state = bc.random_discrete(seed=10)
        action = bc.random_discrete(seed=20)
        result = router.route(state, action)

        assert len(result) >= 1
        assert len(result) <= 2
        for world, weight in result:
            assert isinstance(world, World)
            assert isinstance(weight, float)
            assert weight >= 0.0

    def test_maybe_spawn_creates_world(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.01,
            tau_prune_steps=100,
        )
        bc = self.lib.bc
        residual = bc.random_discrete(seed=99)

        import mowm.domains.physics  # noqa: F401
        from mowm.domains import seed_all
        seed_all(self.lib, domains=["physics"])

        new_world = router.maybe_spawn(residual, self.lib)
        assert new_world is not None
        assert router.num_active_worlds == 1

    def test_prune_removes_inactive(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=5,
        )
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        router.add_world(w)
        pruned = router.prune(current_step=10)
        assert 0 in pruned
        assert router.num_active_worlds == 0

    def test_prune_keeps_active(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        w = World(
            world_id=0, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        router.add_world(w)
        w.mark_selected(5)
        pruned = router.prune(current_step=10)
        assert len(pruned) == 0
        assert router.num_active_worlds == 1

    def test_max_worlds_enforced(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=2, top_k=1,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        for i in range(2):
            w = World(
                world_id=i, k=K, l=L, n_hylas=1,
                z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
            )
            router.add_world(w)
        assert router.num_active_worlds == 2

        w3 = World(
            world_id=99, k=K, l=L, n_hylas=1,
            z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
        )
        with pytest.raises(ValueError, match="max_worlds"):
            router.add_world(w3)

    def test_entropy_regularization(self):
        router = MoWMRouter(
            k=K, l=L, max_worlds=8, top_k=2,
            tau_spawn=0.5, tau_prune_steps=100,
        )
        for i in range(3):
            w = World(
                world_id=i, k=K, l=L, n_hylas=1,
                z_depth=0, z_max=2, tau=0.5, axioms=self.axioms,
                seed=i * 10,
            )
            router.add_world(w)

        ent = router.entropy_reg()
        assert isinstance(ent, float)
        assert ent >= 0.0
