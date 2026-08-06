"""H-F2 M1 kill-criterion: automatic specialization across the Cubby<->MoWM bridge."""
from __future__ import annotations

import numpy as np

from cubbyllm.bridges.world_model import Challenge

from mowm import AxiomLibrary, MoWMRouter, World
from mowm.bridges import CubbyBridge
from mowm.domains import seed_all

K, L = 4, 32


def _bridge():
    lib = AxiomLibrary(k=K, l=L)
    import mowm.domains.physics  # noqa: F401
    import mowm.domains.logic    # noqa: F401
    seed_all(lib, domains=["physics", "logic"])
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    # seed two known worlds from disjoint axiom slices
    phys = lib.select(lib.bc.random_discrete(seed=1), top_n=3)
    logi = lib.select(lib.bc.random_discrete(seed=2), top_n=3)
    router.add_world(World(world_id=100, k=K, l=L, n_hylas=2, z_depth=0,
                           z_max=2, tau=0.5, axioms=phys, seed=100))
    router.add_world(World(world_id=200, k=K, l=L, n_hylas=2, z_depth=0,
                           z_max=2, tau=0.5, axioms=logi, seed=200))
    return CubbyBridge(lib, router)


def test_novel_challenge_spawns_a_specialist_then_reuses_it():
    bridge = _bridge()
    before = bridge.num_worlds
    ctx = bridge.library.encoder._hash_to_vec("hf2::novel_challenge")

    r1 = bridge.attempt(Challenge(context=ctx, tag="novel"))
    assert r1.spawned is True                              # grew a specialist
    assert bridge.num_worlds == before + 1                 # exactly one world added
    assert bridge.registered == [int(r1.world_id)]

    r2 = bridge.attempt(Challenge(context=ctx, tag="novel"))
    assert r2.spawned is False                             # reuse, no re-spawn
    assert r2.world_id == r1.world_id
    assert bridge.num_worlds == before + 1                 # single-spawn control


def test_known_challenge_routes_without_spawning():
    bridge = _bridge()
    before = bridge.num_worlds
    known_ctx = np.asarray(bridge._world_by_id(100).axiom_vec, dtype=np.float32)

    r = bridge.attempt(Challenge(context=known_ctx, tag="known-physics"))
    assert r.spawned is False                              # reuse control
    assert r.world_id == "100"
    assert bridge.num_worlds == before                     # no spurious spawn
