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
    world_100 = bridge._world_by_id(100)
    # world_100.axioms[0] ("Rotational KE", physics.energy) is a real individual
    # axiom of world 100, not the axiom_vec bind-chain. Uniqueness of the route
    # is guaranteed, not assumed: cosine between two one-hot-per-block codes is
    # exactly 1.0 iff every block matches, i.e. iff the vectors are identical --
    # so cosine(known_ctx, ax) < 1.0 for any ax that isn't a literal duplicate of
    # this vector. world 100 (seeded from lib.select(seed=1)) and world 200
    # (seeded from lib.select(seed=2)) are disjoint axiom-vector selections
    # (verified by an exact-equality sweep over both 3-axiom slices -- no shared
    # vector), so this axiom cannot tie world 200 and world 100 wins uniquely.
    known_ctx = np.asarray(world_100.axioms[0].vector, dtype=np.float32)

    r = bridge.attempt(Challenge(context=known_ctx, tag="known-physics"))
    assert r.spawned is False                              # reuse control
    assert r.world_id == "100"
    assert bridge.num_worlds == before                     # no spurious spawn


def test_fuzzy_axiom_perturbation_still_routes_by_similarity():
    """Near-miss coverage for path (b): a context that is NOT an exact axiom
    copy but is close to one (cosine 0.75, above tau_match=0.35) must still
    route to that axiom's world -- proving real similarity routing, not just
    an exact-match short-circuit."""
    bridge = _bridge()
    before = bridge.num_worlds
    world_100 = bridge._world_by_id(100)
    axiom_vec = np.asarray(world_100.axioms[0].vector, dtype=np.float32)

    # Move block 0's active one-hot slot to a different index. k=4 blocks, one
    # perturbed => 3/4 blocks still match => cosine = 3/4 = 0.75 to the
    # original axiom: >= tau_match (0.35) and < 1.0, i.e. genuinely fuzzy, not
    # an exact copy. Verified (see routing script) that this perturbed vector's
    # cosine to every world-200 axiom stays ~0 (no coincidental near-match), so
    # its 0.75 to the world-100 axiom is the unique max across both worlds.
    perturbed = axiom_vec.copy()
    block0 = perturbed[0]
    old_idx = int(np.argmax(block0))
    new_idx = (old_idx + 1) % block0.shape[0]
    block0[old_idx] = 0.0
    block0[new_idx] = 1.0

    # fresh tag: path (a)'s exact-reuse memo can't short-circuit this challenge
    r = bridge.attempt(Challenge(context=perturbed, tag="fuzzy-physics"))
    assert r.spawned is False                               # similarity match, not spawn
    assert r.world_id == "100"
    assert bridge.num_worlds == before                      # no spurious spawn
