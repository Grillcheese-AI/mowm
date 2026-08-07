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
    # world_100.axioms[0] is a real individual axiom of world 100. Under
    # centroid routing (exp_m3_open_set's winning rule) a member axiom scores
    # high against its own world's centroid (~0.58 for 3 distinct one-hot
    # axioms) while world 200's centroid — a disjoint axiom selection (seed=1
    # vs seed=2 slices, verified no shared vector) — scores near the random
    # floor, so world 100 wins with a wide margin.
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
    # perturbed => 3/4 blocks still match => cosine 0.75 to the original axiom
    # and ~0.43 to world 100's centroid: above tau_match (0.35), below 1.0 —
    # genuinely fuzzy. World 200's disjoint centroid stays near the random
    # floor, so the margin gate passes and world 100 wins.
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


def test_axiom_centroid_is_similarity_preserving():
    """The centroid must resemble its member axioms — the property the
    bind-chain axiom_vec provably lacks (the M1 finding)."""
    bridge = _bridge()
    world_100 = bridge._world_by_id(100)
    c = np.asarray(world_100.axiom_centroid, dtype=np.float64).ravel()
    assert abs(np.linalg.norm(c) - 1.0) < 1e-6              # unit norm
    for ax in world_100.axioms:
        v = np.asarray(ax.vector, dtype=np.float64).ravel()
        cos = float(c @ v / (np.linalg.norm(v) + 1e-12))
        assert cos > 0.3, f"member axiom cosine {cos:.3f} — centroid not similar"


def test_ambiguous_challenge_spawns_when_worlds_tie():
    """The margin gate: when two worlds explain a challenge EQUALLY well,
    routing is ambiguous and the bridge must spawn (the recoverable outcome)
    rather than pick arbitrarily — the validated conservative profile."""
    lib = AxiomLibrary(k=K, l=L)
    import mowm.domains.physics  # noqa: F401
    seed_all(lib, domains=["physics"])
    axioms = lib.select(lib.bc.random_discrete(seed=1), top_n=3)
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    # two worlds with IDENTICAL axiom sets -> identical centroids -> margin ~ 0
    for wid in (100, 300):
        router.add_world(World(world_id=wid, k=K, l=L, n_hylas=2, z_depth=0,
                               z_max=2, tau=0.5, axioms=axioms, seed=wid))
    bridge = CubbyBridge(lib, router)
    before = bridge.num_worlds
    ctx = np.asarray(axioms[0].vector, dtype=np.float32)    # matches BOTH worlds

    r = bridge.attempt(Challenge(context=ctx, tag="ambiguous"))
    assert r.spawned is True                                # tie -> spawn, not guess
    assert bridge.num_worlds == before + 1


def test_single_world_routes_on_absolute_floor():
    """With one world the margin is undefined; the absolute floor governs:
    a matching challenge routes, a novel one still spawns."""
    lib = AxiomLibrary(k=K, l=L)
    import mowm.domains.physics  # noqa: F401
    seed_all(lib, domains=["physics"])
    axioms = lib.select(lib.bc.random_discrete(seed=1), top_n=3)
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    router.add_world(World(world_id=100, k=K, l=L, n_hylas=2, z_depth=0,
                           z_max=2, tau=0.5, axioms=axioms, seed=100))
    bridge = CubbyBridge(lib, router)

    r = bridge.attempt(Challenge(context=np.asarray(axioms[0].vector,
                                                    dtype=np.float32), tag="known"))
    assert r.spawned is False and r.world_id == "100"       # floor passes, routes

    novel = bridge.library.encoder._hash_to_vec("m3::single-world-novel")
    r2 = bridge.attempt(Challenge(context=novel, tag="novel"))
    assert r2.spawned is True                               # floor blocks, spawns
