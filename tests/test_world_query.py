"""Inter-world query: a world missing the answer asks the next world.

Pins the delegation protocol validated in CubbyLLM's exp_m3_temporal_causal
screen: worlds ranked by axiom-centroid cosine; the consulted world answers
only when its best individual MEMBER clears tau_answer; otherwise the router
forwards the query to the next world; an exhausted query answers with the
globally best member seen. Worlds never reference each other — the router
mediates every hop.
"""
from __future__ import annotations

import numpy as np

from cubbyllm.bridges.world_model import Challenge

from mowm import AxiomLibrary, MoWMRouter, World
from mowm.axiom_library import Axiom
from mowm.bridges import CubbyBridge
from mowm.domains import seed_all

K, L = 4, 32


def _one_hot(slots: list[int]) -> np.ndarray:
    v = np.zeros((K, L), dtype=np.float32)
    for b, s in enumerate(slots):
        v[b, s] = 1.0
    return v


def _axiom(name: str, vec: np.ndarray) -> Axiom:
    return Axiom(name=name, domain="test", formula_str=name, vector=vec)


def _world(world_id: int, axioms: list[Axiom]) -> World:
    return World(world_id=world_id, k=K, l=L, n_hylas=1, z_depth=0,
                 z_max=2, tau=0.5, axioms=axioms, seed=world_id)


def _near_far_pool():
    """World A: three near-misses of x (cos 0.75 each; centroid ~0.92 to x)
    but NO exact copy. World B: the exact x diluted by four random axioms
    (centroid ~0.5 to x). Centroid routing ranks A first; only B holds the
    answer — the delegation case."""
    x = _one_hot([3, 7, 11, 19])
    perts = []
    for b in range(3):                       # move one block's hot slot each
        p = x.copy()
        p[b, [3, 7, 11][b]] = 0.0
        p[b, ([3, 7, 11][b] + 1) % L] = 1.0
        perts.append(p)
    rng = np.random.default_rng(0)
    randoms = [_one_hot(list(rng.integers(0, L, size=K))) for _ in range(4)]
    world_a = _world(100, [_axiom(f"near{i}", p) for i, p in enumerate(perts)])
    world_b = _world(200, [_axiom("exact", x)]
                     + [_axiom(f"rand{i}", r) for i, r in enumerate(randoms)])
    return x, world_a, world_b


def _router(*worlds: World) -> MoWMRouter:
    r = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    for w in worlds:
        r.add_world(w)
    return r


def test_best_member_finds_exact_axiom():
    x, world_a, world_b = _near_far_pool()
    axiom, score = world_b.best_member(x)
    assert axiom is not None and axiom.name == "exact"
    assert score > 0.999                        # exact copy scores ~1.0
    _, near_score = world_a.best_member(x)
    assert 0.7 < near_score < 0.8               # best near-miss is cos 0.75


def test_best_member_empty_world():
    w = _world(300, [])
    axiom, score = w.best_member(np.ones((K, L), dtype=np.float32))
    assert axiom is None and score == -1.0      # never clears any threshold


def test_query_answers_from_first_world_without_delegation():
    x, world_a, world_b = _near_far_pool()
    router = _router(world_a, world_b)
    ans = router.query_worlds(x, tau_answer=0.6)
    # A wins the centroid ranking AND its best member (0.75) clears 0.6:
    # first world answers, nothing is forwarded.
    assert ans is not None and ans.found is True
    assert ans.world_id == 100
    assert ans.worlds_consulted == 1 and ans.delegated is False


def test_query_delegates_to_the_world_that_holds_the_answer():
    x, world_a, world_b = _near_far_pool()
    router = _router(world_a, world_b)
    ans = router.query_worlds(x, tau_answer=0.9)
    # A still ranks first but its best member (0.75) misses tau=0.9; the
    # query passes to B, whose exact copy answers.
    assert ans is not None and ans.found is True
    assert ans.world_id == 200 and ans.axiom.name == "exact"
    assert ans.worlds_consulted == 2 and ans.delegated is True


def test_query_exhausted_returns_global_best_seen():
    x, world_a, world_b = _near_far_pool()
    router = _router(world_a, world_b)
    ans = router.query_worlds(x, tau_answer=1.01)   # unreachable
    assert ans is not None and ans.found is False   # nobody cleared tau
    assert ans.world_id == 200                      # global best member (1.0)
    assert ans.worlds_consulted == 2                # every world consulted


def test_query_no_active_worlds_returns_none():
    router = _router()
    assert router.query_worlds(np.ones((K, L)), tau_answer=0.5) is None


def _bridge_with_pool(tau_answer):
    x, world_a, world_b = _near_far_pool()
    lib = AxiomLibrary(k=K, l=L)
    import mowm.domains.physics  # noqa: F401
    seed_all(lib, domains=["physics"])
    router = _router(world_a, world_b)
    return x, CubbyBridge(lib, router, tau_answer=tau_answer)


def test_bridge_redirects_to_the_answer_holding_world():
    x, bridge = _bridge_with_pool(tau_answer=0.9)
    before = bridge.num_worlds
    r = bridge.attempt(Challenge(context=x, tag="delegated"))
    # Centroid+margin routes to A (0.92 vs ~0.5 — wide margin), but A is
    # missing the answer at tau=0.9; tier (b2) redirects to B, which holds
    # the exact axiom. Route-vs-spawn is untouched: no spawn happened.
    assert r.spawned is False
    assert r.world_id == "200"
    assert bridge.num_worlds == before


def test_bridge_without_delegation_stays_with_centroid_winner():
    x, bridge = _bridge_with_pool(tau_answer=None)
    r = bridge.attempt(Challenge(context=x, tag="no-delegation"))
    assert r.spawned is False
    assert r.world_id == "100"                  # the control: A keeps it
