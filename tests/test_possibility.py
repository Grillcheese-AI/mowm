"""H-F2 M4: MoWM as the world model behind CubbyLLM's possibility oracle (k=4, l=32; model-free encoder)."""
from __future__ import annotations

import numpy as np
import pytest

from cubbyllm.bridges.possibility import LATENT, PossibilityOracle, PossibleRelations, Verdict

from mowm import AxiomLibrary, MoWMRouter
from mowm.bridges import MoWMPossibility, fact_language
from mowm.encoding.semantic_words import FastWordEncoder

K, L = 4, 32
GEO = ["Paris is the capital of France", "Berlin is the capital of Germany", "Rome is the capital of Italy",
       "Madrid is the capital of Spain", "Lisbon is the capital of Portugal"]
MUSIC = ["Thriller is the album of Michael Jackson", "Abbey Road is the album of The Beatles",
         "Nevermind is the album of Nirvana"]


def _encoder() -> FastWordEncoder:
    """An empty table: every word falls back to its seeded signal code (surface identity only), with the
    function words down-weighted the way a real table's IDF does."""
    return FastWordEncoder(words=[], vectors=np.zeros((0, K * L), dtype=np.float32),
                           idf={w: 0.1 for w in ("is", "the", "of", "a", "an")}, default_idf=1.0, k=K, l=L)


def _oracle(library=None, **kw) -> MoWMPossibility:
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    o = MoWMPossibility(router, encode=_encoder().encode, library=library, **kw)
    o.add_world("geo", GEO)
    o.add_world("music", MUSIC)
    return o


def test_is_a_possibility_oracle_and_a_knows_world():
    o = _oracle()
    assert isinstance(o, PossibilityOracle)
    assert o.worlds() == ["geo", "music"] and o.members("geo") == GEO
    assert o.name == "mowm" and o.domain and o.covers("Whose album is Thriller?") > o.covers("Quuxium of Fnordovia")


def test_route_reports_score_and_margin_under_both_rules():
    for routing in ("member", "centroid"):
        o = _oracle(routing=routing)
        name, score, margin = o.route("Which city is the capital of Germany?")
        assert name == "geo" and score >= 0.35 and margin is not None and margin >= 0.02, routing
        assert o.route("Whose album is Nevermind?")[0] == "music", routing
    with pytest.raises(ValueError):
        _oracle(routing="nearest")


def test_member_is_possible_with_a_prediction():
    o = _oracle()
    p = o.possible("Paris is the capital of France")
    assert p.verdict is Verdict.POSSIBLE and p.world_id == "geo" and p.why == "member" and p.score == 1.0
    assert p.support[0] == ("Paris is the capital of France", pytest.approx(1.0))
    assert p.predicted and all(m in GEO for m, _s in p.predicted)          # symbols, cleaned up against members
    assert all(isinstance(s, float) for _m, s in p.predicted)


def test_contradiction_is_impossible_by_the_hosts_rule():
    o = _oracle()
    p = o.possible("Lyon is the capital of France")
    assert p.impossible and p.world_id == "geo" and p.why == "contradiction"
    assert p.support == (("Paris is the capital of France", pytest.approx(p.score)),)
    assert p.latent is None


def test_contradiction_in_another_world_is_still_impossible():
    o = _oracle()
    f = "Michael Jackson Thriller Beatles is the capital of France"   # obj | capital | France, worded like music
    assert o.route(f)[0] == "music"
    p = o.possible(f)
    assert p.impossible and p.world_id == "geo" and p.why == "contradiction_elsewhere"
    assert p.support[0][0] == "Paris is the capital of France"
    assert not _oracle(cross_world=False).possible(f).impossible
    assert o.members_about(o._by_name("geo"), "Lyon is the capital of France") == ["Paris is the capital of France"]


def test_no_contradiction_rule_never_refuses():
    o = _oracle(contradicts=None)
    assert not o.possible("Lyon is the capital of France").impossible


def test_covered_but_unsupported_waits_in_the_routed_world():
    o = _oracle(tau_answer=0.7)
    p = o.possible("Paris is the largest city of France")      # two words in common with a member: covered, short of tau_answer
    assert p.unknown and p.world_id == "geo" and p.latent == "geo" and p.why == "covered_unsupported"
    assert p.support and p.predicted
    assert _oracle(n_hylas=0, tau_answer=0.7).possible("Paris is the largest city of France").predicted == ()   # no transition model


def test_uncovered_goes_to_the_plain_latent_world_and_learn_fills_it():
    o = _oracle()
    f = "Quuxium is the element of Fnordovia"
    p = o.possible(f)
    assert p.unknown and p.latent == LATENT and p.why == "below_tau_match" and o.spawned == []
    assert o.learn(f, LATENT) is True and LATENT in o.worlds() and o.members(LATENT) == [f]
    assert o.learn(f, LATENT) is False                                  # already a member
    assert o.learn(f, "nowhere") is False
    again = o.possible(f)
    assert again.unknown and again.world_id == LATENT and again.why == "latent_world"   # routes there, stays unverified
    assert again.support[0] == (f, pytest.approx(1.0))


def test_uncovered_spawns_a_specialist_from_the_library():
    lib = AxiomLibrary(k=K, l=L)
    import mowm.domains.physics  # noqa: F401
    from mowm.domains import seed_all
    seed_all(lib, domains=["physics"])
    o = _oracle(library=lib)
    before = o.router.num_active_worlds
    f = "Quuxium is the element of Fnordovia"
    p = o.possible(f)
    assert p.unknown and p.latent.startswith(f"{LATENT}:") and o.spawned == [p.latent]
    assert o.router.num_active_worlds == before + 1
    assert o.learn(f, p.latent) is True
    again = o.possible(f)
    assert again.world_id == p.latent and again.why == "latent_world" and o.router.num_active_worlds == before + 1


def test_learn_makes_a_fact_a_member_of_its_world():
    o = _oracle()
    f = "Vienna is the capital of Austria"
    assert o.possible(f).why != "member"
    assert o.learn(f, "geo") is True and f in o.members("geo")
    assert o.possible(f).why == "member"


def test_relation_kind_reuse_paraphrase_unknown():
    o = _oracle()
    assert o.possible("capital", kind="relation").why == "member"
    assert o.possible("album", kind="relation").world_id == "music"
    o.learn("Cubism is the movement of Picasso", "music")             # one member states 'movement'
    p = o.possible("movement", kind="relation")
    assert (p.possible and p.why == "paraphrase") or p.unknown
    known = PossibleRelations(o)
    assert "capital" in known and "hometown" not in known


EU = GEO + ["Athens is the capital of Greece", "Dublin is the capital of Ireland", "Oslo is the capital of Norway"]
ASIA = ["Tokyo is the capital of Japan", "Fuji is the mountain of Honshu", "Osaka is the city of Kansai",
        "Sapporo is the city of Hokkaido"]


def test_centroid_routing_delegates_to_the_world_that_holds_the_answer():
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    o = MoWMPossibility(router, encode=_encoder().encode, routing="centroid")
    o.add_world("eu", EU)
    o.add_world("asia", ASIA)
    assert o.route("What is the capital of Japan?")[0] == "eu"       # the centroid winner ('capital' x 8) ...
    p = o.possible("What is the capital of Japan?", kind="question")
    assert p.possible and p.world_id == "asia" and p.why == "hits"    # ... does not hold it; the holder answers
    assert p.support[0][0] == "Tokyo is the capital of Japan" and o.delegated == 1
    assert o.answer("What is the capital of Japan?")[0] == "Tokyo is the capital of Japan"


def test_member_routing_goes_to_the_holder_directly():
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    o = MoWMPossibility(router, encode=_encoder().encode)             # routing='member' (the default)
    o.add_world("eu", EU)
    o.add_world("asia", ASIA)
    assert o.route("What is the capital of Japan?")[0] == "asia"
    p = o.possible("What is the capital of Japan?", kind="question")
    assert p.possible and p.world_id == "asia" and o.delegated == 0
    assert p.support[0][0] == "Tokyo is the capital of Japan"


def test_ambiguous_margin_is_unknown_not_a_spawn():
    router = MoWMRouter(k=K, l=L, max_worlds=8, top_k=2, tau_spawn=0.5, seed=42)
    o = MoWMPossibility(router, encode=_encoder().encode)
    o.add_world("a", GEO)
    o.add_world("b", GEO)
    p = o.possible("Paris is the capital of France")
    assert p.unknown and p.why == "ambiguous_margin" and p.latent == "a" and o.spawned == []


def test_seeded_domain_world_answers_in_fact_language():
    from mowm.axiom_library import Axiom
    o = _oracle()
    ax = Axiom(name="Newton's Second Law", domain="physics", formula_str="F = m * a", vector=o._vec("force mass acceleration"))
    assert fact_language(ax) == "F = m * a is the formula of Newton's Second Law"
    assert fact_language(Axiom(name="x is the y of z", domain="d", formula_str="x is the y of z", vector=ax.vector)) == "x is the y of z"


def test_symbolic_boundary_only_symbols_and_scores_cross():
    o = _oracle()
    for p in (o.possible("Paris is the capital of France"), o.possible("Quuxium is the element of Fnordovia"),
              o.possible("What is the capital of Italy?", kind="question")):
        d = p.as_dict()
        assert isinstance(d["world"], str) and isinstance(d["score"], float)
        assert all(isinstance(m, str) and isinstance(s, float) for m, s in d["support"] + d["predicted"])
    assert o.asked and all("verdict" in a for a in o.asked)


def test_text_needs_an_encoder():
    o = MoWMPossibility(MoWMRouter(k=K, l=L, max_worlds=2))
    with pytest.raises(RuntimeError):
        o.possible("anything")
