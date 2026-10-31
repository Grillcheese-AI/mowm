"""MoWMPossibility -- MoWM as the world model behind CubbyLLM's possibility oracle (H-F2 M4, 2026-09-29).

Dependency points mowm -> cubbyllm (never the reverse): CubbyLLM ships `Possibility`, `Verdict` and the
`PossibilityOracle` Protocol (cubbyllm/bridges/possibility.py) and reads them in its capsule stores, its
world router and plan_verify; this adapter answers them with MoWM's router and worlds. `CubbyBridge` (the
challenge round-trip, block-code in, block-code out) is unchanged beside it; this is the SYMBOLIC side of
the same bridge: text in, symbols and scores out, no vector crosses back.

Worlds here are MoWM `World`s whose axioms are FACTS: `add_world(name, facts)` encodes each fact once with
the injected `encode` (a `FastWordEncoder`: model-free, table lookup, ~25 us/text) and makes it a member
axiom (name = the fact, formula_str = the fact, domain = the world). A seeded domain world (physics,
logic, ... from `mowm.domains`) mounts as-is with `mount(world, name)`; its members answer in fact language
as '<formula> is the formula of <name>'. Time worlds are fact worlds named by their era or year.

Rules (each validated in CubbyLLM's screens; every tau is deployment-calibrated):
  route      each world scores the text -- `routing='member'` (default): by its best member, one matmul
             per world (MoWM's `query_worlds` in its exhaustive form; the rule today's `route_world`
             applies) -- or `routing='centroid'`: cosine to its axiom centroid, the open-set DOMAIN rule
             (exp_m3_open_set: 0.680 macro vs 0.589 for max-over-axioms, 92% route precision, 87% of
             never-seen domains sent to spawn). A world covers the text iff the best score clears
             `tau_match` and, with 2+ worlds, the margin over the runner-up clears `tau_margin`. The
             centroid rule does not transfer to fact worlds: a member sits ~1/sqrt(n) from a centroid of n
             quasi-orthogonal facts (exp_f2m4_possibility, 2026-09-29), so it is kept for law worlds.
  support    `World.best_member` at or above `tau_answer`; under centroid routing, short of it,
             `MoWMRouter.query_worlds` passes the query world-to-world and the holder answers
             (exp_m3_temporal_causal: flat search's answer-found rate at ~60x fewer member comparisons)
  predict    `World.predict(ctx, neutral action)` cleaned up against the world's members: the top-k
             (member, cosine) -- what the world expects next of this fact, as symbols
  refuse     the injected `contradicts(fact, members)` (CubbyLLM's `same_subject_clash` by default): the
             world model never learns the fact grammar; the host's rule is applied to the routed world's
             members
  latent     nothing covers the text -> `MoWMRouter.maybe_spawn` grows a specialist from the library's
             nearest axioms (named 'latent:<id>'); with no library, a full pool or no near axioms, the
             plain latent world takes it. `learn(fact, world)` -- called by CubbyLLM's CapsuleStore after
             it keeps a fact -- makes the fact a member of that world, so a latent world fills with what
             waits in it and the next such fact routes there: spawn a latent world, verify later.
"""
from __future__ import annotations

import itertools

import numpy as np

from cubbyllm.bridges.possibility import (LATENT, Possibility, Verdict, relation_members,
                                          same_subject_clash)

from ..axiom_library import Axiom
from ..router import MoWMRouter
from ..world import World

_fact_world_ids = itertools.count(start=9000)      # spawned worlds count from 5000, sub-worlds from 1000


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def _subject_of(fact: str) -> str | None:
    """The normalized subject of a fact in CubbyLLM's template, or None -- the one place the adapter reads
    the fact grammar, and only to index members by subject (imported lazily: the grammar stays cubbyllm's)."""
    from cubbyllm.reasoning.planner import normalize, parse_fact
    t = parse_fact(fact)
    return normalize(t.subj) if t is not None else None


def fact_language(axiom: Axiom) -> str:
    """A member as a fact string: a fact-world member is its own text; a law is '<formula> is the formula
    of <name>' (the template `parse_fact` reads: obj | rel | subj)."""
    if axiom.formula_str == axiom.name or not axiom.formula_str:
        return axiom.name
    return f"{axiom.formula_str} is the formula of {axiom.name}"


class MoWMPossibility:
    """CubbyLLM's `PossibilityOracle` over a `MoWMRouter` (and a `Knows` world: name, domain, covers, answer).

    `encode(text) -> (k, l)` is required for text (a FastWordEncoder; the same one that encoded the members,
    or nothing routes). `library` enables spawning a specialist for uncovered text; None keeps one plain
    latent world. `tau_answer=None` disables inter-world delegation. `top_k` bounds support and predicted."""

    name = "mowm"
    domain = "what the mounted worlds hold: facts, laws, eras"

    def __init__(self, router: MoWMRouter, encode=None, library=None, tau_match: float = 0.35,
                 tau_margin: float = 0.02, tau_answer: float | None = 0.6, contradicts=same_subject_clash,
                 top_k: int = 5, latent: str = LATENT, predict: bool = True, routing: str = "member",
                 n_hylas: int = 2, cross_world: bool = True) -> None:
        if routing not in ("member", "centroid"):
            raise ValueError(f"routing must be 'member' or 'centroid', got {routing!r}")
        self.router = router
        self.encode = encode
        self.library = library
        self.tau_match, self.tau_margin, self.tau_answer = float(tau_match), float(tau_margin), tau_answer
        self.contradicts = contradicts
        self.top_k = int(top_k)
        self.latent = latent
        self.predict_enabled = bool(predict)
        # 'member': a world scores by its best member (one matmul per world: MoWM's `query_worlds` in its
        # exhaustive form, and the rule today's route_world applies) -- for FACT worlds. 'centroid': cosine to
        # the axiom centroid, the rule validated for open-set DOMAIN routing (exp_m3_open_set). A fact world's
        # centroid is its relation words: n quasi-orthogonal members put each at ~1/sqrt(n) from it, so an
        # absolute floor abstains on most of a 180-fact world (exp_f2m4_possibility smoke, 2026-09-29: 65%
        # abstained, 0.74 accuracy when routed vs 0.90 by best member). Member routing is the default; the
        # centroid rule stays for worlds of few, coherent axioms (laws).
        self.routing = routing
        self.n_hylas = int(n_hylas)                        # 0 builds fact worlds without a transition model (no predict)
        # a contradiction is looked for among the routed world's members ABOUT THE SUBJECT (an index, not the
        # whole world), then -- `cross_world` -- in every other non-latent world that holds the subject: a
        # planted fact whose new object pulls its best member elsewhere still meets the world that holds
        # its subject (exp_f2m4: 29 of 33 escapes routed away from home). Off for time worlds, where the
        # same subject legitimately has another object in another year.
        self.cross_world = bool(cross_world)
        self._about: dict[int, dict[str, list[str]]] = {}  # world_id -> normalized subject -> member facts
        self.names: dict[int, str] = {}                    # world_id -> name (the symbol that crosses)
        self.asked: list[dict] = []                        # every possible(), for the trace
        self.spawned: list[str] = []                       # latent worlds grown here, in order
        self.delegated = 0                                 # answers that came from a world other than the centroid winner

    # -- worlds -----------------------------------------------------------------------------------------
    def _vec(self, text: str) -> np.ndarray:
        if self.encode is None:
            raise RuntimeError("MoWMPossibility needs encode(text) -> (k, l) to answer text")
        return np.asarray(self.encode(text), dtype=np.float32).reshape(self.router.k, self.router.l)

    def _name(self, world: World) -> str:
        return self.names.get(world.world_id, str(world.world_id))

    def _by_name(self, name: str) -> World | None:
        for w in self.router.active_worlds():
            if self._name(w) == name:
                return w
        return None

    def mount(self, world: World, name: str | None = None) -> World:
        """A ready world (a seeded domain, a spawned specialist) under a name."""
        if self.router.get_world_by_id(world.world_id) is None:
            self.router.add_world(world)
        self.names[world.world_id] = name or str(world.world_id)
        about = self._about.setdefault(world.world_id, {})
        for a in world.axioms:
            self._index(about, fact_language(a))
        return world

    @staticmethod
    def _index(about: dict[str, list[str]], fact: str) -> None:
        subj = _subject_of(fact)
        if subj is not None:
            about.setdefault(subj, []).append(fact)

    def members_about(self, world: World, fact: str) -> list[str]:
        """The world's members with `fact`'s subject: what it could contradict."""
        subj = _subject_of(fact)
        if subj is None:
            return []
        return list(self._about.get(world.world_id, {}).get(subj, []))

    def add_world(self, name: str, facts=(), world_id: int | None = None) -> World:
        """A fact world: one member axiom per distinct fact, encoded once. Empty facts make an empty world
        (it covers nothing until `learn` fills it: the plain latent world starts this way)."""
        if self._by_name(name) is not None:
            raise ValueError(f"world already mounted: {name}")
        wid = next(_fact_world_ids) if world_id is None else int(world_id)
        axioms = [self._axiom(name, f) for f in dict.fromkeys(" ".join(f.split()) for f in facts if f and f.strip())]
        # built empty, then given its members: the bind-chain signature World computes over its axioms is
        # inert for routing (the M1 finding) and only conditions `predict`, and a bind per member would make
        # a thousand-fact world at production width cost seconds; the random signature it takes instead is
        # the same constant `learn` leaves untouched when it appends a member.
        w = World(world_id=wid, k=self.router.k, l=self.router.l, n_hylas=self.n_hylas, z_depth=0, z_max=2,
                  tau=0.5, axioms=[], seed=wid)
        w.axioms = axioms
        return self.mount(w, name)

    def _axiom(self, world_name: str, fact: str) -> Axiom:
        return Axiom(name=fact, domain=world_name, formula_str=fact, vector=self._vec(fact))

    def learn(self, fact: str, world: str) -> bool:
        """CapsuleStore's hook after it keeps a fact: the fact becomes a member of `world` (created empty
        when it is the plain latent name and does not exist yet). False when the world is unknown or the
        fact already a member."""
        key = " ".join(fact.split())
        w = self._by_name(world)
        if w is None:
            if world != self.latent:
                return False
            w = self.add_world(self.latent)
        if any(a.name == key for a in w.axioms):
            return False
        w.axioms.append(self._axiom(world, key))          # the member matrix and centroid follow the count; the
        self._index(self._about.setdefault(w.world_id, {}), key)   # bind-chain signature only conditions predict
        return True

    def worlds(self) -> list[str]:
        return sorted(self._name(w) for w in self.router.active_worlds())

    def members(self, world: str) -> list[str]:
        w = self._by_name(world)
        return [fact_language(a) for a in w.axioms] if w is not None else []

    # -- routing (centroid + margin) ------------------------------------------------------------------
    def _scores(self, ctx: np.ndarray) -> list[tuple[World, float]]:
        if self.routing == "centroid":
            scores = [(w, _cosine(ctx, w.axiom_centroid)) for w in self.router.active_worlds()]
        else:
            scores = [(w, max(0.0, w.best_member(ctx)[1])) for w in self.router.active_worlds()]
        scores.sort(key=lambda ws: (-ws[1], self._name(ws[0])))       # ties break by name: reproducible
        return scores

    def _is_latent(self, world: World) -> bool:
        n = self._name(world)
        return n == self.latent or n in self.spawned

    def route(self, text: str) -> tuple[str, float, float | None]:
        """The best world under the routing rule, its score and the margin; whether the score clears a
        threshold is the caller's (`possible` applies tau_match and tau_margin; serve's `route_tau` gates
        retrieval)."""
        ranked = self._scores(self._vec(text))
        if not ranked:
            return self.latent, 0.0, None
        best, s1 = ranked[0]
        return self._name(best), s1, (s1 - ranked[1][1]) if len(ranked) > 1 else None

    def _spawn(self, ctx: np.ndarray) -> str:
        """A latent world for text nothing covers: a specialist from the library's nearest axioms when a
        library is mounted and the pool has room, else the plain latent world."""
        if self.library is not None:
            w = self.router.maybe_spawn(ctx, self.library)
            if w is not None:
                name = f"{self.latent}:{w.world_id}"
                self.names[w.world_id] = name
                self.spawned.append(name)
                return name
        return self.latent

    def _members_scored(self, world: World, ctx: np.ndarray) -> list[tuple[str, float]]:
        if not world.axioms:
            return []
        M, norms = world._member_matrix()                          # stacked once per axiom count
        c = np.asarray(ctx, dtype=np.float32).ravel()
        cn = float(np.linalg.norm(c)) or 1.0
        sims = (M @ c) / (norms * cn)
        k = min(self.top_k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        order = top[np.argsort(-sims[top], kind="stable")]
        return [(fact_language(world.axioms[i]), float(sims[i])) for i in order]

    def _predicted(self, world: World, ctx: np.ndarray) -> tuple[tuple[str, float], ...]:
        """The world's transition model, cleaned up against its members: symbols out, never the vector."""
        if not self.predict_enabled or not world.axioms or not world.hylas:
            return ()
        pred, _conf = world.predict(ctx, np.zeros_like(ctx))
        return tuple(self._members_scored(world, pred))

    # -- the oracle -------------------------------------------------------------------------------------
    def _record(self, text: str, kind: str, p: Possibility) -> Possibility:
        self.asked.append({"text": text, "kind": kind, **p.as_dict()})
        return p

    def possible(self, text: str, kind: str = "fact", context: str | None = None) -> Possibility:
        key = " ".join(text.split())
        ctx = self._vec(key)
        ranked = self._scores(ctx)
        if not ranked:
            latent = self._spawn(ctx)
            return self._record(key, kind, Possibility(latent, Verdict.UNKNOWN, 0.0, None, latent=latent, why="no_worlds"))
        best, s1 = ranked[0]
        margin = (s1 - ranked[1][1]) if len(ranked) > 1 else None
        if s1 < self.tau_match:
            latent = self._spawn(ctx)
            return self._record(key, kind, Possibility(latent, Verdict.UNKNOWN, s1, margin, latent=latent,
                                                       why="below_tau_match"))
        if margin is not None and margin < self.tau_margin:      # two worlds claim it: covered, not placed
            return self._record(key, kind, Possibility(self._name(best), Verdict.UNKNOWN, s1, margin,
                                                       latent=self._name(best), why="ambiguous_margin"))
        world = best
        member, ms = world.best_member(ctx)
        # (b2) under centroid routing the winner may be missing it: delegate world-to-world. Under member
        # routing the winner holds the best member by construction, so there is nothing to delegate.
        if self.routing == "centroid" and self.tau_answer is not None and ms < self.tau_answer:
            ans = self.router.query_worlds(ctx, self.tau_answer)
            if ans is not None and ans.found and ans.world_id != world.world_id:
                holder = self.router.get_world_by_id(ans.world_id)
                if holder is not None:
                    world, member, ms = holder, ans.axiom, ans.member_score
                    self.delegated += 1
        name = self._name(world)
        members = [fact_language(a) for a in world.axioms]
        if kind == "relation":
            return self._record(key, kind, self._relation(key, name, s1, margin, members))
        support = tuple(self._members_scored(world, ctx))
        if self._is_latent(world):                                 # what waits there is unverified
            return self._record(key, kind, Possibility(name, Verdict.UNKNOWN, ms, margin, support=support,
                                                       latent=name, why="latent_world"))
        if kind != "question":
            if key in members:
                return self._record(key, kind, Possibility(name, Verdict.POSSIBLE, 1.0, margin, support=support,
                                                           predicted=self._predicted(world, ctx), why="member"))
            if self.contradicts is not None:
                clash = self.contradicts(key, self.members_about(world, key))
                if clash is not None:
                    cs = next((s for m, s in support if m == clash), _cosine(ctx, self._vec(clash)))
                    return self._record(key, kind, Possibility(name, Verdict.IMPOSSIBLE, float(cs), margin,
                                                               support=((clash, float(cs)),), why="contradiction"))
                if self.cross_world:
                    for other in self.router.active_worlds():
                        if other is world or self._is_latent(other):
                            continue
                        clash = self.contradicts(key, self.members_about(other, key))
                        if clash is not None:
                            return self._record(key, kind, Possibility(self._name(other), Verdict.IMPOSSIBLE, s1, margin,
                                                                       support=((clash, s1),), why="contradiction_elsewhere"))
        if self.tau_answer is not None and ms >= self.tau_answer:
            above = tuple((m, s) for m, s in support if s >= self.tau_answer) or ((fact_language(member), float(ms)),)
            return self._record(key, kind, Possibility(name, Verdict.POSSIBLE, float(ms), margin, support=above,
                                                       predicted=self._predicted(world, ctx),
                                                       why="hits" if kind == "question" else "resembles"))
        return self._record(key, kind, Possibility(name, Verdict.UNKNOWN, s1, margin, support=support,
                                                   predicted=self._predicted(world, ctx), latent=name,
                                                   why="covered_unsupported"))

    def _relation(self, rel: str, name: str, s1: float, margin, members) -> Possibility:
        """A relation is possible in the routed world where >= 2 members state it (the reuse guard
        `StoreRelations` applies); one member is a paraphrase-tier answer, none is unknown."""
        stating = relation_members(rel, members)
        if len(stating) >= 2:
            return Possibility(name, Verdict.POSSIBLE, 1.0, margin, support=tuple((m, 1.0) for m in stating[:self.top_k]),
                               why="member")
        if stating:
            return Possibility(name, Verdict.POSSIBLE, s1, margin, support=((stating[0], 1.0),), why="paraphrase")
        return Possibility(name, Verdict.UNKNOWN, s1, margin, latent=name, why="unknown_relation")

    # -- `Knows`: the world Cubby can ASK (worlds.Worlds.mount) -- knowledge, never state ----------------
    def covers(self, question: str) -> float:
        _name, score, _margin = self.route(question)
        return max(0.0, float(score))

    def answer(self, question: str) -> list[str]:
        p = self.possible(question, kind="question")
        return [m for m, _s in p.support] if p.possible else []
