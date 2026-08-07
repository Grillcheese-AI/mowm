"""CubbyBridge — MoWM's adapter implementing CubbyLLM's WorldModelBridge Protocol.

Dependency points mowm -> cubbyllm (never the reverse): CubbyLLM ships the
Protocol + Challenge/ChallengeResult; this adapter drives MoWM's router/library
to answer a challenge, spawning a specialist world when none fits.

Routing rule history (each step validated in CubbyLLM's validation/logs/):
M1 routed on the axiom_vec bind-chain — measured inert (quasi-orthogonal to
its own parts) and re-based onto individual axioms. The M3 open-set screen
(exp_m3_open_set) then showed CENTROID routing beats max-over-individual-
axioms 0.680 vs 0.589 macro at ~40x fewer comparisons, and that a MARGIN gate
(best minus second-best world) is the right route-vs-spawn signal: at its
calibrated tau it routed with 92% precision while sending 87% of never-seen-
domain challenges to spawn. Tier (b) below implements exactly that: route iff
the best centroid score clears the absolute floor AND (with 2+ worlds) the
margin clears tau_margin; everything else falls through to spawn — the safe
side, since a spawned specialist is a recoverable outcome and a mis-route is
not. Tier (b2) adds the inter-world query validated in
exp_m3_temporal_causal: when the centroid winner's own best MEMBER match
falls short of tau_answer (the world routes well but is missing the specific
answer), the router forwards the query world-to-world and the bridge
redirects to the world that actually holds it — at the screen's calibrated
tau this matched exhaustive flat search's answer-found rate at ~60x fewer
member comparisons, delegating 2% of queries. All taus are
deployment-calibrated (the screens are the method); the defaults preserve
the toy-scale test semantics.
"""
from __future__ import annotations

import numpy as np

from cubbyllm.bridges.world_model import Challenge, ChallengeResult

from ..axiom_library import AxiomLibrary
from ..router import MoWMRouter


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


class CubbyBridge:
    def __init__(self, library: AxiomLibrary, router: MoWMRouter,
                 tau_match: float = 0.35, tau_margin: float = 0.02,
                 tau_answer: float | None = 0.6) -> None:
        self.library = library
        self.router = router
        self.tau_match = tau_match          # absolute floor on the best centroid score
        self.tau_margin = tau_margin        # best-minus-second gate (2+ worlds)
        # member-level inter-world query threshold (exp_m3_temporal_causal's
        # validated operating point); None disables delegation entirely.
        self.tau_answer = tau_answer
        self._specialist_for: dict[str, int] = {}   # challenge key -> world_id
        self.registered: list[int] = []

    @property
    def num_worlds(self) -> int:
        return self.router.num_active_worlds

    def _key(self, challenge: Challenge) -> str:
        return challenge.tag if challenge.tag is not None else repr(
            np.asarray(challenge.context).tobytes()
        )

    def _world_by_id(self, world_id: int):
        return self.router.get_world_by_id(world_id)

    def _world_scores(self, ctx: np.ndarray) -> list[tuple[object, float]]:
        """Active worlds scored by cosine to their AXIOM CENTROID, best first.

        The centroid (normalized mean of a world's individual axioms) is the
        similarity-preserving bundled signature — one comparison per world
        instead of one per axiom, and measurably better routing
        (exp_m3_open_set: 0.680 vs 0.589 macro). The bind-chain ``axiom_vec``
        remains unusable for routing (the M1 finding).
        """
        scores = [(w, _cosine(ctx, np.asarray(w.axiom_centroid)))
                  for w in self.router.active_worlds()]
        scores.sort(key=lambda ws: -ws[1])
        return scores

    def attempt(self, challenge: Challenge) -> ChallengeResult:
        ctx = np.asarray(challenge.context, dtype=np.float32)
        action = np.zeros_like(ctx)                  # neutral action for predict
        key = self._key(challenge)

        # (a) exact reuse of a previously-answered challenge
        if key in self._specialist_for:
            world = self._world_by_id(self._specialist_for[key])
            if world is not None:
                pred, conf = world.predict(ctx, action)
                return ChallengeResult(pred, str(world.world_id), float(conf), False)
            # the memoized world was pruned since; drop the stale entry and
            # fall through to re-route (b) or re-spawn (c) a fresh specialist.
            del self._specialist_for[key]

        # (b) centroid routing with the margin gate: route only when one world
        # clearly wins. Ambiguity (small margin) falls through to spawn — the
        # recoverable outcome — matching the validated conservative profile.
        scores = self._world_scores(ctx)
        if scores:
            best, s1 = scores[0]
            margin_ok = len(scores) < 2 or (s1 - scores[1][1]) >= self.tau_margin
            if s1 >= self.tau_match and margin_ok:
                world = best
                # (b2) inter-world query: the centroid winner may still be
                # MISSING the specific answer. If its best member falls short
                # of tau_answer, the router forwards the query world-to-world
                # (exp_m3_temporal_causal's delegation protocol); the bridge
                # redirects only when another world actually CLEARED the
                # threshold — an exhausted query stays with the centroid
                # winner, and route-vs-spawn is never affected.
                if self.tau_answer is not None:
                    _, own_score = best.best_member(ctx)
                    if own_score < self.tau_answer:
                        ans = self.router.query_worlds(ctx, self.tau_answer)
                        if ans is not None and ans.found:
                            holder = self._world_by_id(ans.world_id)
                            if holder is not None:
                                world = holder
                self._specialist_for[key] = world.world_id
                pred, conf = world.predict(ctx, action)
                return ChallengeResult(pred, str(world.world_id), float(conf), False)

        # (c) novel: grow a specialist from the nearest axioms
        world = self.router.maybe_spawn(ctx, self.library)
        if world is None:                            # pool full / no axioms
            raise RuntimeError("CubbyBridge: could not spawn a specialist")
        self._specialist_for[key] = world.world_id
        self.register_specialist(world.world_id)
        pred, conf = world.predict(ctx, action)
        return ChallengeResult(pred, str(world.world_id), float(conf), True)

    # Remaining WorldModelBridge Protocol channels. attempt() is the primary
    # entry and routes spawned world_ids through register_specialist() below
    # (the one write-path to self.registered); these satisfy the rest of the
    # Protocol for callers that use them.
    def push_context(self, ctx) -> None: ...
    def pull_context(self): return None
    def emit_novelty(self, event: object) -> None: ...
    def register_specialist(self, handle: object) -> None:
        self.registered.append(handle)
