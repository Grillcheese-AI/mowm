"""CubbyBridge — MoWM's adapter implementing CubbyLLM's WorldModelBridge Protocol.

Dependency points mowm -> cubbyllm (never the reverse): CubbyLLM ships the
Protocol + Challenge/ChallengeResult; this adapter drives MoWM's router/library
to answer a challenge, spawning a specialist world when none fits.
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
                 tau_match: float = 0.35) -> None:
        self.library = library
        self.router = router
        self.tau_match = tau_match
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
        for w in self.router._worlds.values():
            if w.world_id == world_id:
                return w
        return None

    def _best_match(self, ctx: np.ndarray):
        best, best_sim = None, -1.0
        for w in self.router._worlds.values():
            sim = _cosine(ctx, np.asarray(w.axiom_vec))
            if sim > best_sim:
                best, best_sim = w, sim
        return best, best_sim

    def attempt(self, challenge: Challenge) -> ChallengeResult:
        ctx = np.asarray(challenge.context, dtype=np.float32)
        action = np.zeros_like(ctx)                  # neutral action for predict
        key = self._key(challenge)

        # (a) exact reuse of a previously-answered challenge
        if key in self._specialist_for:
            world = self._world_by_id(self._specialist_for[key])
            pred, conf = world.predict(ctx, action)
            return ChallengeResult(pred, str(world.world_id), float(conf), False)

        # (b) a known world whose signature matches the challenge
        best, best_sim = self._best_match(ctx)
        if best is not None and best_sim >= self.tau_match:
            self._specialist_for[key] = best.world_id
            pred, conf = best.predict(ctx, action)
            return ChallengeResult(pred, str(best.world_id), float(conf), False)

        # (c) novel: grow a specialist from the nearest axioms
        world = self.router.maybe_spawn(ctx, self.library)
        if world is None:                            # pool full / no axioms
            raise RuntimeError("CubbyBridge: could not spawn a specialist")
        self._specialist_for[key] = world.world_id
        self.registered.append(world.world_id)
        pred, conf = world.predict(ctx, action)
        return ChallengeResult(pred, str(world.world_id), float(conf), True)

    # underlying channels (attempt() drives register_specialist via maybe_spawn)
    def push_context(self, ctx) -> None: ...
    def pull_context(self): return None
    def emit_novelty(self, event: object) -> None: ...
    def register_specialist(self, handle: object) -> None:
        self.registered.append(handle)
