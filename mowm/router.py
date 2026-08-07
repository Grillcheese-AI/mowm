# mowm/router.py
"""MoWMRouter — DSelect-k routing over a dynamic pool of expert worlds.

Wraps cubemind's DSelectKGate with spawn/prune logic for a dynamic number
of expert worlds. Maintains a max-capacity gate and masks inactive slots.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cubemind.ops.block_codes import BlockCodes
from cubemind.routing.moe_gate import DSelectKGate

from mowm.base import BaseRouter
from mowm.world import World

if TYPE_CHECKING:
    from mowm.axiom_library import Axiom, AxiomLibrary

_spawn_id_counter = itertools.count(start=5000)


@dataclass
class WorldAnswer:
    """Outcome of a router-mediated inter-world query.

    found is True when a world's best member cleared tau_answer; False means
    every consulted world was missing the answer and the result is the
    globally best member seen (the protocol's fallback).
    """

    world_id: int
    axiom: "Axiom | None"
    member_score: float
    worlds_consulted: int
    delegated: bool          # True when the answering world was not ranked first
    found: bool


class MoWMRouter(BaseRouter):
    """Dynamic expert routing over a pool of World instances.

    Uses DSelectKGate with a fixed max-capacity gate. Inactive world slots
    are masked to zero weight. Supports dynamic spawning and pruning.

    Args:
        k: Number of blocks.
        l: Block length.
        max_worlds: Maximum number of expert worlds in the pool.
        top_k: Number of experts to select per routing step.
        tau_spawn: Residual norm threshold for spawning new worlds.
        tau_prune_steps: Number of idle steps before a world is pruned.
        seed: Random seed for the gate.
    """

    def __init__(
        self,
        k: int,
        l: int,  # noqa: E741
        max_worlds: int,
        top_k: int = 2,
        tau_spawn: float = 0.5,
        tau_prune_steps: int = 100,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.max_worlds = max_worlds
        self.top_k = top_k
        self.tau_spawn = tau_spawn
        self.tau_prune_steps = tau_prune_steps

        self.bc = BlockCodes(k=k, l=l)

        self._gate = DSelectKGate(
            num_experts=max_worlds,
            k=min(top_k, max_worlds),
            gamma=1.0,
            seed=seed,
        )

        self._worlds: dict[int, World] = {}
        self._slot_to_world_id: dict[int, int] = {}
        self._world_id_to_slot: dict[int, int] = {}
        self._next_slot: int = 0

    @property
    def num_active_worlds(self) -> int:
        return len(self._worlds)

    def add_world(self, world: World) -> int:
        if len(self._worlds) >= self.max_worlds:
            raise ValueError(
                f"Cannot add world: pool at max_worlds={self.max_worlds}"
            )
        slot = self._next_slot
        self._next_slot += 1
        if slot >= self.max_worlds:
            used = set(self._worlds.keys())
            for s in range(self.max_worlds):
                if s not in used:
                    slot = s
                    break
            else:
                raise ValueError(
                    f"Cannot add world: max_worlds={self.max_worlds}"
                )

        self._worlds[slot] = world
        self._slot_to_world_id[slot] = world.world_id
        self._world_id_to_slot[world.world_id] = slot
        return slot

    def route(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
    ) -> list[tuple[World, float]]:
        if not self._worlds:
            return []

        input_flat = np.concatenate(
            [state_vec.ravel(), action_vec.ravel()]
        ).astype(np.float64)

        expert_weights = self._gate.forward(scores=input_flat)

        active_mask = np.zeros(self.max_worlds)
        for slot in self._worlds:
            if slot < self.max_worlds:
                active_mask[slot] = 1.0
        expert_weights = expert_weights * active_mask

        total = expert_weights.sum()
        if total > 0:
            expert_weights = expert_weights / total

        top_indices = np.argsort(-expert_weights)
        results: list[tuple[World, float]] = []
        for idx in top_indices:
            idx_int = int(idx)
            if idx_int in self._worlds and expert_weights[idx_int] > 1e-10:
                results.append(
                    (self._worlds[idx_int], float(expert_weights[idx_int]))
                )
                if len(results) >= self.top_k:
                    break

        return results

    def maybe_spawn(
        self,
        residual_vec: np.ndarray,
        library: "AxiomLibrary",
    ) -> World | None:
        residual_norm = float(np.linalg.norm(residual_vec))
        if residual_norm < self.tau_spawn:
            return None

        if len(self._worlds) >= self.max_worlds:
            return None

        selected_axioms = library.select(residual_vec, top_n=5)
        if not selected_axioms:
            return None

        world_id = next(_spawn_id_counter)
        new_world = World(
            world_id=world_id,
            k=self.k,
            l=self.l,
            n_hylas=2,
            z_depth=0,
            z_max=2,
            tau=0.5,
            axioms=selected_axioms,
            seed=world_id,
        )

        self.add_world(new_world)
        return new_world

    def prune(self, current_step: int) -> list[int]:
        pruned: list[int] = []
        slots_to_remove: list[int] = []

        for slot, world in self._worlds.items():
            idle_steps = current_step - max(world.last_selected_step, 0)
            if idle_steps > self.tau_prune_steps:
                pruned.append(world.world_id)
                slots_to_remove.append(slot)

        for slot in slots_to_remove:
            wid = self._slot_to_world_id.pop(slot, None)
            if wid is not None:
                self._world_id_to_slot.pop(wid, None)
            del self._worlds[slot]

        return pruned

    def entropy_reg(self) -> float:
        expert_weights = self._gate.forward()
        return float(self._gate.entropy_regularization(expert_weights))

    def query_worlds(
        self,
        ctx: np.ndarray,
        tau_answer: float,
        max_worlds: int | None = None,
    ) -> WorldAnswer | None:
        """Router-mediated inter-world query: a world missing the answer
        passes the query to the next world.

        The delegation protocol validated in CubbyLLM's
        exp_m3_temporal_causal screen: rank active worlds by cosine to their
        axiom CENTROID (the similarity-preserving bundle); consult the top
        world's individual members (``World.best_member``); if the best
        member clears ``tau_answer`` that world answers, otherwise the query
        passes to the next world; when every world has been consulted the
        globally best member seen answers (``found=False``). At its
        calibrated tau this matched exhaustive flat search's answer-found
        rate at ~60x fewer member comparisons with 2% of queries delegated
        — and tau_answer, like every tau in this project, is
        deployment-calibrated and does not transfer across corpora.

        Worlds never reference each other; the router mediates every hop
        (deny-by-default). Returns None when no worlds are active.
        """
        worlds = self.active_worlds()
        if not worlds:
            return None
        c = np.asarray(ctx, dtype=np.float64).ravel()
        cn = np.linalg.norm(c)

        def centroid_score(w: World) -> float:
            v = np.asarray(w.axiom_centroid, dtype=np.float64).ravel()
            vn = np.linalg.norm(v)
            if cn == 0.0 or vn == 0.0:
                return 0.0
            return float(c @ v / (cn * vn))

        ranked = sorted(worlds, key=centroid_score, reverse=True)
        if max_worlds is not None:
            ranked = ranked[:max_worlds]

        best: WorldAnswer | None = None
        for rank, w in enumerate(ranked):
            axiom, score = w.best_member(ctx)
            if best is None or score > best.member_score:
                best = WorldAnswer(world_id=w.world_id, axiom=axiom,
                                   member_score=score,
                                   worlds_consulted=rank + 1,
                                   delegated=rank > 0, found=False)
            if score >= tau_answer:
                return WorldAnswer(world_id=w.world_id, axiom=axiom,
                                   member_score=score,
                                   worlds_consulted=rank + 1,
                                   delegated=rank > 0, found=True)
        assert best is not None
        best.worlds_consulted = len(ranked)     # exhausted every world
        return best

    def get_world_by_id(self, world_id: int) -> World | None:
        slot = self._world_id_to_slot.get(world_id)
        if slot is None:
            return None
        return self._worlds.get(slot)

    def active_worlds(self) -> list[World]:
        return list(self._worlds.values())
