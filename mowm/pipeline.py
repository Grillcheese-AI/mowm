# mowm/pipeline.py
"""MoWMPipeline — end-to-end orchestrator for the mixture of world models.

Pipeline flow:
  1. HYLA Encoder: state, action -> block-code vectors
  2. Router: select top-k worlds
  3. Each world: predict(state_vec, action_vec)
  4. Mixture: weighted sum of predictions
  5. Cleanup: per-block argmax every C steps
  6. CVL: estimate Q-value
  7. HYLA Decoder: block-code -> predicted next state + value
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cubemind.execution.cvl import ContrastiveValueEstimator
from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes

from mowm.axiom_library import AxiomLibrary
from mowm.domains import seed_all
from mowm.router import MoWMRouter
from mowm.world import ArgmaxCleanup, World


@dataclass
class StepResult:
    """Result of a single pipeline step."""

    predicted_state: np.ndarray
    q_value: float
    confidence: float
    active_worlds: int
    discovered: bool


class MoWMPipeline:
    """End-to-end MoWM pipeline orchestrator.

    Manages the full predict cycle: encode inputs, route to expert worlds,
    merge predictions, cleanup, estimate Q-values, and decode.

    Args:
        k: Number of blocks.
        l: Block length.
        d_hidden: HYLA hidden dimension.
        d_out: HYLA output dimension (typically k * l).
        max_worlds: Maximum expert worlds in pool.
        top_k: Number of experts selected per step.
        z_max: Maximum recursion depth for sub-worlds.
        tau: Confidence threshold for sub-world spawning.
        cleanup_interval: Number of steps between cleanup operations.
        gamma: Discount factor for CVL.
        domains: List of domain names to seed (None = all registered).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int,
        l: int,  # noqa: E741
        d_hidden: int,
        d_out: int,
        max_worlds: int,
        top_k: int,
        z_max: int,
        tau: float,
        cleanup_interval: int,
        gamma: float,
        domains: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.cleanup_interval = cleanup_interval
        self._seed = seed

        self.bc = BlockCodes(k=k, l=l)
        self.encoder = WorldEncoder(k=k, l=l)
        self._cleanup = ArgmaxCleanup()

        self.library = AxiomLibrary(k=k, l=l, domains=domains)
        self._import_domains(domains)
        seed_all(self.library, domains=domains)

        self.router = MoWMRouter(
            k=k, l=l,
            max_worlds=max_worlds,
            top_k=top_k,
            tau_spawn=tau,
            tau_prune_steps=200,
            seed=seed,
        )

        self.cvl = ContrastiveValueEstimator(
            d_state=self.d_vsa,
            d_action=self.d_vsa,
            d_latent=min(128, self.d_vsa),
            d_rff=min(256, self.d_vsa * 2),
            gamma=gamma,
            seed=seed,
        )

        self._step_count: int = 0
        self._init_worlds(seed)

    @property
    def step_count(self) -> int:
        return self._step_count

    def _import_domains(self, domains: list[str] | None) -> None:
        all_domain_modules = [
            "physics", "logic", "economics", "social", "linguistics",
            "causal", "atomic", "bio", "music", "identity", "colors",
            "language", "historical", "ensemble",
        ]
        targets = domains if domains else all_domain_modules
        for name in targets:
            try:
                __import__(f"mowm.domains.{name}")
            except ImportError:
                pass

    def _init_worlds(self, seed: int) -> None:
        domain_groups = self.library.domains()
        for i, (domain_name, axioms) in enumerate(domain_groups.items()):
            if self.router.num_active_worlds >= self.router.max_worlds:
                break
            world_axioms = axioms[:10]
            world = World(
                world_id=i,
                k=self.k,
                l=self.l,
                n_hylas=2,
                z_depth=0,
                z_max=2,
                tau=0.5,
                axioms=world_axioms,
                seed=seed + i,
            )
            try:
                self.router.add_world(world)
            except ValueError:
                break

    def _encode_input(
        self,
        state: dict[str, str] | str | np.ndarray,
        action: str | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(state, np.ndarray):
            state_vec = state.astype(np.float32)
        elif isinstance(state, dict):
            state_vec = self.encoder.encode_state(state)
        else:
            state_vec = self.encoder.encode_action(str(state))

        if isinstance(action, np.ndarray):
            action_vec = action.astype(np.float32)
        else:
            action_vec = self.encoder.encode_action(str(action))

        return state_vec, action_vec

    def step(
        self,
        state: dict[str, str] | str | np.ndarray,
        action: str | np.ndarray,
        ground_truth: np.ndarray | None = None,
    ) -> StepResult:
        self._step_count += 1

        state_vec, action_vec = self._encode_input(state, action)

        routed = self.router.route(state_vec, action_vec)

        if not routed:
            new_world = self.router.maybe_spawn(state_vec, self.library)
            if new_world is not None:
                routed = [(new_world, 1.0)]

        predictions: list[np.ndarray] = []
        weights: list[float] = []
        total_confidence = 0.0

        for world, weight in routed:
            pred, conf = world.predict(state_vec, action_vec)
            predictions.append(pred)
            weights.append(weight)
            total_confidence += conf * weight
            world.mark_selected(self._step_count)

        if predictions:
            weighted_preds = [w * p for w, p in zip(weights, predictions)]
            stacked = np.stack(weighted_preds, axis=0)
            mixed = stacked.sum(axis=0).astype(np.float32)
            block_sums = mixed.sum(axis=-1, keepdims=True)
            block_sums = np.where(block_sums == 0, 1.0, block_sums)
            mixed = (mixed / block_sums).astype(np.float32)
        else:
            mixed = state_vec.copy()
            total_confidence = 0.0

        if self._step_count % self.cleanup_interval == 0:
            mixed = self._cleanup.cleanup(mixed)

        q_value = float(
            self.cvl.encode_state_action(
                state_vec.ravel(), action_vec.ravel()
            ).mean()
        )

        discovered = False
        if ground_truth is not None:
            residual = ground_truth - mixed
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > self.router.tau_spawn:
                new_world = self.router.maybe_spawn(residual, self.library)
                discovered = new_world is not None

        return StepResult(
            predicted_state=mixed,
            q_value=q_value,
            confidence=total_confidence,
            active_worlds=self.router.num_active_worlds,
            discovered=discovered,
        )

    def rollout(
        self,
        state: dict[str, str] | str | np.ndarray,
        actions: list[str | np.ndarray],
        horizon: int,
    ) -> list[StepResult]:
        results: list[StepResult] = []
        current_state = state

        for t in range(min(horizon, len(actions))):
            result = self.step(current_state, actions[t])
            results.append(result)
            current_state = result.predicted_state

        return results
