# mowm/training.py
"""MoWMTrainer — training loop for the Mixture of World Models pipeline.

Provides train_step (per-batch gradient-free loss computation),
train_epoch_from_env (full epoch using GridWorld trajectory generation),
and evaluate (episodic reward / confidence metrics).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mowm.pipeline import MoWMPipeline
    from mowm.sim_environment import GridWorld

_ACTIONS = ("up", "down", "left", "right")


class MoWMTrainer:
    """Trainer for MoWMPipeline with prediction and confidence losses.

    The trainer is gradient-free: losses are computed as scalar metrics used
    to track learning progress and guide world-model discovery rather than
    back-propagating gradients.

    Args:
        pipeline: Initialised :class:`~mowm.pipeline.MoWMPipeline` instance.
        lr: Learning-rate placeholder (reserved for future gradient passes).
        lambda_pred: Weight for the MSE prediction loss term.
        lambda_conf: Weight for the confidence penalty term.
    """

    def __init__(
        self,
        pipeline: MoWMPipeline,
        lr: float = 1e-3,
        lambda_pred: float = 1.0,
        lambda_conf: float = 0.1,
    ) -> None:
        self.pipeline = pipeline
        self.lr = lr
        self.lambda_pred = lambda_pred
        self.lambda_conf = lambda_conf
        self._train_step_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_step_count(self) -> int:
        """Number of train_step calls executed so far."""
        return self._train_step_count

    # ------------------------------------------------------------------
    # Core training methods
    # ------------------------------------------------------------------

    def train_step(
        self,
        trajectories: list[list[tuple[np.ndarray, str, np.ndarray, float]]],
    ) -> dict[str, float]:
        """Compute prediction and confidence losses for a batch of trajectories.

        Each trajectory is a list of ``(state, action, next_state, reward)``
        tuples where *state* and *next_state* are block-code VSA vectors of
        shape ``(k, l)`` and *action* is a string action name.

        Args:
            trajectories: Batch of trajectories.  Empty trajectories are
                silently skipped.

        Returns:
            Dictionary with keys ``total_loss``, ``prediction_loss``, and
            ``confidence_loss``.  All values are finite floats.
        """
        self._train_step_count += 1

        total_pred = 0.0
        total_conf = 0.0
        n_transitions = 0

        for traj in trajectories:
            for state_vec, action, next_state_vec, _reward in traj:
                result = self.pipeline.step(
                    state=state_vec,
                    action=action,
                    ground_truth=next_state_vec,
                )
                # MSE prediction loss (per element, then mean over trajectory)
                diff = result.predicted_state - next_state_vec
                pred_loss = float(np.mean(diff ** 2))

                # Confidence penalty: low confidence = high penalty
                conf_loss = float(1.0 - min(1.0, max(0.0, result.confidence)))

                total_pred += pred_loss
                total_conf += conf_loss
                n_transitions += 1

        if n_transitions == 0:
            return {"total_loss": 0.0, "prediction_loss": 0.0, "confidence_loss": 0.0}

        pred_mean = total_pred / n_transitions
        conf_mean = total_conf / n_transitions
        total = self.lambda_pred * pred_mean + self.lambda_conf * conf_mean

        return {
            "total_loss": float(total),
            "prediction_loss": float(pred_mean),
            "confidence_loss": float(conf_mean),
        }

    def train_epoch_from_env(
        self,
        env: GridWorld,
        n_trajectories: int = 100,
        batch_size: int = 10,
    ) -> dict[str, float]:
        """Run one training epoch using the GridWorld environment.

        Generates *n_trajectories* random trajectories from *env*, then
        processes them in mini-batches of *batch_size*, accumulating losses.

        Args:
            env: Configured :class:`~mowm.sim_environment.GridWorld` instance.
            n_trajectories: Total number of trajectories to generate.
            batch_size: Number of trajectories per train_step call.

        Returns:
            Dictionary with keys ``total_loss``, ``prediction_loss``,
            ``confidence_loss``, and ``n_batches``.
        """
        raw_trajectories = env.generate_trajectories(n_trajectories)

        # Convert GridWorld StepResult sequences to (state, action, next_state, reward)
        # tuples using consecutive pairs of StepResults.
        action_cycle = list(_ACTIONS)
        rng = np.random.default_rng(self.pipeline._seed)

        converted: list[list[tuple[np.ndarray, str, np.ndarray, float]]] = []
        for raw_traj in raw_trajectories:
            traj: list[tuple[np.ndarray, str, np.ndarray, float]] = []
            for i in range(len(raw_traj) - 1):
                sr_cur = raw_traj[i]
                sr_next = raw_traj[i + 1]
                action = action_cycle[int(rng.integers(0, len(action_cycle)))]
                state_vec = env._encoder.encode_state(
                    {"x": str(sr_cur.x), "y": str(sr_cur.y)}
                )
                next_vec = env._encoder.encode_state(
                    {"x": str(sr_next.x), "y": str(sr_next.y)}
                )
                traj.append((state_vec, action, next_vec, sr_next.reward))
            if traj:
                converted.append(traj)

        # Split into mini-batches
        total_loss = 0.0
        total_pred = 0.0
        total_conf = 0.0
        n_batches = 0

        for start in range(0, len(converted), batch_size):
            batch = converted[start : start + batch_size]
            if not batch:
                continue
            metrics = self.train_step(batch)
            total_loss += metrics["total_loss"]
            total_pred += metrics["prediction_loss"]
            total_conf += metrics["confidence_loss"]
            n_batches += 1

        if n_batches == 0:
            return {
                "total_loss": 0.0,
                "prediction_loss": 0.0,
                "confidence_loss": 0.0,
                "n_batches": 0,
            }

        return {
            "total_loss": total_loss / n_batches,
            "prediction_loss": total_pred / n_batches,
            "confidence_loss": total_conf / n_batches,
            "n_batches": n_batches,
        }

    def evaluate(
        self,
        env: GridWorld,
        n_episodes: int = 10,
        max_steps: int = 20,
    ) -> dict[str, float]:
        """Evaluate the pipeline on random GridWorld episodes.

        For each episode, resets the environment at a random position and
        runs up to *max_steps* steps using random actions.

        Args:
            env: Configured :class:`~mowm.sim_environment.GridWorld` instance.
            n_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.

        Returns:
            Dictionary with keys ``mean_reward``, ``mean_confidence``,
            ``total_reward``, ``n_episodes``.  All values are finite.
        """
        action_names = list(_ACTIONS)
        rng = np.random.default_rng(self.pipeline._seed + 1)

        total_reward = 0.0
        total_confidence = 0.0
        n_steps_total = 0

        for _ep in range(n_episodes):
            sx = int(rng.integers(0, env.size))
            sy = int(rng.integers(0, env.size))
            # Avoid walls
            attempts = 0
            while (sx, sy) in env._walls and attempts < 25:
                sx = int(rng.integers(0, env.size))
                sy = int(rng.integers(0, env.size))
                attempts += 1

            env.reset(sx, sy)

            for _step in range(max_steps):
                action = action_names[int(rng.integers(0, len(action_names)))]
                step_result = env.step(action)
                state_vec = env._encoder.encode_state(
                    {"x": str(step_result.x), "y": str(step_result.y)}
                )
                pipe_result = self.pipeline.step(state=state_vec, action=action)
                total_reward += step_result.reward
                total_confidence += pipe_result.confidence
                n_steps_total += 1

        if n_steps_total == 0:
            return {
                "mean_reward": 0.0,
                "mean_confidence": 0.0,
                "total_reward": 0.0,
                "n_episodes": n_episodes,
            }

        return {
            "mean_reward": total_reward / n_steps_total,
            "mean_confidence": total_confidence / n_steps_total,
            "total_reward": total_reward,
            "n_episodes": n_episodes,
        }
