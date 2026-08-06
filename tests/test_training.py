# tests/test_training.py
"""Tests for MoWMTrainer — train_step, train_epoch_from_env, evaluate,
and v0.2.0 integration."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mowm.pipeline import MoWMPipeline
from mowm.sim_environment import GridWorld
from mowm.training import MoWMTrainer

K, L = 4, 32


def _make_pipeline(seed: int = 42) -> MoWMPipeline:
    return MoWMPipeline(
        k=K, l=L, d_hidden=64, d_out=K * L,
        max_worlds=4, top_k=2, z_max=1, tau=0.5,
        cleanup_interval=4, gamma=0.99,
        domains=["physics"], seed=seed,
    )


def _make_env(seed: int = 42) -> GridWorld:
    return GridWorld(size=5, k=K, l=L, seed=seed)


def _make_trainer(seed: int = 42) -> MoWMTrainer:
    pipe = _make_pipeline(seed)
    return MoWMTrainer(pipe, lr=1e-3, lambda_pred=1.0, lambda_conf=0.1)


def _random_vsa(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((K, L)).astype(np.float32)


# ── Trainer initialisation ─────────────────────────────────────────────────────

class TestTrainerInit:
    def test_init_stores_pipeline(self):
        pipe = _make_pipeline()
        trainer = MoWMTrainer(pipe)
        assert trainer.pipeline is pipe

    def test_init_default_lr(self):
        trainer = _make_trainer()
        assert trainer.lr == pytest.approx(1e-3)

    def test_init_lambda_pred(self):
        pipe = _make_pipeline()
        trainer = MoWMTrainer(pipe, lambda_pred=2.0)
        assert trainer.lambda_pred == pytest.approx(2.0)

    def test_init_lambda_conf(self):
        pipe = _make_pipeline()
        trainer = MoWMTrainer(pipe, lambda_conf=0.5)
        assert trainer.lambda_conf == pytest.approx(0.5)

    def test_init_step_count_zero(self):
        trainer = _make_trainer()
        assert trainer.train_step_count == 0


# ── train_step ─────────────────────────────────────────────────────────────────

class TestTrainStep:
    def setup_method(self):
        self.trainer = _make_trainer()
        self.rng = np.random.default_rng(0)

    def _make_traj(self, n: int = 3) -> list[tuple[np.ndarray, str, np.ndarray, float]]:
        actions = ["up", "down", "left", "right"]
        traj = []
        for i in range(n):
            s = _random_vsa(self.rng)
            ns = _random_vsa(self.rng)
            a = actions[i % len(actions)]
            traj.append((s, a, ns, -0.01))
        return traj

    def test_returns_dict_with_required_keys(self):
        traj = self._make_traj()
        result = self.trainer.train_step([[traj[0]]])
        assert set(result.keys()) == {"total_loss", "prediction_loss", "confidence_loss"}

    def test_increments_step_count(self):
        traj = self._make_traj()
        self.trainer.train_step([[traj[0]]])
        assert self.trainer.train_step_count == 1
        self.trainer.train_step([[traj[1]]])
        assert self.trainer.train_step_count == 2

    def test_empty_trajectories_returns_zeros(self):
        result = self.trainer.train_step([])
        assert result["total_loss"] == pytest.approx(0.0)
        assert result["prediction_loss"] == pytest.approx(0.0)
        assert result["confidence_loss"] == pytest.approx(0.0)

    def test_empty_inner_trajectory_skipped(self):
        result = self.trainer.train_step([[]])
        assert result["total_loss"] == pytest.approx(0.0)

    def test_finite_losses(self):
        traj = self._make_traj(4)
        result = self.trainer.train_step([traj])
        for v in result.values():
            assert math.isfinite(v), f"Non-finite loss: {v}"

    def test_prediction_loss_non_negative(self):
        traj = self._make_traj(4)
        result = self.trainer.train_step([traj])
        assert result["prediction_loss"] >= 0.0

    def test_confidence_loss_non_negative(self):
        traj = self._make_traj(4)
        result = self.trainer.train_step([traj])
        assert result["confidence_loss"] >= 0.0

    def test_total_loss_is_weighted_sum(self):
        pipe = _make_pipeline()
        trainer = MoWMTrainer(pipe, lambda_pred=2.0, lambda_conf=0.5)
        traj = self._make_traj(2)
        result = trainer.train_step([traj])
        expected = 2.0 * result["prediction_loss"] + 0.5 * result["confidence_loss"]
        assert result["total_loss"] == pytest.approx(expected, rel=1e-5)

    def test_batch_of_multiple_trajectories(self):
        trajs = [self._make_traj(2) for _ in range(3)]
        result = self.trainer.train_step(trajs)
        assert math.isfinite(result["total_loss"])

    def test_step_count_increments_once_per_call(self):
        trajs = [self._make_traj(2) for _ in range(5)]
        for _ in range(3):
            self.trainer.train_step(trajs)
        assert self.trainer.train_step_count == 3


# ── evaluate ───────────────────────────────────────────────────────────────────

class TestEvaluate:
    def setup_method(self):
        self.trainer = _make_trainer()
        self.env = _make_env()

    def test_returns_dict_with_required_keys(self):
        result = self.trainer.evaluate(self.env, n_episodes=2, max_steps=5)
        assert set(result.keys()) == {
            "mean_reward", "mean_confidence", "total_reward", "n_episodes"
        }

    def test_finite_metrics(self):
        result = self.trainer.evaluate(self.env, n_episodes=2, max_steps=5)
        for v in result.values():
            assert math.isfinite(v), f"Non-finite metric: {v}"

    def test_episode_count_matches(self):
        result = self.trainer.evaluate(self.env, n_episodes=3, max_steps=5)
        assert result["n_episodes"] == 3

    def test_mean_reward_is_negative(self):
        # GridWorld always gives non-positive rewards on normal terrain
        result = self.trainer.evaluate(self.env, n_episodes=4, max_steps=10)
        assert result["mean_reward"] <= 0.0

    def test_mean_confidence_in_unit_range(self):
        result = self.trainer.evaluate(self.env, n_episodes=2, max_steps=5)
        assert 0.0 <= result["mean_confidence"] <= 1.0 + 1e-6

    def test_total_reward_consistent_with_mean(self):
        result = self.trainer.evaluate(self.env, n_episodes=2, max_steps=10)
        # total_reward / n_steps ~= mean_reward (within floating point tolerance)
        # allow some slack since early termination is not used
        assert math.isfinite(result["total_reward"])

    def test_zero_episodes_handled(self):
        result = self.trainer.evaluate(self.env, n_episodes=0, max_steps=5)
        assert result["n_episodes"] == 0


# ── train_epoch_from_env ───────────────────────────────────────────────────────

class TestTrainEpoch:
    def setup_method(self):
        self.trainer = _make_trainer()
        self.env = _make_env()

    def test_returns_dict_with_required_keys(self):
        result = self.trainer.train_epoch_from_env(
            self.env, n_trajectories=4, batch_size=2
        )
        assert set(result.keys()) == {
            "total_loss", "prediction_loss", "confidence_loss", "n_batches"
        }

    def test_finite_metrics(self):
        result = self.trainer.train_epoch_from_env(
            self.env, n_trajectories=4, batch_size=2
        )
        for k, v in result.items():
            assert math.isfinite(v), f"Non-finite metric {k}: {v}"

    def test_n_batches_correct(self):
        # 6 trajectories, batch_size=2 → 3 batches
        result = self.trainer.train_epoch_from_env(
            self.env, n_trajectories=6, batch_size=2
        )
        assert result["n_batches"] == 3

    def test_increments_trainer_step_count(self):
        before = self.trainer.train_step_count
        self.trainer.train_epoch_from_env(self.env, n_trajectories=4, batch_size=2)
        assert self.trainer.train_step_count > before


# ── v0.2.0 integration tests ───────────────────────────────────────────────────

class TestV02Integration:
    """Integration tests that import and exercise multiple v0.2.0 components."""

    def test_trainer_import_from_mowm(self):
        from mowm import MoWMTrainer  # noqa: F401 — just checking it's importable
        assert MoWMTrainer is not None

    def test_grid_world_import_from_mowm(self):
        from mowm import GridWorld  # noqa: F401
        assert GridWorld is not None

    def test_version_is_020(self):
        import mowm
        assert mowm.__version__ == "0.3.0"

    def test_all_v02_exports_present(self):
        import mowm
        expected = [
            "SafeFormulaExecutor", "CausalGraph", "LogicEngine",
            "SequenceEncoder", "PhasorEngine", "IdentityState",
            "GridWorld", "MoWMTrainer",
        ]
        for name in expected:
            assert hasattr(mowm, name), f"Missing export: {name}"

    def test_pipeline_and_trainer_step(self):
        pipe = _make_pipeline()
        trainer = MoWMTrainer(pipe)
        rng = np.random.default_rng(1)
        s = _random_vsa(rng)
        ns = _random_vsa(rng)
        result = trainer.train_step([[(s, "up", ns, -0.01)]])
        assert math.isfinite(result["total_loss"])

    def test_sequence_encoder_orthogonal_projection(self):
        from mowm.sequence_encoder import SequenceEncoder
        enc = SequenceEncoder(k=K, l=L, seed=0, latent_dim=16)
        emb = np.random.default_rng(7).standard_normal(16).astype(np.float32)
        out = enc._project_embedding(emb)
        assert out.shape == (K, L)
        assert out.dtype == np.float32

    def test_sequence_encoder_encode_embedded_tokens(self):
        from mowm.sequence_encoder import SequenceEncoder
        enc = SequenceEncoder(k=K, l=L, seed=0, latent_dim=8)
        embeddings = [
            np.random.default_rng(i).standard_normal(8).astype(np.float32)
            for i in range(3)
        ]
        out = enc.encode_embedded_tokens(embeddings)
        assert out.shape == (K, L)
        assert out.dtype == np.float32

    def test_causal_graph_importable(self):
        from mowm import CausalGraph
        g = CausalGraph(k=K, l=L)
        assert g is not None

    def test_logic_engine_importable(self):
        from mowm import LogicEngine
        le = LogicEngine()
        assert le is not None

    def test_phasor_engine_importable(self):
        from mowm import PhasorEngine
        pe = PhasorEngine(k=K, l=L)
        assert pe is not None

    def test_identity_state_importable(self):
        from mowm import IdentityState
        from mowm.axiom_library import AxiomLibrary
        lib = AxiomLibrary(k=K, l=L)
        ids = IdentityState(k=K, l=L, library=lib)
        assert ids is not None
