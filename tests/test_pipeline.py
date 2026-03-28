# tests/test_pipeline.py
"""Tests for MoWMPipeline — end-to-end orchestrator."""
from __future__ import annotations

import numpy as np

from mowm.pipeline import MoWMPipeline, StepResult

K, L = 4, 32


class TestMoWMPipeline:
    def setup_method(self):
        self.pipe = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=8, top_k=2, z_max=2, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"],
        )

    def test_creation(self):
        assert self.pipe is not None
        assert self.pipe.library is not None
        assert len(self.pipe.library) > 0

    def test_step_with_dict_state(self):
        result = self.pipe.step(
            state={"position": "origin", "velocity": "zero"},
            action="apply_force",
        )
        assert isinstance(result, StepResult)
        assert result.predicted_state.shape == (K, L)
        assert isinstance(result.q_value, float)
        assert isinstance(result.confidence, float)
        assert result.active_worlds >= 1

    def test_step_with_text_state(self):
        result = self.pipe.step(
            state="the ball is at position x=0",
            action="push right",
        )
        assert isinstance(result, StepResult)
        assert result.predicted_state.shape == (K, L)

    def test_step_with_array_state(self):
        bc = self.pipe.library.bc
        state_vec = bc.random_discrete(seed=10)
        action_vec = bc.random_discrete(seed=20)
        result = self.pipe.step(
            state=state_vec,
            action=action_vec,
        )
        assert result.predicted_state.shape == (K, L)

    def test_step_increments_step_count(self):
        self.pipe.step(state="s1", action="a1")
        self.pipe.step(state="s2", action="a2")
        assert self.pipe.step_count == 2

    def test_cleanup_happens_at_interval(self):
        for i in range(3):
            self.pipe.step(state=f"state_{i}", action=f"action_{i}")
        assert self.pipe.step_count == 3

    def test_rollout(self):
        results = self.pipe.rollout(
            state="initial_state",
            actions=["a1", "a2", "a3"],
            horizon=3,
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, StepResult)
            assert r.predicted_state.shape == (K, L)

    def test_auto_spawn_worlds(self):
        result = self.pipe.step(state="test", action="test_action")
        assert result.active_worlds >= 1

    def test_deterministic_with_seed(self):
        pipe1 = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=4, top_k=1, z_max=1, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"], seed=42,
        )
        pipe2 = MoWMPipeline(
            k=K, l=L, d_hidden=64, d_out=K * L,
            max_worlds=4, top_k=1, z_max=1, tau=0.5,
            cleanup_interval=2, gamma=0.99,
            domains=["physics"], seed=42,
        )
        r1 = pipe1.step(state="test_state", action="test_action")
        r2 = pipe2.step(state="test_state", action="test_action")
        np.testing.assert_array_equal(r1.predicted_state, r2.predicted_state)
