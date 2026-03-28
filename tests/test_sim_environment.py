# tests/test_sim_environment.py
"""Tests for GridWorld — 5x5 simulation environment with walls and terrain."""
from __future__ import annotations

import numpy as np
import pytest

from mowm.sim_environment import GridWorld, StepResult, _ACTIONS, _TERRAIN_REWARDS


K, L = 4, 32


class TestInit:
    def test_default_size(self):
        gw = GridWorld(k=K, l=L)
        assert gw.size == 5

    def test_custom_size(self):
        gw = GridWorld(size=3, k=K, l=L)
        assert gw.size == 3

    def test_initial_position(self):
        gw = GridWorld(k=K, l=L)
        assert gw.x == 0
        assert gw.y == 0

    def test_position_property(self):
        gw = GridWorld(k=K, l=L)
        assert gw.position == (0, 0)


class TestReset:
    def test_reset_default_origin(self):
        gw = GridWorld(k=K, l=L)
        gw.step("right")  # move away first
        result = gw.reset()
        assert result.x == 0
        assert result.y == 0

    def test_reset_custom_position(self):
        gw = GridWorld(k=K, l=L)
        result = gw.reset(2, 3)
        assert result.x == 2
        assert result.y == 3

    def test_reset_returns_step_result(self):
        gw = GridWorld(k=K, l=L)
        result = gw.reset()
        assert isinstance(result, StepResult)

    def test_reset_clamps_out_of_bounds(self):
        gw = GridWorld(size=5, k=K, l=L)
        result = gw.reset(10, 10)
        assert result.x == 4
        assert result.y == 4


class TestStep:
    def setup_method(self):
        self.gw = GridWorld(size=5, k=K, l=L)
        self.gw.reset(2, 2)

    def test_step_up(self):
        result = self.gw.step("up")
        assert result.y == 3
        assert result.x == 2

    def test_step_down(self):
        result = self.gw.step("down")
        assert result.y == 1
        assert result.x == 2

    def test_step_left(self):
        result = self.gw.step("left")
        assert result.x == 1
        assert result.y == 2

    def test_step_right(self):
        result = self.gw.step("right")
        assert result.x == 3
        assert result.y == 2

    def test_boundary_clip_bottom(self):
        self.gw.reset(0, 0)
        result = self.gw.step("down")
        assert result.y == 0  # clipped

    def test_boundary_clip_left(self):
        self.gw.reset(0, 0)
        result = self.gw.step("left")
        assert result.x == 0  # clipped

    def test_boundary_clip_top(self):
        self.gw.reset(0, 4)
        result = self.gw.step("up")
        assert result.y == 4  # clipped

    def test_boundary_clip_right(self):
        self.gw.reset(4, 0)
        result = self.gw.step("right")
        assert result.x == 4  # clipped

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Unknown action"):
            self.gw.step("fly")

    def test_step_returns_step_result(self):
        result = self.gw.step("up")
        assert isinstance(result, StepResult)

    def test_default_reward_normal_terrain(self):
        result = self.gw.step("right")
        assert result.reward == pytest.approx(_TERRAIN_REWARDS["normal"])


class TestEncodeState:
    def test_encode_state_shape(self):
        gw = GridWorld(k=K, l=L)
        gw.reset(1, 2)
        vec = gw.encode_state()
        assert vec.shape == (K, L)

    def test_encode_state_dtype(self):
        gw = GridWorld(k=K, l=L)
        vec = gw.encode_state()
        assert vec.dtype == np.float32

    def test_encode_state_nonzero(self):
        gw = GridWorld(k=K, l=L)
        gw.reset(3, 3)
        assert np.any(gw.encode_state() != 0)

    def test_encode_state_differs_by_position(self):
        gw = GridWorld(k=K, l=L)
        gw.reset(0, 0)
        v1 = gw.encode_state()
        gw.reset(4, 4)
        v2 = gw.encode_state()
        assert not np.array_equal(v1, v2)


class TestWalls:
    def test_add_wall_blocks_movement(self):
        gw = GridWorld(size=5, k=K, l=L)
        gw.reset(2, 2)
        gw.add_wall(3, 2)
        result = gw.step("right")
        assert result.x == 2  # blocked

    def test_wall_does_not_block_other_direction(self):
        gw = GridWorld(size=5, k=K, l=L)
        gw.reset(2, 2)
        gw.add_wall(3, 2)
        result = gw.step("up")
        assert result.y == 3  # unaffected

    def test_wall_at_boundary_still_clips(self):
        gw = GridWorld(size=5, k=K, l=L)
        gw.reset(0, 0)
        gw.add_wall(0, 1)  # wall above origin
        result = gw.step("up")
        assert result.y == 0  # blocked by wall


class TestTerrain:
    def test_ice_terrain_reward(self):
        gw = GridWorld(size=5, k=K, l=L)
        gw.set_terrain(3, 2, "ice")
        gw.reset(2, 2)
        result = gw.step("right")  # land on (3,2)
        assert result.reward == pytest.approx(_TERRAIN_REWARDS["ice"])

    def test_mud_terrain_reward(self):
        gw = GridWorld(size=5, k=K, l=L)
        gw.set_terrain(3, 2, "mud")
        gw.reset(2, 2)
        result = gw.step("right")  # land on (3,2)
        assert result.reward == pytest.approx(_TERRAIN_REWARDS["mud"])

    def test_unknown_terrain_raises(self):
        gw = GridWorld(size=5, k=K, l=L)
        with pytest.raises(ValueError, match="Unknown terrain"):
            gw.set_terrain(1, 1, "lava")


class TestGenerateTrajectories:
    def test_returns_n_trajectories(self):
        gw = GridWorld(size=5, k=K, l=L, seed=0)
        trajs = gw.generate_trajectories(3)
        assert len(trajs) == 3

    def test_trajectory_length(self):
        gw = GridWorld(size=5, k=K, l=L, seed=0)
        trajs = gw.generate_trajectories(2)
        expected_len = 1 + gw.size * 2  # reset + steps
        for traj in trajs:
            assert len(traj) == expected_len

    def test_trajectory_contains_step_results(self):
        gw = GridWorld(size=5, k=K, l=L, seed=1)
        trajs = gw.generate_trajectories(1)
        for result in trajs[0]:
            assert isinstance(result, StepResult)

    def test_positions_within_bounds(self):
        gw = GridWorld(size=5, k=K, l=L, seed=2)
        trajs = gw.generate_trajectories(5)
        for traj in trajs:
            for result in traj:
                assert 0 <= result.x < gw.size
                assert 0 <= result.y < gw.size

    def test_zero_trajectories(self):
        gw = GridWorld(size=5, k=K, l=L)
        assert gw.generate_trajectories(0) == []
