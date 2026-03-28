# mowm/sim_environment.py
"""GridWorld simulation environment for MoWM trajectory generation.

Provides a 5x5 grid with walls, terrain types, and four cardinal actions.
States are encoded into block-code VSA vectors via WorldEncoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cubemind.execution.world_encoder import WorldEncoder

# ── Action definitions ────────────────────────────────────────────────────────
# Action name -> (dx, dy) displacement
_ACTIONS: dict[str, tuple[int, int]] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# ── Terrain reward shaping ────────────────────────────────────────────────────
_TERRAIN_REWARDS: dict[str, float] = {
    "normal": -0.01,
    "ice": 0.0,
    "mud": -0.1,
}

_DEFAULT_GRID_SIZE = 5


@dataclass
class StepResult:
    """Result of a single GridWorld step."""

    x: int
    y: int
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class GridWorld:
    """A 2-D grid world with walls, terrain, and VSA state encoding.

    The grid has coordinates ``(0, 0)`` at the bottom-left and
    ``(size-1, size-1)`` at the top-right.  Actions are clipped at
    boundaries and blocked by walls.

    Args:
        size: Side length of the square grid (default 5).
        k: Number of VSA blocks passed to WorldEncoder.
        l: VSA block length passed to WorldEncoder.
        seed: RNG seed for random trajectory generation.
    """

    def __init__(
        self,
        size: int = _DEFAULT_GRID_SIZE,
        k: int = 4,
        l: int = 32,  # noqa: E741
        seed: int = 42,
    ) -> None:
        self.size = size
        self.k = k
        self.l = l
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._encoder = WorldEncoder(k=k, l=l)

        # Agent position
        self._x: int = 0
        self._y: int = 0

        # Sets of impassable positions and terrain overrides
        self._walls: set[tuple[int, int]] = set()
        self._terrain: dict[tuple[int, int], str] = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def position(self) -> tuple[int, int]:
        return (self._x, self._y)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def add_wall(self, x: int, y: int) -> None:
        """Mark cell ``(x, y)`` as an impassable wall.

        Args:
            x: Column index (0-based).
            y: Row index (0-based).
        """
        self._walls.add((x, y))

    def set_terrain(self, x: int, y: int, terrain_type: str) -> None:
        """Assign a terrain type to cell ``(x, y)``.

        Args:
            x: Column index.
            y: Row index.
            terrain_type: One of ``"normal"``, ``"ice"``, or ``"mud"``.

        Raises:
            ValueError: If *terrain_type* is not recognised.
        """
        if terrain_type not in _TERRAIN_REWARDS:
            raise ValueError(
                f"Unknown terrain type {terrain_type!r}. "
                f"Valid types: {list(_TERRAIN_REWARDS)}"
            )
        self._terrain[(x, y)] = terrain_type

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self, x: int = 0, y: int = 0) -> StepResult:
        """Reset the agent to position ``(x, y)``.

        Args:
            x: Starting column index (default 0).
            y: Starting row index (default 0).

        Returns:
            A :class:`StepResult` reflecting the starting position.
        """
        self._x = max(0, min(x, self.size - 1))
        self._y = max(0, min(y, self.size - 1))
        terrain = self._terrain.get((self._x, self._y), "normal")
        return StepResult(
            x=self._x, y=self._y,
            reward=_TERRAIN_REWARDS[terrain],
            done=False,
        )

    def step(self, action: str) -> StepResult:
        """Take a step in direction *action*.

        The agent's position is updated if:
        1. The action is valid.
        2. The resulting cell is within grid boundaries.
        3. The resulting cell is not a wall.

        If any condition fails the agent stays put.  The reward is
        determined by the terrain of the *landing* cell (or the current
        cell when the move is blocked).

        Args:
            action: One of ``"up"``, ``"down"``, ``"left"``, ``"right"``.

        Returns:
            :class:`StepResult` with updated position and reward.

        Raises:
            ValueError: If *action* is not a recognised action name.
        """
        if action not in _ACTIONS:
            raise ValueError(
                f"Unknown action {action!r}. Valid actions: {list(_ACTIONS)}"
            )

        dx, dy = _ACTIONS[action]
        nx = self._x + dx
        ny = self._y + dy

        # Clip to grid boundaries
        nx = max(0, min(nx, self.size - 1))
        ny = max(0, min(ny, self.size - 1))

        # Block on wall
        if (nx, ny) not in self._walls:
            self._x = nx
            self._y = ny

        terrain = self._terrain.get((self._x, self._y), "normal")
        reward = _TERRAIN_REWARDS[terrain]
        return StepResult(x=self._x, y=self._y, reward=reward, done=False)

    def encode_state(self) -> np.ndarray:
        """Encode the current agent position as a VSA block-code vector.

        The state dictionary ``{"x": str(x), "y": str(y)}`` is passed to
        :meth:`cubemind.execution.world_encoder.WorldEncoder.encode_state`.

        Returns:
            Block-code vector of shape ``(k, l)``, dtype float32.
        """
        return self._encoder.encode_state({"x": str(self._x), "y": str(self._y)})

    def generate_trajectories(self, n: int) -> list[list[StepResult]]:
        """Generate *n* random trajectories from random starting positions.

        Each trajectory begins at a random unoccupied cell and takes
        ``size * 2`` random steps.

        Args:
            n: Number of trajectories to generate.

        Returns:
            List of *n* trajectories, each a list of :class:`StepResult` objects
            (including the initial reset result).
        """
        action_names = list(_ACTIONS.keys())
        trajectory_length = self.size * 2

        trajectories: list[list[StepResult]] = []
        for _ in range(n):
            # Pick a random starting cell that is not a wall
            while True:
                sx = int(self._rng.integers(0, self.size))
                sy = int(self._rng.integers(0, self.size))
                if (sx, sy) not in self._walls:
                    break

            traj: list[StepResult] = [self.reset(sx, sy)]
            for _ in range(trajectory_length):
                action = action_names[int(self._rng.integers(0, len(action_names)))]
                traj.append(self.step(action))
            trajectories.append(traj)

        return trajectories
