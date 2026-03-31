"""Oracle policy for L-navigate: moves to the intersection then up to goal.

Strategy:
  1. Move right until inside the vertical corridor.
  2. Move up toward the goal center.

This is guaranteed to succeed because the path stays entirely within the
L-shaped free space.
"""

import numpy as np
from numpy.typing import NDArray

from starter.envs.l_navigate import LNavigateConfig


class LNavigateOracle:
    """Deterministic oracle that solves the L-navigate environment."""

    def __init__(self, config: LNavigateConfig | None = None) -> None:
        self.cfg = config or LNavigateConfig()
        # x-target: center of the vertical corridor.
        self._corridor_x = (
            self.cfg.intersection_x_min + self.cfg.intersection_x_max
        ) / 2.0
        # Goal center.
        self._goal_x = (self.cfg.goal_x_min + self.cfg.goal_x_max) / 2.0
        self._goal_y = (self.cfg.goal_y_min + self.cfg.goal_y_max) / 2.0

    def act(self, obs: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return an action given the current observation [x, y]."""
        x, y = float(obs[0]), float(obs[1])
        s = self.cfg.max_step_size

        # Phase 1: Move right into the vertical corridor.
        if x < self.cfg.intersection_x_min + self.cfg.robot_radius + s:
            dx = np.clip(self._corridor_x - x, -s, s)
            dy = 0.0
        # Phase 2: Move toward the goal center.
        else:
            dx = np.clip(self._goal_x - x, -s, s)
            dy = np.clip(self._goal_y - y, -s, s)

        return np.array([dx, dy], dtype=np.float32)
