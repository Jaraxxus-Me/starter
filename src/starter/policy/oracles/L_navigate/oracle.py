"""Oracle policy for L-navigate: navigates the L-shaped corridor to the
goal.

Strategy:
  1. Rotate robot to theta=0 (arm points right, away from obstacle).
  2. Move right to a safe x inside the vertical corridor.
  3. Move up to the target region's bottom edge.
  4. Rotate to theta=-pi/2 (arm points down) for clearance from walls.
  5. Move toward the target center.

Using arm-right for the L-traverse and arm-down for the final approach
avoids collisions with both the obstacle and the right/top world walls.
"""

import numpy as np
from numpy.typing import NDArray

from starter.envs.l_navigate import LNavigateEnvConfig

# Observation indices for the vectorized LNavigateEnv.
_ROBOT_X = 0
_ROBOT_Y = 1
_ROBOT_THETA = 2
_TARGET_X = 9
_TARGET_Y = 10
_TARGET_WIDTH = 17
_TARGET_HEIGHT = 18


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class LNavigateOracle:
    """Deterministic oracle that solves the L-navigate environment.

    Works with the vectorized LNavigateEnv (ConstantObjectKinDEREnv).
    Action: 5D [dx, dy, dtheta, darm, vac].
    """

    def __init__(self, config: LNavigateEnvConfig | None = None) -> None:
        self.cfg = config or LNavigateEnvConfig()
        corridor_x_min = self.cfg.obstacle_x + self.cfg.obstacle_width
        corridor_x_max = self.cfg.world_max_x
        clearance = self.cfg.robot_base_radius + self.cfg.robot_arm_length + 0.05
        self._safe_x = (corridor_x_min + corridor_x_max) / 2.0
        self._safe_x = np.clip(
            self._safe_x, corridor_x_min + clearance, corridor_x_max - clearance
        )

    def act(self, obs: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return a 5D action given the observation vector."""
        x = float(obs[_ROBOT_X])
        y = float(obs[_ROBOT_Y])
        theta = float(obs[_ROBOT_THETA])
        target_bottom = float(obs[_TARGET_Y])
        s = self.cfg.max_dx
        s_theta = self.cfg.max_dtheta

        # Once at safe_x and above the target's bottom edge, switch to
        # arm-down so the arm does not collide with the right/top walls.
        in_approach_zone = (x >= self._safe_x - s) and (y >= target_bottom - s)

        if not in_approach_zone:
            # --- Navigate the L-shape with arm pointing right (theta=0) ---

            angle_err = _wrap_angle(0.0 - theta)
            if abs(angle_err) > s_theta:
                dtheta = np.clip(angle_err, -s_theta, s_theta)
                return np.array([0.0, 0.0, dtheta, 0.0, 0.0], dtype=np.float32)

            if abs(x - self._safe_x) > s:
                dx = np.clip(self._safe_x - x, -s, s)
                return np.array([dx, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            dy = np.clip(target_bottom - y, -s, s)
            return np.array([0.0, dy, 0.0, 0.0, 0.0], dtype=np.float32)

        # --- Approach target with arm pointing down (theta=-pi/2) ---

        angle_err = _wrap_angle(-np.pi / 2.0 - theta)
        if abs(angle_err) > s_theta:
            dtheta = np.clip(angle_err, -s_theta, s_theta)
            return np.array([0.0, 0.0, dtheta, 0.0, 0.0], dtype=np.float32)

        # Aim for the nearest interior point of the target so we only
        # move in directions that are actually needed, avoiding wall
        # collisions from unnecessary movement components.
        t_left = float(obs[_TARGET_X])
        t_bottom = float(obs[_TARGET_Y])
        t_right = t_left + float(obs[_TARGET_WIDTH])
        t_top = t_bottom + float(obs[_TARGET_HEIGHT])
        margin = s
        aim_x = np.clip(x, t_left + margin, t_right - margin)
        aim_y = np.clip(y, t_bottom + margin, t_top - margin)
        dx = np.clip(aim_x - x, -s, s)
        dy = np.clip(aim_y - y, -s, s)
        return np.array([dx, dy, 0.0, 0.0, 0.0], dtype=np.float32)
