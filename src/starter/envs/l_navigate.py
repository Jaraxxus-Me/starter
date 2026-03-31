"""L-shaped 2D navigation environment.

A simple continuous navigation task where the free space forms an "L" shape: a
horizontal corridor joined to a vertical corridor. The robot must navigate from the
horizontal corridor to a goal at the top of the vertical corridor.

Follows the design of KinDER's Motion2D environment but implemented as a standalone
gymnasium environment with no heavy dependencies.
"""

from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray


@dataclass(frozen=True)
class LNavigateConfig:
    """Configuration for the L-navigate environment."""

    # Horizontal corridor: x in [0, h_length], y in [0, h_width].
    h_length: float = 3.0
    h_width: float = 1.0

    # Vertical corridor: x in [h_length - v_width, h_length],
    #                     y in [0, v_height].
    v_width: float = 1.0
    v_height: float = 3.0

    # Goal region at the top of the vertical corridor.
    goal_x_min: float = 2.0  # = h_length - v_width
    goal_x_max: float = 3.0  # = h_length
    goal_y_min: float = 2.5
    goal_y_max: float = 3.0  # = v_height

    # Robot config.
    robot_radius: float = 0.05
    max_step_size: float = 0.1

    # Rendering.
    render_dpi: int = 150

    def __post_init__(self) -> None:
        assert self.v_width <= self.h_length
        assert self.h_width <= self.v_height

    @property
    def intersection_x_min(self) -> float:
        """Left edge of the intersection region."""
        return self.h_length - self.v_width

    @property
    def intersection_x_max(self) -> float:
        """Right edge of the intersection region."""
        return self.h_length


def _default_config() -> LNavigateConfig:
    return LNavigateConfig()


@dataclass(frozen=True)
class LNavigateEnvParams:
    """All parameters for environment construction."""

    config: LNavigateConfig = field(default_factory=_default_config)


class LNavigateEnv(gymnasium.Env):
    """L-shaped 2D navigation environment.

    The free space is the union of:
      - Horizontal corridor: [0, h_length] x [0, h_width]
      - Vertical corridor:   [h_length - v_width, h_length] x [0, v_height]

    The robot starts uniformly in the horizontal corridor (excluding the
    intersection), and must reach a goal region at the top of the vertical
    corridor.

    Observation: [robot_x, robot_y]
    Action:      [dx, dy]  (clipped to max_step_size)
    Reward:      -1.0 per step
    Termination: robot center enters the goal region
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: LNavigateConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or LNavigateConfig()
        self.render_mode = render_mode

        # Observation: robot (x, y).
        self.observation_space = Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.cfg.h_length, self.cfg.v_height], dtype=np.float32),
        )

        # Action: (dx, dy).
        s = self.cfg.max_step_size
        self.action_space = Box(
            low=np.array([-s, -s], dtype=np.float32),
            high=np.array([s, s], dtype=np.float32),
        )

        self._robot_pos: NDArray[np.floating] = np.zeros(2, dtype=np.float32)

    def _is_in_free_space(self, x: float, y: float) -> bool:
        """Check whether position (x, y) is inside the L-shaped free space.

        Accounts for robot radius so the robot body stays fully inside.
        """
        r = self.cfg.robot_radius
        in_horizontal = (
            r <= x <= self.cfg.h_length - r and r <= y <= self.cfg.h_width - r
        )
        in_vertical = (
            self.cfg.intersection_x_min + r <= x <= self.cfg.h_length - r
            and r <= y <= self.cfg.v_height - r
        )
        return in_horizontal or in_vertical

    def _is_in_goal(self, x: float, y: float) -> bool:
        """Check whether the robot center is inside the goal region."""
        return (
            self.cfg.goal_x_min <= x <= self.cfg.goal_x_max
            and self.cfg.goal_y_min <= y <= self.cfg.goal_y_max
        )

    def _sample_start(self) -> NDArray[np.floating]:
        """Sample a start position in the horizontal corridor."""
        r = self.cfg.robot_radius
        # Horizontal corridor excluding intersection.
        x = self.np_random.uniform(r, self.cfg.intersection_x_min - r)
        y = self.np_random.uniform(r, self.cfg.h_width - r)
        return np.array([x, y], dtype=np.float32)

    def _get_obs(self) -> NDArray[np.floating]:
        return self._robot_pos.copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if options and "init_pos" in options:
            self._robot_pos = np.array(options["init_pos"], dtype=np.float32)
        else:
            self._robot_pos = self._sample_start()
        return self._get_obs(), {}

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        assert isinstance(self.action_space, Box)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_pos = self._robot_pos + action
        new_x, new_y = float(new_pos[0]), float(new_pos[1])

        if self._is_in_free_space(new_x, new_y):
            self._robot_pos = new_pos.astype(np.float32)

        terminated = self._is_in_goal(
            float(self._robot_pos[0]), float(self._robot_pos[1])
        )
        return self._get_obs(), -1.0, terminated, False, {}

    def render(self) -> NDArray[np.uint8] | None:  # type: ignore[override]
        """Render the environment as an RGB image."""
        if self.render_mode != "rgb_array":
            return None
        return self._render_rgb()

    def _render_rgb(self) -> NDArray[np.uint8]:
        # Lazy import so matplotlib is only needed for rendering.
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg")
        from matplotlib import patches  # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        cfg = self.cfg
        fig, ax = plt.subplots(
            1, 1, figsize=(cfg.h_length, cfg.v_height), dpi=cfg.render_dpi
        )

        # Background (obstacle / blocked region): upper-left block.
        ax.set_xlim(0, cfg.h_length)
        ax.set_ylim(0, cfg.v_height)
        ax.set_facecolor("#333333")

        # Draw free space.
        h_rect = patches.Rectangle((0, 0), cfg.h_length, cfg.h_width, facecolor="white")
        v_rect = patches.Rectangle(
            (cfg.intersection_x_min, 0), cfg.v_width, cfg.v_height, facecolor="white"
        )
        ax.add_patch(h_rect)
        ax.add_patch(v_rect)

        # Draw goal region.
        goal_rect = patches.Rectangle(
            (cfg.goal_x_min, cfg.goal_y_min),
            cfg.goal_x_max - cfg.goal_x_min,
            cfg.goal_y_max - cfg.goal_y_min,
            facecolor="#80008040",
            edgecolor="purple",
            linewidth=1,
        )
        ax.add_patch(goal_rect)

        # Draw robot.
        robot_circle = patches.Circle(
            (float(self._robot_pos[0]), float(self._robot_pos[1])),
            cfg.robot_radius,
            facecolor="dodgerblue",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(robot_circle)

        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0)

        # Convert to numpy array.
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
        img = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
        return img
