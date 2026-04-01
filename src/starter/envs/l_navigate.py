"""L-shaped 2D navigation environment.

A simple continuous navigation task where one large rectangular obstacle in the upper-
left corner creates an L-shaped free space. The robot starts in the horizontal part of
the L and must reach a goal at the top of the vertical part.

Inherits from KinDER's Motion2D environment.
"""

from dataclasses import dataclass

import numpy as np
from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic2d.base_env import (
    Kinematic2DRobotEnvConfig,
    ObjectCentricKinematic2DRobotEnv,
)
from kinder.envs.kinematic2d.motion2d import TargetRegionType
from kinder.envs.kinematic2d.object_types import (
    CRVRobotType,
    Kinematic2DRobotEnvTypeFeatures,
    RectangleType,
)
from kinder.envs.kinematic2d.structs import SE2Pose, ZOrder
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace
from kinder.envs.utils import BLACK, PURPLE, sample_se2_pose, state_2d_has_collision
from relational_structs import Object, ObjectCentricState


@dataclass(frozen=True)
class LNavigateEnvConfig(Kinematic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for the L-navigate environment.

    The world is [0, 3] x [0, 3]. A single obstacle occupies the upper-left
    region [0, 2] x [1, 3], leaving an L-shaped free space:
      - Horizontal corridor: [0, 3] x [0, 1]
      - Vertical corridor:   [2, 3] x [0, 3]
    """

    # World boundaries.
    world_min_x: float = 0.0
    world_max_x: float = 3.0
    world_min_y: float = 0.0
    world_max_y: float = 3.0

    # Action space parameters (same as Motion2D).
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # Robot hyperparameters.
    robot_base_radius: float = 0.1
    robot_arm_length: float = 2 * robot_base_radius
    robot_gripper_height: float = 0.07
    robot_gripper_width: float = 0.01

    # Robot starts in the horizontal corridor (left side).
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 2.5 * robot_base_radius,
            world_min_y + 2.5 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            1.8,  # stay well left of the vertical corridor
            1.0 - 2.5 * robot_base_radius,
            np.pi,
        ),
    )

    # Target region at the top of the vertical corridor.
    target_region_rgb: tuple[float, float, float] = PURPLE
    target_region_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(2.25, 2.5, 0),
        SE2Pose(2.75, 2.75, 0),
    )
    target_region_shape: tuple[float, float] = (0.5, 0.5)

    # Single obstacle: upper-left block creating the L-shape.
    obstacle_rgb: tuple[float, float, float] = BLACK
    obstacle_x: float = 0.0
    obstacle_y: float = 1.0
    obstacle_width: float = 2.0
    obstacle_height: float = 2.0

    # Rendering.
    render_dpi: int = 300
    render_fps: int = 20


class ObjectCentricLNavigateEnv(
    ObjectCentricKinematic2DRobotEnv[LNavigateEnvConfig],
):
    """L-shaped navigation: object-centric version.

    One rectangular obstacle in the upper-left creates an L-shaped free space.
    The robot must navigate from the horizontal corridor to a goal at the top
    of the vertical corridor.
    """

    def __init__(
        self,
        config: LNavigateEnvConfig = LNavigateEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)

    def _sample_initial_state(self) -> ObjectCentricState:
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        target_region_pose = sample_se2_pose(
            self.config.target_region_init_bounds, self.np_random
        )

        # Single obstacle forming the L-shape.
        obstacle_pose = SE2Pose(self.config.obstacle_x, self.config.obstacle_y, 0.0)
        obstacle_shape = (self.config.obstacle_width, self.config.obstacle_height)
        obstacles = [(obstacle_pose, obstacle_shape)]

        state = self._create_initial_state(robot_pose, target_region_pose, obstacles)

        # Sanity check: robot and target should not collide with anything.
        robot = state.get_objects(CRVRobotType)[0]
        target_region = state.get_objects(TargetRegionType)[0]
        assert not state_2d_has_collision(state, {robot, target_region}, set(state), {})

        return state

    def _create_constant_initial_state_dict(
        self,
    ) -> dict[Object, dict[str, float]]:
        # Reuse Motion2D's wall creation.
        from kinder.envs.kinematic2d.utils import (  # pylint: disable=import-outside-toplevel
            create_walls_from_world_boundaries,
        )

        init_state_dict: dict[Object, dict[str, float]] = {}
        assert isinstance(self.action_space, CRVRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)
        return init_state_dict

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        target_region_pose: SE2Pose | None = None,
        obstacles: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
    ) -> ObjectCentricState:
        # Delegate to Motion2D's state creation pattern.
        from relational_structs.utils import (  # pylint: disable=import-outside-toplevel
            create_state_from_dict,
        )

        init_state_dict: dict[Object, dict[str, float]] = {}

        # Robot.
        robot = Object("robot", CRVRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,
            "arm_length": self.config.robot_arm_length,
            "vacuum": 0.0,
            "gripper_height": self.config.robot_gripper_height,
            "gripper_width": self.config.robot_gripper_width,
        }

        # Target region.
        if target_region_pose is not None:
            target_region = Object("target_region", TargetRegionType)
            init_state_dict[target_region] = {
                "x": target_region_pose.x,
                "y": target_region_pose.y,
                "theta": target_region_pose.theta,
                "width": self.config.target_region_shape[0],
                "height": self.config.target_region_shape[1],
                "static": True,
                "color_r": self.config.target_region_rgb[0],
                "color_g": self.config.target_region_rgb[1],
                "color_b": self.config.target_region_rgb[2],
                "z_order": ZOrder.NONE.value,
            }

        # Obstacles.
        if obstacles:
            for i, (obstacle_pose, obstacle_shape) in enumerate(obstacles):
                obstacle = Object(f"obstacle{i}", RectangleType)
                init_state_dict[obstacle] = {
                    "x": obstacle_pose.x,
                    "y": obstacle_pose.y,
                    "theta": obstacle_pose.theta,
                    "width": obstacle_shape[0],
                    "height": obstacle_shape[1],
                    "static": True,
                    "color_r": self.config.obstacle_rgb[0],
                    "color_g": self.config.obstacle_rgb[1],
                    "color_b": self.config.obstacle_rgb[2],
                    "z_order": ZOrder.ALL.value,
                }

        return create_state_from_dict(init_state_dict, Kinematic2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        from kinder.envs.kinematic2d.utils import (  # pylint: disable=import-outside-toplevel
            rectangle_object_to_geom,
        )

        assert self._current_state is not None
        robot = self._current_state.get_objects(CRVRobotType)[0]
        x = self._current_state.get(robot, "x")
        y = self._current_state.get(robot, "y")
        target_region = self._current_state.get_objects(TargetRegionType)[0]
        target_region_geom = rectangle_object_to_geom(
            self._current_state,
            target_region,
            self._static_object_body_cache,
        )
        terminated = target_region_geom.contains_point(x, y)
        return -1.0, terminated


class LNavigateEnv(ConstantObjectKinDEREnv):
    """L-navigate env with constant number of objects and Box spaces."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return ObjectCentricLNavigateEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "target_region"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstacle"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        return (
            "An L-shaped 2D navigation environment. A single rectangular "
            "obstacle in the upper-left corner creates an L-shaped free space. "
            "The robot must navigate from the horizontal corridor to a goal "
            "at the top of the vertical corridor.\n"
        )

    def _create_reward_markdown_description(self) -> str:
        return (
            "A penalty of -1.0 is given at every time step until termination, "
            "which occurs when the robot's position is within the target "
            "region.\n"
        )

    def _create_references_markdown_description(self) -> str:
        return "L-shaped corridors are a basic test for navigation planning.\n"
