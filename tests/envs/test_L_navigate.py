"""Tests for the L-navigate environment and oracle policy."""

import numpy as np
from gymnasium.wrappers import RecordVideo

from starter.envs.l_navigate import LNavigateConfig, LNavigateEnv
from starter.policy.oracles.L_navigate.oracle import LNavigateOracle
from tests.conftest import MAKE_VIDEOS


def test_env_reset_returns_valid_obs():
    """Reset returns an observation within the horizontal corridor."""
    env = LNavigateEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2,)
    cfg = env.cfg
    # Robot should be in the horizontal corridor (before intersection).
    assert cfg.robot_radius <= obs[0] <= cfg.intersection_x_min - cfg.robot_radius
    assert cfg.robot_radius <= obs[1] <= cfg.h_width - cfg.robot_radius


def test_env_step_returns_correct_shape():
    """Step returns the expected tuple structure."""
    env = LNavigateEnv()
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (2,)
    assert reward == -1.0
    assert isinstance(terminated, bool)
    assert truncated is False
    assert isinstance(info, dict)


def test_collision_blocks_movement():
    """Moving into the obstacle (upper-left block) is rejected."""
    env = LNavigateEnv()
    cfg = env.cfg
    # Place robot at the top-left of the horizontal corridor,
    # just inside the free space near the obstacle boundary.
    start_x = cfg.intersection_x_min - cfg.robot_radius - 0.01
    start_y = cfg.h_width - cfg.robot_radius - 0.01
    env.reset(seed=0, options={"init_pos": [start_x, start_y]})
    # Try to move up into the obstacle.
    obs, _, _, _, _ = env.step(np.array([0.0, cfg.max_step_size], dtype=np.float32))
    # Should be rejected — y should not increase past the corridor boundary.
    assert obs[1] <= cfg.h_width


def test_goal_termination():
    """Placing the robot inside the goal should terminate."""
    env = LNavigateEnv()
    cfg = env.cfg
    goal_x = (cfg.goal_x_min + cfg.goal_x_max) / 2
    goal_y = (cfg.goal_y_min + cfg.goal_y_max) / 2
    env.reset(seed=0, options={"init_pos": [goal_x, goal_y]})
    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated


def test_not_terminated_at_start():
    """Robot should not be in the goal at reset."""
    env = LNavigateEnv()
    env.reset(seed=0)
    # A no-op step from the start should not terminate.
    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not terminated


def test_render_rgb():
    """Rendering produces a valid RGB image."""
    env = LNavigateEnv(render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_oracle_solves_env_single():
    """Oracle reaches the goal from a single starting position."""
    env = LNavigateEnv(render_mode="rgb_array")
    oracle = LNavigateOracle()

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="l_navigate_oracle")


    obs, _ = env.reset(seed=42)
    for _ in range(500):
        action = oracle.act(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    env.close()
    assert terminated, f"Oracle failed to reach goal. Final position: {obs}"


def test_oracle_100_percent_success():
    """Oracle achieves 100% success rate over many random seeds."""
    num_episodes = 50
    max_steps = 500
    successes = 0
    env = LNavigateEnv()
    oracle = LNavigateOracle()

    for seed in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        terminated = False
        for _ in range(max_steps):
            action = oracle.act(obs)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        if terminated:
            successes += 1

    success_rate = successes / num_episodes
    assert (
        success_rate == 1.0
    ), f"Oracle success rate: {success_rate:.0%} ({successes}/{num_episodes})"


def test_custom_config():
    """Environment works with a non-default config."""
    cfg = LNavigateConfig(
        h_length=4.0,
        h_width=1.5,
        v_width=1.5,
        v_height=4.0,
        goal_x_min=2.5,
        goal_x_max=4.0,
        goal_y_min=3.5,
        goal_y_max=4.0,
    )
    env = LNavigateEnv(config=cfg)
    oracle = LNavigateOracle(config=cfg)
    obs, _ = env.reset(seed=0)
    for _ in range(1000):
        action = oracle.act(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated


def test_movement_along_l_path():
    """Robot can move right through horizontal corridor and up through vertical
    corridor."""
    env = LNavigateEnv()
    cfg = env.cfg
    # Start at left side of horizontal corridor.
    env.reset(seed=0, options={"init_pos": [0.5, 0.5]})

    # Move right.
    for _ in range(30):
        obs, _, _, _, _ = env.step(np.array([cfg.max_step_size, 0.0], dtype=np.float32))
    assert obs[0] > 2.0, "Robot should have moved right"

    # Move up.
    for _ in range(30):
        obs, _, _, _, _ = env.step(np.array([0.0, cfg.max_step_size], dtype=np.float32))
    assert obs[1] > 2.0, "Robot should have moved up in vertical corridor"
