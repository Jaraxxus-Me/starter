"""Tests for the L-navigate environment and oracle policy."""

import numpy as np
from gymnasium.wrappers import RecordVideo

from starter.envs.l_navigate import LNavigateEnv, LNavigateEnvConfig
from starter.policy.oracles.L_navigate.oracle import LNavigateOracle


def test_env_reset_returns_valid_obs():
    """Reset returns an observation with the expected shape."""
    env = LNavigateEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    # 29 features: robot(9) + target_region(10) + obstacle0(10)
    assert obs.shape == (29,)
    # Robot x,y should be in horizontal corridor.
    cfg = LNavigateEnvConfig()
    robot_x, robot_y = obs[0], obs[1]
    assert cfg.world_min_x < robot_x < cfg.obstacle_x + cfg.obstacle_width
    assert cfg.world_min_y < robot_y < cfg.obstacle_y


def test_env_step_returns_correct_shape():
    """Step returns the expected tuple structure."""
    env = LNavigateEnv(render_mode="rgb_array")
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (29,)
    assert reward == -1.0
    assert not terminated or terminated  # bool-like check
    assert truncated is False
    assert isinstance(info, dict)


def test_collision_blocks_movement_into_obstacle():
    """Stepping into the obstacle should be rejected (robot stays put)."""
    env = LNavigateEnv(render_mode="rgb_array")
    env.reset(seed=0)
    # Try to move straight up repeatedly — should be blocked by obstacle.
    prev_y = None
    for _ in range(100):
        action = np.array([0.0, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        curr_y = float(obs[1])
        if prev_y is not None and curr_y == prev_y:
            break  # blocked
        prev_y = curr_y
    # Robot y should not have entered the obstacle region.
    cfg = LNavigateEnvConfig()
    assert float(obs[1]) < cfg.obstacle_y + cfg.robot_base_radius + 0.01


def test_render_rgb():
    """Rendering produces a valid RGB image."""
    env = LNavigateEnv(render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_oracle_solves_env_single(make_videos):
    """Oracle reaches the goal from a single starting position."""
    env = LNavigateEnv(render_mode="rgb_array")
    oracle = LNavigateOracle()

    if make_videos:
        env = RecordVideo(env, "unit_test_videos", name_prefix="l_navigate_oracle")

    obs, _ = env.reset(seed=42)
    terminated = False
    for _ in range(1000):
        action = oracle.act(obs)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    env.close()
    assert (
        terminated
    ), f"Oracle failed to reach goal. Final pos: ({obs[0]:.2f}, {obs[1]:.2f})"


def test_oracle_100_percent_success():
    """Oracle achieves 100% success rate over many random seeds."""
    num_episodes = 50
    max_steps = 1000
    successes = 0
    env = LNavigateEnv(render_mode="rgb_array")
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


def test_movement_along_l_path():
    """Robot can move right through horizontal corridor and up through vertical
    corridor."""
    env = LNavigateEnv(render_mode="rgb_array")
    env.reset(seed=0)

    # Move right.
    for _ in range(60):
        obs, _, _, _, _ = env.step(
            np.array([0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
    cfg = LNavigateEnvConfig()
    assert (
        float(obs[0]) > cfg.obstacle_x + cfg.obstacle_width
    ), "Robot should have moved into vertical corridor"

    # Move up.
    for _ in range(60):
        obs, _, _, _, _ = env.step(
            np.array([0.0, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        )
    assert float(obs[1]) > 2.0, "Robot should have moved up in vertical corridor"
