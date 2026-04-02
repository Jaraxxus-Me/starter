"""Tests for the TD3 agent and network components."""

import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig

from starter.policy.rl.td3_agent import (
    ReplayBuffer,
    TD3Actor,
    TD3Agent,
    TD3QNetwork,
)

# ------------------------------------------------------------------ #
# TD3QNetwork
# ------------------------------------------------------------------ #


def _make_dummy_spaces() -> tuple[spaces.Box, spaces.Box]:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return obs_space, act_space


def test_td3_qnetwork_output_shape():
    """Q-network returns a scalar Q-value per (obs, action) pair."""
    obs_space, act_space = _make_dummy_spaces()
    qnet = TD3QNetwork(obs_space, act_space, hidden_size=32)
    obs = torch.randn(4, 10)
    act = torch.randn(4, 3)
    q_values = qnet(obs, act)
    assert q_values.shape == (4, 1)


def test_td3_qnetwork_deterministic():
    """Q-network is deterministic for the same input."""
    obs_space, act_space = _make_dummy_spaces()
    qnet = TD3QNetwork(obs_space, act_space, hidden_size=32)
    obs = torch.randn(2, 10)
    act = torch.randn(2, 3)
    q1 = qnet(obs, act)
    q2 = qnet(obs, act)
    assert torch.allclose(q1, q2)


# ------------------------------------------------------------------ #
# TD3Actor
# ------------------------------------------------------------------ #


def test_td3_actor_output_shape():
    """Actor returns an action per observation."""
    obs_space, act_space = _make_dummy_spaces()
    actor = TD3Actor(obs_space, act_space, hidden_size=32)
    obs = torch.randn(4, 10)
    actions = actor(obs)
    assert actions.shape == (4, 3)


def test_td3_actor_deterministic():
    """Deterministic actor produces the same action for the same input."""
    obs_space, act_space = _make_dummy_spaces()
    actor = TD3Actor(obs_space, act_space, hidden_size=32)
    obs = torch.randn(1, 10)
    a1 = actor(obs)
    a2 = actor(obs)
    assert torch.allclose(a1, a2)


def test_td3_actor_action_bounds():
    """Actor actions are within the action space bounds."""
    obs_space, act_space = _make_dummy_spaces()
    actor = TD3Actor(obs_space, act_space, hidden_size=32)
    obs = torch.randn(100, 10)
    actions = actor(obs).detach().numpy()
    assert np.all(actions >= act_space.low - 1e-6)
    assert np.all(actions <= act_space.high + 1e-6)


# ------------------------------------------------------------------ #
# ReplayBuffer
# ------------------------------------------------------------------ #


def test_replay_buffer_add_and_sample():
    """Replay buffer stores and returns transitions."""
    obs_space, act_space = _make_dummy_spaces()
    device = torch.device("cpu")
    rb = ReplayBuffer(100, obs_space, act_space, device, n_envs=1)

    for _ in range(10):
        obs = np.random.randn(1, 10).astype(np.float32)
        next_obs = np.random.randn(1, 10).astype(np.float32)
        action = np.random.randn(1, 3).astype(np.float32)
        reward = np.array([1.0], dtype=np.float32)
        done = np.array([0.0], dtype=np.float32)
        rb.add(obs, next_obs, action, reward, done)

    s_obs, s_next_obs, s_act, s_rew, s_done = rb.sample(5)
    assert s_obs.shape == (5, 10)
    assert s_next_obs.shape == (5, 10)
    assert s_act.shape == (5, 3)
    assert s_rew.shape == (5,)
    assert s_done.shape == (5,)


# ------------------------------------------------------------------ #
# TD3Agent
# ------------------------------------------------------------------ #


def _make_short_td3_cfg() -> DictConfig:
    """Config for a very short training run."""
    return DictConfig(
        {
            "name": "td3",
            "cuda": False,
            "tf_log": False,
            "args": {
                "total_timesteps": 512,
                "num_envs": 1,
                "hidden_size": 32,
                "batch_size": 32,
                "buffer_size": 1000,
                "learning_starts": 100,
                "save_model": False,
                "async_envs": False,
                "learning_rate": 3e-4,
            },
        }
    )


def test_td3_agent_creates_spaces():
    """TD3Agent extracts obs/action spaces from the environment."""
    cfg = _make_short_td3_cfg()
    agent = TD3Agent(seed=0, env_id="LNavigate-v0", max_episode_steps=50, cfg=cfg)
    assert isinstance(agent.observation_space, spaces.Box)
    assert isinstance(agent.action_space, spaces.Box)


def test_td3_agent_short_training():
    """TD3Agent.train() completes and returns metrics."""
    cfg = _make_short_td3_cfg()
    agent = TD3Agent(seed=0, env_id="LNavigate-v0", max_episode_steps=50, cfg=cfg)
    metrics = agent.train(eval_episodes=3)
    assert "train" in metrics
    assert "eval" in metrics
    assert "episodic_return" in metrics["eval"]
    assert len(metrics["eval"]["episodic_return"]) >= 3
