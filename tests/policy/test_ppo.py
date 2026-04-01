"""Tests for the PPO agent and RL gym utilities."""

import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig

from starter.policy.rl.gym_utils import ENV_REGISTRY, make_env
from starter.policy.rl.ppo_agent import PPOAgent, PPONetwork

# ------------------------------------------------------------------ #
# gym_utils
# ------------------------------------------------------------------ #


def test_env_registry_contains_l_navigate():
    """The environment registry includes the L-navigate env."""
    assert "LNavigate-v0" in ENV_REGISTRY


def test_make_env_returns_callable():
    """make_env returns a thunk that produces a gym environment."""
    thunk = make_env("LNavigate-v0", max_episode_steps=50)
    env = thunk()
    obs, _ = env.reset(seed=0)
    assert obs.ndim == 1  # flattened
    env.close()


def test_make_env_wrapper_stack():
    """The wrapper stack normalises observations and rewards."""
    env = make_env("LNavigate-v0", max_episode_steps=50)()
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, _, _, _ = env.step(action)
    # Observations should be clipped to [-10, 10].
    assert float(np.max(np.abs(obs))) <= 10.0
    # Reward should be clipped to [-10, 10].
    assert -10.0 <= float(reward) <= 10.0
    env.close()


# ------------------------------------------------------------------ #
# PPONetwork
# ------------------------------------------------------------------ #


def _make_dummy_spaces() -> tuple[spaces.Box, spaces.Box]:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return obs_space, act_space


def test_ppo_network_get_value():
    """get_value returns a scalar value per observation."""
    obs_space, act_space = _make_dummy_spaces()
    net = PPONetwork(obs_space, act_space, hidden_size=32)
    obs = torch.randn(4, 10)
    values = net.get_value(obs)
    assert values.shape == (4, 1)


def test_ppo_network_get_action_deterministic():
    """Deterministic actions are reproducible."""
    obs_space, act_space = _make_dummy_spaces()
    net = PPONetwork(obs_space, act_space, hidden_size=32)
    obs = torch.randn(1, 10)
    a1 = net.get_action(obs, deterministic=True)
    a2 = net.get_action(obs, deterministic=True)
    assert torch.allclose(a1, a2)


def test_ppo_network_get_action_stochastic():
    """Stochastic sampling produces varying actions."""
    obs_space, act_space = _make_dummy_spaces()
    net = PPONetwork(obs_space, act_space, hidden_size=32)
    obs = torch.randn(1, 10)
    actions = [net.get_action(obs, deterministic=False) for _ in range(20)]
    # At least some should differ.
    differs = any(not torch.allclose(actions[0], a) for a in actions[1:])
    assert differs


def test_ppo_network_get_action_and_value():
    """get_action_and_value returns action, logprob, entropy, value."""
    obs_space, act_space = _make_dummy_spaces()
    net = PPONetwork(obs_space, act_space, hidden_size=32)
    obs = torch.randn(4, 10)
    action, logprob, entropy, value = net.get_action_and_value(obs)
    assert action.shape == (4, 3)
    assert logprob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4, 1)


# ------------------------------------------------------------------ #
# PPOAgent
# ------------------------------------------------------------------ #


def _make_short_ppo_cfg() -> DictConfig:
    """Config for a very short training run."""
    return DictConfig(
        {
            "name": "ppo",
            "cuda": False,
            "tf_log": False,
            "args": {
                "total_timesteps": 512,
                "num_envs": 2,
                "num_steps": 64,
                "num_minibatches": 2,
                "update_epochs": 2,
                "hidden_size": 32,
                "save_model": False,
                "async_envs": False,
            },
        }
    )


def test_ppo_agent_creates_spaces():
    """PPOAgent extracts obs/action spaces from the environment."""
    cfg = _make_short_ppo_cfg()
    agent = PPOAgent(seed=0, env_id="LNavigate-v0", max_episode_steps=50, cfg=cfg)
    assert isinstance(agent.observation_space, spaces.Box)
    assert isinstance(agent.action_space, spaces.Box)


def test_ppo_agent_short_training():
    """PPOAgent.train() completes and returns metrics."""
    cfg = _make_short_ppo_cfg()
    agent = PPOAgent(seed=0, env_id="LNavigate-v0", max_episode_steps=50, cfg=cfg)
    metrics = agent.train(eval_episodes=3)
    assert "train" in metrics
    assert "eval" in metrics
    assert "episodic_return" in metrics["eval"]
    assert len(metrics["eval"]["episodic_return"]) >= 3
