"""Overfitting tests for PPO and TD3 on the fixed L-navigate environment.

These tests verify that both algorithms can learn to solve a simplified (fixed-init,
fixed-goal) version of L-navigate within 200k env steps. They are slow (~2 minutes each)
and marked accordingly.
"""

import numpy as np
import pytest
from omegaconf import DictConfig

from starter.policy.rl.ppo_agent import PPOAgent
from starter.policy.rl.td3_agent import TD3Agent

# Mark all tests in this module as slow.
pytestmark = pytest.mark.slow

ENV_ID = "FixedLNavigate-v0"
MAX_EPISODE_STEPS = 100
EVAL_EPISODES = 10
MIN_SUCCESS_RATE = 0.8  # require >=80% eval success


def _success_rate(eval_steps: list[int]) -> float:
    return sum(1 for s in eval_steps if s < MAX_EPISODE_STEPS) / len(eval_steps)


def test_ppo_overfit_fixed_l_navigate():
    """PPO can overfit on FixedLNavigate-v0 within 100k env steps."""
    cfg = DictConfig(
        {
            "name": "ppo",
            "cuda": False,
            "tf_log": False,
            "args": {
                "total_timesteps": 100_000,
                "num_envs": 8,
                "num_steps": 128,
                "num_minibatches": 4,
                "update_epochs": 10,
                "hidden_size": 64,
                "save_model": False,
                "async_envs": False,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "ent_coef": 0.0,
                "gae_lambda": 0.95,
                "normalize": False,
            },
        }
    )
    agent = PPOAgent(
        seed=0,
        env_id=ENV_ID,
        max_episode_steps=MAX_EPISODE_STEPS,
        cfg=cfg,
    )
    metrics = agent.train(eval_episodes=EVAL_EPISODES)

    eval_steps = metrics["eval"]["step_length"]
    rate = _success_rate(eval_steps)
    mean_steps = np.mean(eval_steps)
    print(f"PPO overfit: success={rate:.0%}, mean_steps={mean_steps:.0f}")
    assert (
        rate >= MIN_SUCCESS_RATE
    ), f"PPO success rate {rate:.0%} < {MIN_SUCCESS_RATE:.0%}"


def test_td3_overfit_fixed_l_navigate():
    """TD3 can overfit on FixedLNavigate-v0 within 200k env steps."""
    cfg = DictConfig(
        {
            "name": "td3",
            "cuda": False,
            "tf_log": False,
            "args": {
                "total_timesteps": 50_000,
                "num_envs": 4,
                "hidden_size": 256,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 10_000,
                "save_model": False,
                "async_envs": False,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_frequency": 2,
                "exploration_noise": 0.1,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "normalize": False,
                "action_reg": 0.1,
            },
        }
    )
    agent = TD3Agent(
        seed=0,
        env_id=ENV_ID,
        max_episode_steps=MAX_EPISODE_STEPS,
        cfg=cfg,
    )
    metrics = agent.train(eval_episodes=EVAL_EPISODES)

    eval_steps = metrics["eval"]["step_length"]
    rate = _success_rate(eval_steps)
    mean_steps = np.mean(eval_steps)
    print(f"TD3 overfit: success={rate:.0%}, mean_steps={mean_steps:.0f}")
    assert (
        rate >= MIN_SUCCESS_RATE
    ), f"TD3 success rate {rate:.0%} < {MIN_SUCCESS_RATE:.0%}"
