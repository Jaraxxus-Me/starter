"""RL training agents for starter environments."""

from omegaconf import DictConfig

from starter.policy.rl.agent import BaseRLAgent
from starter.policy.rl.ppo_agent import PPOAgent

__all__ = ["create_rl_agent"]


def create_rl_agent(
    agent_cfg: DictConfig,
    env_id: str,
    max_episode_steps: int,
    seed: int,
) -> BaseRLAgent:
    """Create agent based on configuration."""
    if agent_cfg.name == "ppo":
        return PPOAgent(seed, env_id, max_episode_steps, agent_cfg)
    raise ValueError(f"Unknown agent type: {agent_cfg.name}")
