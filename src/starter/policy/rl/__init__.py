"""RL training agents for starter environments."""

from omegaconf import DictConfig

from starter.policy.rl.agent import BaseRLAgent
from starter.policy.rl.ppo_agent import PPOAgent
from starter.policy.rl.td3_agent import TD3Agent

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
    if agent_cfg.name == "td3":
        return TD3Agent(seed, env_id, max_episode_steps, agent_cfg)
    raise ValueError(f"Unknown agent type: {agent_cfg.name}")
