"""Utilities for creating and wrapping gymnasium environments."""

from typing import Callable

import gymnasium as gym
import numpy as np

from starter.envs.l_navigate import FixedLNavigateEnv, LNavigateEnv

# Registry mapping env_id strings to environment classes.
ENV_REGISTRY: dict[str, type[gym.Env]] = {
    "LNavigate-v0": LNavigateEnv,
    "FixedLNavigate-v0": FixedLNavigateEnv,
}


def make_env(
    env_id: str,
    max_episode_steps: int,
    gamma: float = 0.99,
    normalize: bool = True,
) -> Callable[[], gym.Env]:
    """Return an environment factory with the standard wrapper stack.

    Wrapper order:
      base env -> TimeLimit -> FlattenObservation ->
      RecordEpisodeStatistics -> ClipAction ->
      [NormalizeObservation -> clip obs -> NormalizeReward -> clip reward]

    When *normalize* is False the observation/reward normalisation wrappers
    are skipped (useful for overfitting tests with constant rewards).
    """

    def thunk() -> gym.Env:
        env_cls = ENV_REGISTRY.get(env_id)
        if env_cls is None:
            raise ValueError(
                f"Unknown env_id '{env_id}'. " f"Available: {list(ENV_REGISTRY.keys())}"
            )
        env: gym.Env = env_cls(render_mode="rgb_array")  # type: ignore[call-arg]
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = gym.wrappers.ClipAction(env)
        if normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10)
            )
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )
        return env

    return thunk
