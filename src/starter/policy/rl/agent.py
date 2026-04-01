"""Base RL agent interface for starter environments."""

import abc
from typing import Any, TypeVar

from omegaconf import DictConfig

from starter.policy.rl.gym_utils import make_env

_O = TypeVar("_O")
_U = TypeVar("_U")


class BaseRLAgent(abc.ABC):
    """Base class for RL agents."""

    def __init__(
        self,
        seed: int,
        env_id: str | None = None,
        max_episode_steps: int | None = None,
        cfg: DictConfig | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else DictConfig({})
        self.env_id = env_id if env_id is not None else ""
        self.max_episode_steps = (
            max_episode_steps if max_episode_steps is not None else 0
        )
        self._seed = seed

        # Create temporary env to extract spaces.
        temp_env = make_env(self.env_id, self.max_episode_steps)()
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        self.action_space.seed(seed)
        temp_env.close()  # type: ignore[no-untyped-call]

    @abc.abstractmethod
    def train(
        self, eval_episodes: int = 10, render_eval_video: bool = False
    ) -> dict[str, Any]:
        """Train the agent and return metrics."""

    def save(self, filepath: str) -> None:
        """Save agent parameters."""

    def load(self, filepath: str) -> None:
        """Load agent parameters."""

    def close(self) -> None:
        """Clean up resources."""
