"""TD3 agent implementation for starter environments.

Based on CleanRL's TD3 continuous action implementation:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dacite
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig
from torch import nn, optim
from torch.nn import functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

from starter.policy.rl.agent import BaseRLAgent
from starter.policy.rl.gym_utils import make_env


@dataclass
class TD3Args:
    """Hyperparameters for TD3."""

    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = False
    save_model: bool = True

    # Environment.
    num_envs: int = 1
    save_model_freq: int = 25

    # Network.
    hidden_size: int = 256

    # Optimization.
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 25_000
    policy_frequency: int = 2

    # Noise.
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    # Regularization.
    action_reg: float = 0.0

    # Parallelization.
    async_envs: bool = False

    # Wrappers.
    normalize: bool = True


class ReplayBuffer:
    """Simple replay buffer for off-policy RL."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        device: torch.device,
        n_envs: int = 1,
    ) -> None:
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.n_envs = n_envs
        self.device = device
        self.pos = 0
        self.full = False

        obs_shape = observation_space.shape
        act_shape = action_space.shape
        assert obs_shape is not None and act_shape is not None

        self.observations = np.zeros(
            (self.buffer_size, n_envs) + obs_shape, dtype=np.float32
        )
        self.next_observations = np.zeros(
            (self.buffer_size, n_envs) + obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, n_envs) + act_shape, dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Add a batch of transitions."""
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions."""
        upper = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper, size=batch_size)
        env_inds = np.random.randint(0, self.n_envs, size=batch_size)
        return (
            torch.tensor(self.observations[batch_inds, env_inds]).to(self.device),
            torch.tensor(self.next_observations[batch_inds, env_inds]).to(self.device),
            torch.tensor(self.actions[batch_inds, env_inds]).to(self.device),
            torch.tensor(self.rewards[batch_inds, env_inds]).to(self.device),
            torch.tensor(self.dones[batch_inds, env_inds]).to(self.device),
        )


class TD3QNetwork(nn.Module):
    """TD3 Q-network (critic)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        obs_dim = int(np.array(observation_space.shape).prod())
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return Q-value for (state, action) pair."""
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TD3Actor(nn.Module):
    """TD3 deterministic actor."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        obs_dim = int(np.array(observation_space.shape).prod())
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, act_dim)

        # Small init for output layer to prevent tanh saturation.
        nn.init.uniform_(self.fc_mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_mu.bias, -3e-3, 3e-3)

        # Action rescaling.
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic action."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class TD3Agent(BaseRLAgent):
    """TD3 agent for continuous control tasks."""

    def _log_scalar(
        self, tag: str, value: float, step: int
    ) -> None:  # pylint: disable=missing-function-docstring
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)  # type: ignore[no-untyped-call]

    def __init__(
        self,
        seed: int,
        env_id: str | None = None,
        max_episode_steps: int | None = None,
        cfg: DictConfig | None = None,
    ) -> None:
        super().__init__(seed, env_id, max_episode_steps, cfg)

        if cfg is None:
            cfg = DictConfig({})

        cuda_enabled = cfg.get("cuda", False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cuda_enabled else "cpu"
        )

        args_dict = cfg.get("args", cfg) if "args" in cfg else dict(cfg)
        self.args = dacite.from_dict(TD3Args, args_dict)

        # TensorBoard.
        if cfg.get("tf_log", True):
            exp_name = cfg.get("exp_name", "td3_experiment")
            tb_log_dir = cfg.get("tb_log_dir", "runs")
            self.log_path = Path(tb_log_dir) / exp_name
            self.writer = SummaryWriter(self.log_path)  # type: ignore
        else:
            self.log_path = Path("runs/td3_experiment")
            self.writer = None  # type: ignore

        assert isinstance(self.observation_space, spaces.Box)
        assert isinstance(self.action_space, spaces.Box)

        # Actor.
        self.actor = TD3Actor(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.target_actor = TD3Actor(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.args.learning_rate
        )

        # Critics.
        self.qf1 = TD3QNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf2 = TD3QNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf1_target = TD3QNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf2_target = TD3QNetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=self.args.learning_rate,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_on_env(
        self,
        envs: gym.vector.VectorEnv,
        eval_episodes: int,
    ) -> dict[str, Any]:
        """Evaluate the agent and return episode metrics."""
        self.actor.eval()
        obs, _ = envs.reset()
        episodic_returns: list[float] = []
        step_lengths: list[int] = []
        step_counts = [0] * envs.num_envs

        while len(episodic_returns) < eval_episodes:
            with torch.no_grad():
                actions = self.actor(torch.Tensor(obs).to(self.device)).cpu().numpy()
            obs, _, _, _, infos = envs.step(actions)
            step_counts = [s + 1 for s in step_counts]

            if "final_info" in infos:
                for env_idx, info in enumerate(infos["final_info"]):
                    if info is None or "episode" not in info:
                        continue
                    raw_return = info["episode"]["r"]
                    ep_return = (
                        raw_return.item()
                        if hasattr(raw_return, "item")
                        else float(raw_return)
                    )
                    episodic_returns.append(ep_return)
                    step_lengths.append(step_counts[env_idx])
                    step_counts[env_idx] = 0

        return {
            "episodic_return": episodic_returns,
            "step_length": step_lengths,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(  # type: ignore[override]
        self, eval_episodes: int = 10, render_eval_video: bool = False
    ) -> dict[str, Any]:
        args = self.args
        episodic_returns: list[float] = []

        env_fns = [
            make_env(
                self.env_id,
                self.max_episode_steps,
                gamma=args.gamma,
                normalize=args.normalize,
            )
            for _ in range(args.num_envs)
        ]
        envs: gym.vector.VectorEnv
        if args.async_envs:
            envs = gym.vector.AsyncVectorEnv(env_fns)
        else:
            envs = gym.vector.SyncVectorEnv(env_fns)

        assert isinstance(envs.single_observation_space, spaces.Box)
        assert isinstance(envs.single_action_space, spaces.Box)

        # Replay buffer.
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            n_envs=args.num_envs,
        )

        obs, _ = envs.reset(seed=args.seed)
        start_time = time.time()
        actor_loss_val = 0.0

        for global_step in range(args.total_timesteps):
            # Action selection.
            if global_step < args.learning_starts:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(args.num_envs)]
                )
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(
                        0,
                        self.actor.action_scale * args.exploration_noise,
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(
                            envs.single_action_space.low,
                            envs.single_action_space.high,
                        )
                    )

            # Environment step.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # Log episodic returns.
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        raw_r = info["episode"]["r"]
                        ep_r = raw_r.item() if hasattr(raw_r, "item") else float(raw_r)
                        episodic_returns.append(ep_r)
                        self._log_scalar("charts/episodic_return", ep_r, global_step)
                        raw_l = info["episode"]["l"]
                        ep_l = raw_l.item() if hasattr(raw_l, "item") else float(raw_l)
                        self._log_scalar(
                            "charts/episodic_length",
                            ep_l,
                            global_step,
                        )

            # Handle truncation for replay buffer.
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            rb.add(
                obs,
                real_next_obs,
                actions,
                rewards,
                terminations,
            )
            obs = next_obs

            # Training.
            if global_step > args.learning_starts:
                (
                    s_obs,
                    s_next_obs,
                    s_actions,
                    s_rewards,
                    s_dones,
                ) = rb.sample(args.batch_size)

                with torch.no_grad():
                    clipped_noise = (
                        torch.randn_like(s_actions) * args.policy_noise
                    ).clamp(-args.noise_clip, args.noise_clip)
                    clipped_noise = clipped_noise * self.target_actor.action_scale

                    next_state_actions = (
                        self.target_actor(s_next_obs) + clipped_noise
                    ).clamp(
                        envs.single_action_space.low[0],
                        envs.single_action_space.high[0],
                    )
                    qf1_next = self.qf1_target(s_next_obs, next_state_actions)
                    qf2_next = self.qf2_target(s_next_obs, next_state_actions)
                    min_qf_next = torch.min(qf1_next, qf2_next)
                    next_q_value = s_rewards + (
                        1 - s_dones
                    ) * args.gamma * min_qf_next.view(-1)

                qf1_a = self.qf1(s_obs, s_actions).view(-1)
                qf2_a = self.qf2(s_obs, s_actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a, next_q_value)
                qf2_loss = F.mse_loss(qf2_a, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                self.q_optimizer.zero_grad()
                qf_loss.backward()  # type: ignore
                self.q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    actor_actions = self.actor(s_obs)
                    actor_loss = -self.qf1(s_obs, actor_actions).mean()
                    if args.action_reg > 0:
                        actor_loss = (
                            actor_loss + args.action_reg * (actor_actions**2).mean()
                        )
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    actor_loss_val = actor_loss.item()

                    # Soft update target networks.
                    for param, target_param in zip(
                        self.actor.parameters(),
                        self.target_actor.parameters(),
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf1.parameters(),
                        self.qf1_target.parameters(),
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(),
                        self.qf2_target.parameters(),
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    self._log_scalar(
                        "losses/qf1_values", qf1_a.mean().item(), global_step
                    )
                    self._log_scalar(
                        "losses/qf2_values", qf2_a.mean().item(), global_step
                    )
                    self._log_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    self._log_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    self._log_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    self._log_scalar("losses/actor_loss", actor_loss_val, global_step)
                    elapsed = time.time() - start_time
                    sps = int(global_step / elapsed) if elapsed > 0 else 0
                    self._log_scalar("charts/SPS", sps, global_step)

            if global_step % 10000 == 0:
                print(f"global_step={global_step}/{args.total_timesteps}")

        # Final checkpoint.
        if args.save_model:
            self.save(str(self.log_path / "final_ckpt.pt"))

        # Evaluate.
        logging.info("Starting evaluation for %d episodes...", eval_episodes)
        eval_metrics = self.evaluate_on_env(envs, eval_episodes)

        if eval_metrics["episodic_return"]:
            avg = np.mean(eval_metrics["episodic_return"])
            logging.info("Evaluation average return: %.2f", avg)

        envs.close()  # type: ignore[no-untyped-call]
        if self.writer is not None:
            self.writer.close()  # type: ignore[no-untyped-call]

        return {
            "train": {"episodic_return": episodic_returns},
            "eval": eval_metrics,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "qf1_state_dict": self.qf1.state_dict(),
                "qf2_state_dict": self.qf2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.qf1.load_state_dict(checkpoint["qf1_state_dict"])
        self.qf2.load_state_dict(checkpoint["qf2_state_dict"])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
