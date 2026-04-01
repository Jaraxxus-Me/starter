"""PPO agent implementation for starter environments.

Based on CleanRL's PPO continuous action implementation:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
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
from torch.distributions.normal import Normal

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore
    TENSORBOARD_AVAILABLE = False

from starter.policy.rl.agent import BaseRLAgent
from starter.policy.rl.gym_utils import make_env


@dataclass
class PPOArgs:
    """Hyperparameters for PPO."""

    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = False
    save_model: bool = True

    # Environment.
    num_envs: int = 1
    num_steps: int = 2048
    save_model_freq: int = 25

    # Network.
    hidden_size: int = 64

    # Optimization.
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Parallelization.
    async_envs: bool = False

    # Computed at runtime.
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def _layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)  # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const)  # type: ignore
    return layer


class PPONetwork(nn.Module):
    """PPO actor-critic MLP."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        obs_dim = int(np.array(observation_space.shape).prod())
        act_dim = int(np.prod(action_space.shape))

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return state value estimate."""
        return self.critic(x)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample an action from the policy."""
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))  # type: ignore
        return probs.sample()  # type: ignore

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, entropy, value)."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)  # type: ignore
        if action is None:
            action = probs.sample()  # type: ignore[no-untyped-call]
        return (
            action,
            probs.log_prob(action).sum(1),  # type: ignore[no-untyped-call]
            probs.entropy().sum(1),  # type: ignore[no-untyped-call]
            self.critic(x),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (deterministic action)."""
        return self.get_action(x, deterministic=True)


class PPOAgent(BaseRLAgent):
    """PPO agent for continuous control tasks."""

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
        self.args = dacite.from_dict(PPOArgs, args_dict)

        # TensorBoard.
        if cfg.get("tf_log", True):
            exp_name = cfg.get("exp_name", "ppo_experiment")
            tb_log_dir = cfg.get("tb_log_dir", "runs")
            self.log_path = Path(tb_log_dir) / exp_name
            self.writer = SummaryWriter(self.log_path)  # type: ignore
        else:
            self.log_path = Path("runs/ppo_experiment")
            self.writer = None  # type: ignore

        assert isinstance(self.observation_space, spaces.Box)
        assert isinstance(self.action_space, spaces.Box)

        self.network = PPONetwork(
            self.observation_space,
            self.action_space,
            hidden_size=self.args.hidden_size,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.args.learning_rate, eps=1e-5
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
        self.network.eval()
        obs, _ = envs.reset()
        episodic_returns: list[float] = []
        step_lengths: list[int] = []
        step_counts = [0] * envs.num_envs

        while len(episodic_returns) < eval_episodes:
            with torch.no_grad():
                actions = self.network(torch.Tensor(obs).to(self.device)).cpu().numpy()
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
            make_env(self.env_id, self.max_episode_steps, gamma=args.gamma)
            for _ in range(args.num_envs)
        ]
        envs: gym.vector.VectorEnv
        if args.async_envs:
            envs = gym.vector.AsyncVectorEnv(env_fns)
        else:
            envs = gym.vector.SyncVectorEnv(env_fns)

        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size

        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape
        assert obs_shape is not None and action_shape is not None

        obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(self.device)
        actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(
            self.device
        )
        logprobs = torch.zeros(args.num_steps, args.num_envs).to(self.device)
        rewards = torch.zeros(args.num_steps, args.num_envs).to(self.device)
        dones = torch.zeros(args.num_steps, args.num_envs).to(self.device)
        values = torch.zeros(args.num_steps, args.num_envs).to(self.device)

        next_obs_np, _ = envs.reset(seed=args.seed)
        next_obs_t = torch.Tensor(next_obs_np).to(self.device)
        next_done = torch.zeros(args.num_envs).to(self.device)
        global_step = 0
        start_time = time.time()

        for iteration in range(1, args.num_iterations + 1):
            print(
                f"Iteration: {iteration}/{args.num_iterations}, "
                f"global_step={global_step}"
            )

            # Checkpoint.
            if args.save_model and iteration % args.save_model_freq == 1:
                ckpt_dir = self.log_path / "policies"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.save(str(ckpt_dir / f"ckpt_{global_step}.pt"))

            # LR annealing.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            # ---- Rollout collection ----
            rollout_t0 = time.time()
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs_t
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.network.get_action_and_value(
                        next_obs_t
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                (
                    next_obs_np,
                    reward,
                    terminations,
                    truncations,
                    infos,
                ) = envs.step(action.cpu().numpy())
                next_done_np = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs_t = torch.Tensor(next_obs_np).to(self.device)
                next_done = torch.Tensor(next_done_np).to(self.device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            raw_r = info["episode"]["r"]
                            ep_r = (
                                raw_r.item() if hasattr(raw_r, "item") else float(raw_r)
                            )
                            episodic_returns.append(ep_r)
                            self._log_scalar(
                                "charts/episodic_return", ep_r, global_step
                            )
            rollout_time = time.time() - rollout_t0

            # ---- GAE ----
            with torch.no_grad():
                next_value = self.network.get_value(next_obs_t).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam: torch.Tensor | float = 0.0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # ---- Flatten batch ----
            b_obs = obs.reshape((-1,) + obs_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + action_shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # ---- PPO update ----
            self.network.train()
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            update_t0 = time.time()

            for _ in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = (
                        self.network.get_action_and_value(
                            b_obs[mb_inds], b_actions[mb_inds]
                        )
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_adv = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    # Policy loss.
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio,
                        1 - args.clip_coef,
                        1 + args.clip_coef,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss.
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = (
                            0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        )
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()  # type: ignore
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), args.max_grad_norm
                    )
                    self.optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            update_time = time.time() - update_t0

            # ---- Logging ----
            elapsed = time.time() - start_time
            self._log_scalar("charts/SPS", int(global_step / elapsed), global_step)
            self._log_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self._log_scalar("losses/value_loss", v_loss.item(), global_step)
            self._log_scalar("losses/entropy", entropy_loss.item(), global_step)
            self._log_scalar("time/rollout_time", rollout_time, global_step)
            self._log_scalar("time/update_time", update_time, global_step)

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
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
