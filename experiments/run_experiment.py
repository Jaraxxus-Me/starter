"""Main entry point for running RL experiments.

Examples:
    python experiments/run_experiment.py agent=ppo_l_navigate \
        env_id=LNavigate-v0 seed=0
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, read_write

from starter.policy.rl import create_rl_agent


def _get_output_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    """Compute output and runs directories based on config."""
    seed_str = f"seed{cfg.seed}"
    output_dir = Path("outputs") / cfg.env_id / seed_str / cfg.agent.name
    runs_dir = Path("runs") / cfg.env_id / seed_str
    return output_dir, runs_dir


def _print_results_summary(metrics: dict, cfg: DictConfig) -> None:
    """Print a summary of training and evaluation results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"Agent: {cfg.agent.name}")
    print(f"Environment: {cfg.env_id}")
    print(f"Seed: {cfg.seed}")
    print("-" * 60)

    if "train" in metrics and metrics["train"].get("episodic_return"):
        tr = metrics["train"]["episodic_return"]
        print("TRAINING:")
        print(f"  Total episodes: {len(tr)}")
        print(f"  Mean return: {np.mean(tr):.2f}")
        print(f"  Std return: {np.std(tr):.2f}")
        if len(tr) >= 10:
            print(f"  Last 10 mean: {np.mean(tr[-10:]):.2f}")
        print("-" * 60)

    if "eval" in metrics and metrics["eval"].get("episodic_return"):
        ev = metrics["eval"]["episodic_return"]
        step_lens = metrics["eval"]["step_length"]
        print("EVALUATION:")
        print(f"  Episodes: {len(ev)}")
        print(f"  Mean return: {np.mean(ev):.2f}")
        print(f"  Std return: {np.std(ev):.2f}")
        successes = sum(1 for s in step_lens if s < cfg.max_episode_steps)
        pct = 100 * successes / len(ev)
        print(
            f"  Success rate (step < {cfg.max_episode_steps}): "
            f"{successes}/{len(ev)} ({pct:.1f}%)"
        )
    print("=" * 60 + "\n")


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    output_dir, runs_dir = _get_output_dirs(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Running agent=%s, env=%s, seed=%s",
        cfg.agent.name,
        cfg.env_id,
        cfg.seed,
    )

    with read_write(cfg):
        cfg.agent.tb_log_dir = str(runs_dir)
        cfg.agent.exp_name = f"{cfg.agent.name}_{cfg.env_id}"

    agent = create_rl_agent(cfg.agent, cfg.env_id, cfg.max_episode_steps, cfg.seed)

    metrics = agent.train(eval_episodes=cfg.eval_episodes)

    # Save outputs.
    agent.save(str(output_dir / "agent.pkl"))
    if "train" in metrics:
        pd.DataFrame(metrics["train"]).to_csv(
            output_dir / "train_results.csv", index=False
        )
    if "eval" in metrics:
        pd.DataFrame(metrics["eval"]).to_csv(
            output_dir / "eval_results.csv", index=False
        )
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    _print_results_summary(metrics, cfg)


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
