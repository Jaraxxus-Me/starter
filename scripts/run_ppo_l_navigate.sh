#!/bin/bash
# Train PPO on L-navigate across multiple seeds.
# Usage: ./scripts/run_ppo_l_navigate.sh

set -e

for seed in 0 1 2 3 4; do
    echo "===== Running PPO on LNavigate-v0, seed=${seed} ====="
    python experiments/run_experiment.py \
        agent=ppo_l_navigate \
        env_id=LNavigate-v0 \
        max_episode_steps=300 \
        eval_episodes=50 \
        seed=${seed} \
        agent.args.total_timesteps=1000000 \
        agent.args.num_envs=8 \
        agent.args.num_steps=256 \
        agent.args.hidden_size=128
done
