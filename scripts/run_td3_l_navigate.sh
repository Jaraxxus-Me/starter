#!/bin/bash
# Train TD3 on L-navigate across multiple seeds.
# Usage: ./scripts/run_td3_l_navigate.sh

set -e

for seed in 0 1 2 3 4; do
    echo "===== Running TD3 on LNavigate-v0, seed=${seed} ====="
    python experiments/run_experiment.py \
        agent=td3_l_navigate \
        env_id=LNavigate-v0 \
        max_episode_steps=300 \
        eval_episodes=50 \
        seed=${seed} \
        agent.args.total_timesteps=1000000 \
        agent.args.num_envs=1 \
        agent.args.hidden_size=256 \
        agent.args.learning_starts=25000 \
        agent.args.batch_size=256 \
        agent.args.buffer_size=1000000
done
