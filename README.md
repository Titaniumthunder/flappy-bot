# Flappy Bird RL Agent

A reinforcement learning agent trained to play a custom Flappy Bird environment using PPO (Proximal Policy Optimization).

## Results
- Average reward: ~200 / episode
- Peak reward: 1300+
- Trained for 4M+ timesteps

## How it works
The agent uses a custom CNN to process raw 84x84 grayscale pixel frames and outputs flap or do-nothing actions. It was trained using PPO with frame stacking (4 frames) so the model can infer motion and velocity. The environment randomizes pipe gap size, scroll speed, and pipe spacing every episode so the agent must generalize rather than memorize.

## Setup
pip install stable-baselines3 pygame torch gymnasium pillow

## Train from scratch
python train.py

Training automatically loads the previous model and continues improving each run.

## Watch it play
python watch.py

Press Q to quit, R to restart the episode.

## Architecture
- Policy: Custom 3-layer CNN (32→64→128 filters) + 2-layer MLP (256→256)
- Algorithm: PPO with frame stacking (n_stack=4)
- Parallel environments: 256
- Hardware: Apple M5 Pro (MPS acceleration)

## Environment
- Observation: 84x84 grayscale pixel frame
- Actions: 0 = do nothing, 1 = flap
- Reward: +5.0 per pipe passed · +0.1 per frame survived · -2.0 on death
- Procedural generation: pipe gap, speed, and spacing randomized each episode