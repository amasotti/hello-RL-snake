# Hello RL - Snake Game Training

A learning project for [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning): train a recurrent
PPO agent to play Snake using [Gymnasium](https://gymnasium.farama.org/index.html).

The agent starts completely random and gradually learns to seek food and avoid walls. Training pauses periodically to
render episodes with Pygame so you can watch it improve in real time.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python train.py
```

### Arguments

| Flag | Default | Description |
|---------------------|--------------|------------------------------------|
| `--grid-size` | 10 | Size of the NxN game grid |
| `--total-timesteps` | 500000 | Total training steps |
| `--render-every` | 20000 | Steps between visualization pauses |
| `--render-episodes` | 3 | Episodes to render at each pause |
| `--checkpoint-dir` | checkpoints/ | Where to save models |
| `--load-checkpoint` | None | Resume from a saved model |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--n-steps` | 128 | Steps per rollout |
| `--batch-size` | 128 | PPO minibatch size |
| `--n-epochs` | 10 | PPO update epochs |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-range` | 0.2 | PPO clip range |
| `--ent-coef` | 0.01 | Entropy coefficient |
| `--reward-food` | 10.0 | Reward for eating food |
| `--reward-crash` | -100.0 | Penalty for hitting wall/self |
| `--reward-step` | -0.1 | Per-step penalty |

## What the agent sees

The observation is a 15-dimensional feature vector:

| Features | Dims | Range | Description |
|-------------------|------|---------|-------------------------------------------------------------------------------------------------|
| Head position | 2 | [0, 1] | Row and column of the snake's head, normalized by grid size |
| Food position | 2 | [0, 1] | Row and column of the food |
| Direction to food | 2 | [-1, 1] | Relative vector from head to food |
| Danger flags | 4 | {0, 1} | Whether the immediate neighbor in each direction (up/down/left/right) is a wall or body segment |
| Current direction | 4 | one-hot | Which way the snake is currently facing |
| Snake length | 1 | [0, 1] | Body length normalized by total grid area |

The agent chooses from 4 discrete actions: up, down, left, right. Attempting a 180-degree turn (e.g. pressing left while
moving right) is ignored as the snake cannot walk on itself.
