# Experiments Guide

This document lists every tunable knob in the Snake Hello-RL project.

## Reward Shaping

| Flag | Default | What it does |
|------------------|---------|----------------------------------------------------------------------------------------------------------------------|
| `--reward-food` | 10.0 | Reward for eating food. Higher values make the agent prioritize food-seeking more aggressively. |
| `--reward-crash` | -100.0 | Penalty for hitting a wall or itself. Larger magnitude makes the agent more cautious. |
| `--reward-step` | -0.1 | Per-step penalty. Encourages the agent to find food quickly rather than wandering. Set to 0 to remove time pressure. |

**Experiments to try:**

- Set `--reward-step 0` — the agent may survive longer but wander aimlessly
- Set `--reward-crash -10` — the agent becomes less afraid of walls, more exploratory
- Set `--reward-food 1 --reward-crash -1 --reward-step 0` — sparse, balanced signal; slower learning but potentially more robust

## Grid Size

| Flag | Default | What it does |
|---------------|---------|----------------------------------------------------------|
| `--grid-size` | 10 | Size of the NxN grid. Smaller grids are easier to learn. |

**Experiments to try:**

- `--grid-size 5` — very fast training, agent should learn well within 50k steps
- `--grid-size 20` — much harder; requires more training time and possibly a larger network

## PPO Hyperparameters

| Flag | Default | What it does |
|-------------------|---------|-----------------------------------------------------------------------------------------------------------|
| `--learning-rate` | 3e-4 | Step size for policy updates. Too high = unstable, too low = slow. |
| `--n-steps` | 128 | Steps collected per environment before each update. Larger = more stable gradients. |
| `--batch-size` | 128 | Minibatch size for PPO updates. Must be ≤ n_steps. |
| `--n-epochs` | 10 | Number of passes over collected data per update. More = better sample efficiency but risk of overfitting. |
| `--gamma` | 0.99 | Discount factor. Lower values (0.9) make the agent more short-sighted. |
| `--gae-lambda` | 0.95 | GAE smoothing. Lower values reduce variance but increase bias. |
| `--clip-range` | 0.2 | PPO clipping parameter. Keeps policy updates conservative. |
| `--ent-coef` | 0.01 | Entropy bonus coefficient. Higher values encourage more exploration. |

**Experiments to try:**

- `--ent-coef 0.1` — much more exploration, useful early in training
- `--learning-rate 1e-3` — faster but potentially unstable
- `--gamma 0.9` — agent focuses on immediate rewards (food nearby)
- `--n-steps 256 --batch-size 256` — more stable updates, slower iteration

## Training Loop

| Flag | Default | What it does |
|---------------------|--------------|---------------------------------------------------------------------|
| `--total-timesteps` | 500000 | Total environment steps to train for. |
| `--render-every` | 5000 | Steps between visualization pauses. Lower = more frequent watching. |
| `--render-episodes` | 3 | Episodes to render at each visualization pause. |
| `--checkpoint-dir` | checkpoints/ | Where to save model checkpoints. |
| `--load-checkpoint` | None | Path to resume training from a saved model. |

## Observation Design

The current observation is a 15-dimensional feature vector (defined in `snake_env.py`):

- Head position (2) — normalized to [0, 1]
- Food position (2) — normalized to [0, 1]
- Direction to food (2) — normalized to [-1, 1]
- Danger in 4 directions (4) — binary, 1 if wall/body adjacent
- Current direction one-hot (4) — which way the snake faces
- Snake length (1) — normalized by grid area

**Ideas to try (requires code changes in `snake_env.py`):**

- Add distance-to-food as a scalar feature
- Use a full NxN grid observation (channels for: empty, snake body, snake head, food)
- Add "look ahead" rays — cast rays in 8 directions, report distance to wall/body/food
- Remove food position and only keep direction — forces the agent to learn spatial reasoning

## Quick Start Recipes

```bash
# Fast iteration on a tiny grid
uv run python train.py --grid-size 5 --total-timesteps 50000 --render-every 2000

# Long training run with less frequent visualization
uv run python train.py --total-timesteps 2000000 --render-every 50000

# Resume from checkpoint
uv run python train.py --load-checkpoint checkpoints/snake_ppo_100000

# Aggressive exploration
uv run python train.py --ent-coef 0.1 --reward-step 0

# Conservative, cautious agent
uv run python train.py --reward-crash -500 --gamma 0.95
```
