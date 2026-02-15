import argparse
import time
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_env import SnakeEnv

# ── Configuration defaults ─────
DEFAULTS = {
    "grid_size": 10,
    "total_timesteps": 500_000,
    "render_every": 20_000,
    "render_episodes": 3,
    "checkpoint_dir": "checkpoints",
    "load_checkpoint": None,

    # PPO hyperparameters
    "learning_rate": 5e-4,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,

    # Reward shaping
    "reward_food": 50.0,
    "reward_crash": -500.0,
    "reward_step": -0.5,
}


def parse_args() -> argparse.Namespace:
    """Replace default values.

    Since I am experimenting with RL and parameter tuning
    it's useful to be able to easily change config without editing code every time.

    :return: Parsed arguments with defaults from DEFAULTS dict.
    """
    p = argparse.ArgumentParser(description="Train Snake RL agent")
    for key, default in DEFAULTS.items():
        flag = f"--{key.replace('_', '-')}"
        if default is None:
            p.add_argument(flag, default=default, type=str)
        else:
            p.add_argument(flag, default=default, type=type(default))
    return p.parse_args()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_env(grid_size: int, rewards: dict) -> SnakeEnv:
    """Helper to create a new SnakeEnv instance with given config."""
    return SnakeEnv(grid_size=grid_size, rewards=rewards)


def render_episodes(model: RecurrentPPO, env_kwargs: dict, n_episodes: int) -> None:
    """Run a few episodes with Pygame rendering so we can watch the agent."""
    vis_env = SnakeEnv(**env_kwargs, render_mode="human")
    for ep in range(n_episodes):
        obs, info = vis_env.reset()
        lstm_states = None
        episode_start = True
        done = False
        total_reward = 0.0

        while not done:
            vis_env.render()
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = False
            obs, reward, terminated, truncated, info = vis_env.step(int(action))
            total_reward += reward
            done = terminated or truncated

        vis_env.render()
        time.sleep(0.5)
        print(
            f"  Episode {ep + 1}: score={info['score']}, total_reward={total_reward:.1f}"
        )

    vis_env.close()


def train(args: argparse.Namespace) -> None:
    device = detect_device()
    print(f"Using device: {device}")

    rewards = {
        "food": args.reward_food,
        "crash": args.reward_crash,
        "step": args.reward_step,
    }
    env_kwargs = {"grid_size": args.grid_size, "rewards": rewards}

    env = DummyVecEnv([lambda: make_env(args.grid_size, rewards)])

    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        model = RecurrentPPO.load(args.load_checkpoint, env=env, device=device)
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            device=device,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    trained_so_far = 0
    while trained_so_far < args.total_timesteps:
        chunk = min(args.render_every, args.total_timesteps - trained_so_far)
        print(f"\n{'=' * 60}")
        print(f"Training steps {trained_so_far} → {trained_so_far + chunk}")
        print(f"{'=' * 60}")

        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        trained_so_far += chunk

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"snake_ppo_{trained_so_far}"
        model.save(str(ckpt_path))
        print(f"Checkpoint saved: {ckpt_path}")

        # Render a few episodes to visualize progress
        print(f"\nRendering {args.render_episodes} episodes...")
        render_episodes(model, env_kwargs, args.render_episodes)

    # Save final model
    final_path = checkpoint_dir / "snake_ppo_final"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved: {final_path}")
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
