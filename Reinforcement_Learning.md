# Reinforcement Learning Resources

## 📚 Essential Learning Paths & Webpages

- **[OpenAI Spinning Up](https://spinningup.openai.com/):** The best starting point by OpenAI with practical courses and explanations.
- **[Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/):** A very modern, hands-on course that covers everything from basics to RLHF (the tech behind ChatGPT).
- **[DeepMind's RL Lecture Series](https://www.youtube.com/watch?v=TCCjZe0y4Qc&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm):** Taught by David Silver (the mind behind AlphaGo). This is the "University Level" deep dive into RL.

______________________________________________________________________

## 📖 Textbooks

 Understand *Why* the math works, these are the bibles.

- **[Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (Sutton & Barto):** The absolute "Bible" of the field. It focuses on the fundamental concepts (Markov Decision Processes, Value Functions) that exist even without Neural Networks.
- **[Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) (Csaba Szepesvári):** A shorter, more mathematically dense book with rigorous formalization.

______________________________________________________________________

## 📄 Seminal Papers

To be a senior in this field, you should eventually read these "founding documents."

1. **[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013):** The birth of Deep Q-Learning (DQN). This proved RL could work with raw pixels.
1. **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017):** The paper behind PPO, the most used algorithm in the world today.
1. **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT):** The blueprint for how RL is used to make LLMs safe and helpful.

______________________________________________________________________

## 🏗️ Core Frameworks & Libraries

- **[Gymnasium](https://gymnasium.farama.org/):** The industry-standard API for RL environments. It’s the "socket" that connects your logic to RL algorithms.
- **[Stable Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/):** PyTorch implementations of RL algorithms (PPO, DQN, SAC). Great for baseline testing.
- **[CleanRL](https://docs.cleanrl.dev/):** High-quality, single-file implementations. If you want to see exactly how PPO is coded without jumping through 10 libraries.
- **[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html):** The "Senior Architect" choice. Designed for distributed RL at scale (e.g., training across 100 GPUs).

______________________________________________________________________

## 🛠️ Debugging & Visualization Tools

RL is notoriously hard to debug. You need "eyes" on your training.

- **[Weights & Biases (W&B)](https://wandb.ai/):** The industry standard for tracking experiments. It will plot your rewards, losses, and even video of your agent in real-time.
- **[TensorBoard](https://www.tensorflow.org/tensorboard):** The classic, local way to visualize your PyTorch gradients and reward curves.
