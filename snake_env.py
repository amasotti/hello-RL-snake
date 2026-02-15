"""Snake game as a Gymnasium environment for RL training."""

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

# Directions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
DIRECTION_DELTAS = {
    0: np.array([-1, 0]),  # UP
    1: np.array([1, 0]),  # DOWN
    2: np.array([0, -1]),  # LEFT
    3: np.array([0, 1]),  # RIGHT
}

# Opposite directions (to prevent 180-degree turns)
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

DEFAULT_REWARDS = {"food": 10.0, "crash": -100.0, "step": -0.1}

# Pygame colors
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (200, 0, 0)
GRAY = (40, 40, 40)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)


class SnakeEnv(gym.Env):
    """A Snake game compatible with Gymnasium."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 10,
        rewards: dict[str, float] | None = None,
        render_mode: str | None = None,
        max_steps_without_food: int | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.render_mode = render_mode
        self.max_steps_without_food = max_steps_without_food or grid_size * grid_size

        self.action_space = spaces.Discrete(4)

        # 15-dimensional feature vector
        # This is what the agent will see at each step and is
        # the basis for the policy update
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )

        # Pygame state
        self._cell_size = 20
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        # Game state (initialized in reset)
        self.snake: deque[tuple[int, int]] = deque()
        self.direction: int = 3  # START facing RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.steps_without_food: int = 0
        self.score: int = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        mid = self.grid_size // 2
        self.snake = deque([(mid, mid - 1), (mid, mid), (mid, mid + 1)])
        self.direction = 3  # facing RIGHT, head is last element
        self.steps_without_food = 0
        self.score = 0
        self._place_food()

        return self._get_obs(), {"score": self.score}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action and return (obs, reward, terminated, truncated, info).
        :param action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        :return: obs: 15-dim feature vector building the new observation after applying the action
        """

        # Prevent 180-degree turns (snake can't walk on his own body)
        if action != OPPOSITE.get(self.direction, -1):
            self.direction = action

        head_r, head_c = self.snake[-1]
        delta = DIRECTION_DELTAS[self.direction]
        new_head = (head_r + delta[0], head_c + delta[1])

        # Check collision with walls (row and column of the head position)
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return (
                self._get_obs(),
                self.rewards["crash"],
                True,
                False,
                {"score": self.score},
            )

        # Check collision with self (exclude tail — it will move unless we eat food)
        if new_head in list(self.snake)[1:]:  # skip tail since it might move
            return (
                self._get_obs(),
                self.rewards["crash"],
                True,
                False,
                {"score": self.score},
            )

        self.snake.append(new_head)

        # Check food
        if new_head == self.food:
            reward = self.rewards["food"]
            self.score += 1
            self.steps_without_food = 0
            self._place_food()
        else:
            self.snake.popleft()  # move forward
            reward = self.rewards["step"]
            self.steps_without_food += 1

        # Truncate if stuck in a loop
        truncated = self.steps_without_food >= self.max_steps_without_food

        return self._get_obs(), reward, False, truncated, {"score": self.score}

    def _place_food(self) -> None:
        snake_set = set(self.snake)
        free_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in snake_set
        ]
        if free_cells:
            idx = self.np_random.integers(len(free_cells))
            self.food = free_cells[idx]

    def _get_obs(self) -> np.ndarray:
        gs = self.grid_size
        head_r, head_c = self.snake[-1]  # last element of the snake deque is the head
        food_r, food_c = self.food

        # Normalized head/food positions
        # WHY: Normalizing by grid size will stabilize learning across different grid sizes
        # This allows the model to generalize better if we change the grid size later.
        head_norm = np.array([head_r / (gs - 1), head_c / (gs - 1)])
        food_norm = np.array([food_r / (gs - 1), food_c / (gs - 1)])

        # Direction to food (normalized)
        delta_food = np.array(
            [(food_r - head_r) / (gs - 1), (food_c - head_c) / (gs - 1)]
        )

        # Danger detection: 1 if immediate neighbor in that direction is wall or body
        snake_set = set(self.snake)
        dangers = np.zeros(4, dtype=np.float32)
        for d, dv in DIRECTION_DELTAS.items():
            nr, nc = head_r + dv[0], head_c + dv[1]
            if nr < 0 or nr >= gs or nc < 0 or nc >= gs or (nr, nc) in snake_set:
                dangers[d] = 1.0

        # Current direction one-hot
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.direction] = 1.0

        # Snake length normalized (not part of the reward itself which is given by the food
        # but is useful information for the model to learn from: should avoid risky moves)
        length_norm = np.array([len(self.snake) / (gs * gs)], dtype=np.float32)

        obs = np.concatenate(
            [head_norm, food_norm, delta_food, dangers, dir_onehot, length_norm]
        ).astype(np.float32)

        return obs

    def render(self) -> None:
        if self.render_mode != "human":
            return

        if self._window is None:
            pygame.init()
            size = self.grid_size * self._cell_size
            self._window = pygame.display.set_mode((size, size))
            pygame.display.set_caption("Snake RL")
            self._clock = pygame.time.Clock()

        self._window.fill(WHITE)

        # Draw grid
        size = self.grid_size * self._cell_size
        for i in range(self.grid_size + 1):
            pos = i * self._cell_size
            pygame.draw.line(self._window, BLACK, (pos, 0), (pos, size))
            pygame.draw.line(self._window, BLACK, (0, pos), (size, pos))

        # Draw snake body
        for r, c in self.snake:
            rect = pygame.Rect(
                c * self._cell_size + 1,
                r * self._cell_size + 1,
                self._cell_size - 2,
                self._cell_size - 2,
            )
            pygame.draw.rect(self._window, GREEN, rect)

        # Draw head differently
        head_r, head_c = self.snake[-1]
        head_rect = pygame.Rect(
            head_c * self._cell_size + 1,
            head_r * self._cell_size + 1,
            self._cell_size - 2,
            self._cell_size - 2,
        )
        pygame.draw.rect(self._window, YELLOW, head_rect)

        # Draw food
        food_r, food_c = self.food
        food_rect = pygame.Rect(
            food_c * self._cell_size + 1,
            food_r * self._cell_size + 1,
            self._cell_size - 2,
            self._cell_size - 2,
        )
        pygame.draw.rect(self._window, RED, food_rect)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        # Process Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self) -> None:
        if self._window is not None:
            pygame.quit()
            self._window = None
            self._clock = None
