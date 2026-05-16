"""
Flood-It: Gymnasium Environment
================================

Wraps the terminal Flood-It game into a Gymnasium-compatible environment.
Uses existing floodit.py game logic with minimal modifications for importability.
"""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Import game logic from floodit.py
from floodit import (
    BOARD_SIZE,
    NUM_COLORS,
    MAX_MOVES,
    generate_board,
    flood_fill_stack,
    check_win,
    get_connected_cells,
)


class FloodItEnv(gym.Env):
    """
    Gymnasium environment for the Flood-It game.

    Game rules:
    - 14x14 grid with colors 0-5
    - Each turn, choose a color to flood from top-left region
    - Win when entire board becomes one color
    - Max 25 moves allowed
    - Reward: Difference in controlled territory (connected cells) after each move.
      This encourages expanding your connected region.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        render_mode: str = None,
        board_size: int = BOARD_SIZE,
        num_colors: int = NUM_COLORS,
        max_moves: int = MAX_MOVES,
    ) -> None:
        """
        Initialize the Flood-It environment.

        Args:
            render_mode: How to render the environment. Options are:
                - None (no rendering)
                - "human" (render in terminal)
            board_size: Width and height of the square board.
            num_colors: Number of color/action ids.
            max_moves: Episode move limit.
        """
        super().__init__()

        self.render_mode = render_mode
        self.board_size = int(board_size)
        self.num_colors = int(num_colors)
        self.max_moves = int(max_moves)

        if self.board_size <= 0:
            raise ValueError("board_size must be positive")
        if self.num_colors < 2:
            raise ValueError("num_colors must be at least 2")
        if self.max_moves <= 0:
            raise ValueError("max_moves must be positive")

        # Observation space: board color ids with shape (board_size, board_size).
        observation_shape = (self.board_size, self.board_size)
        self.observation_space = spaces.Box(
            low=0,
            high=self.num_colors - 1,
            shape=observation_shape,
            dtype=np.int64,
        )

        # Action space: choose a color from 0 to num_colors-1.
        self.action_space = spaces.Discrete(self.num_colors)

        # Store game state attributes (not in observation space)
        self._current_color: int = 0
        self._moves_used: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        options = options or {}

        if "fixed_board" in options:
            board = [row[:] for row in options["fixed_board"]]
        else:
            if seed is not None:
                np.random.seed(seed)
                import random
                random.seed(seed)

            board = generate_board(self.board_size, self.num_colors)

        board_array = np.array(board, dtype=np.int64)
        if board_array.shape != self.observation_space.shape:
            raise ValueError(
                f"fixed_board shape must be {self.observation_space.shape}, "
                f"got {board_array.shape}"
            )
        if board_array.min() < 0 or board_array.max() >= self.num_colors:
            raise ValueError(
                f"fixed_board colors must be in range [0, {self.num_colors - 1}]"
            )

        self._board = board_array.tolist()
        self._current_color = self._board[0][0]
        self._moves_used = 0
        self._solved = False

        observation = np.array(self._board, dtype=np.int64)

        info = {
            "moves": self._moves_used,
            "moves_left": self.max_moves - self._moves_used,
            "solved": False,
            "action_mask": self._action_mask(),
            "board_size": self.board_size,
            "num_colors": self.num_colors,
            "max_moves": self.max_moves,
        }

        return observation.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Args:
            action: Color to flood (0 to NUM_COLORS-1)

        Returns:
            observation: Updated board state
            reward: Difference in controlled territory between after and before
            terminated: True if game ended by winning
            truncated: True if max moves reached
            info: Dictionary with metadata
        """
        # Validate action
        if not isinstance(action, int):
            action = int(action)

        if action < 0 or action >= self.num_colors:
            raise ValueError(
                f"Invalid action {action}. Must be in range [0, {self.num_colors - 1}]"
            )

        self._moves_used += 1

        # Count controlled territory BEFORE the move
        from floodit import get_connected_cells
        territory_before = len(get_connected_cells(self._board, self._current_color))

        # Do flood fill with chosen color
        _ = flood_fill_stack(self._board, 0, 0, action)

        # Count controlled territory AFTER the move
        # After flood fill, the new connected region has the NEW color at (0,0)
        new_color_at_top_left = self._board[0][0]
        territory_after = len(get_connected_cells(self._board, new_color_at_top_left))

        # Reward is the difference: cells newly controlled this turn
        reward = float(territory_after - territory_before) -1

        # Bonus for solving (given once when game ends)

        # Check termination conditions
        terminated = check_win(self._board)
        truncated = self._moves_used >= self.max_moves

        # Reward for solving game (give bonus once, not on each step)
        if terminated and not getattr(self, "_solved", False):
            reward = 100.0 + reward  # Win bonus combined with move penalty
            self._solved = True

        # Update current color after flood fill
        self._current_color = self._board[0][0]

        # Convert board to numpy array for observation
        observation = np.array(self._board, dtype=np.int64)

        info = {
            "moves": self._moves_used,
            "moves_left": self.max_moves - self._moves_used,
            "solved": terminated,
            "action_mask": self._action_mask(),
            "board_size": self.board_size,
            "num_colors": self.num_colors,
            "max_moves": self.max_moves,
        }

        return observation.copy(), reward, terminated, truncated, info

    def _action_mask(self) -> np.ndarray:
        """Return valid color choices; choosing the current color is a no-op."""
        mask = np.ones(self.num_colors, dtype=np.int8)
        if hasattr(self, "_board") and self._board:
            mask[int(self._board[0][0])] = 0
        return mask

    @property
    def board(self) -> np.ndarray:
        """Get the current board as a numpy array."""
        if not hasattr(self, "_board"):
            raise RuntimeError("Board not initialized. Call reset() first.")
        return np.array(self._board, dtype=np.int64).copy()

    @property
    def moves_used(self) -> int:
        """Get number of moves used so far."""
        return self._moves_used

    @property
    def solved(self) -> bool:
        """Check if the game has been solved."""
        return getattr(self, "_solved", False)

    def render(self) -> Optional[str]:
        """
        Render the environment (terminal output).

        Returns:
            String containing terminal representation or None for non-human mode.
        """
        if self.render_mode == "human":
            # Print game state info and flush to ensure display updates
            import sys
            print(f"Moves left: {self.max_moves - self._moves_used}", end="\n\n")
            sys.stdout.flush()

            # Print board with aligned columns
            for row in self._board:
                line = ' '.join(f"{cell:2d}" for cell in row)
                print(line)
            sys.stdout.flush()

    def _render_text(self) -> str:
        """Generate terminal-compatible text representation."""
        lines = []

        # Header
        lines.append(f"Moves used: {self._moves_used}/{self.max_moves}")
        lines.append("=" * 38)

        # Board with color labels
        lines.append("Color key:")
        for c in range(self.num_colors):
            lines.append(f"  {c}: color {c}")
        lines.append("=" * 38)

        # Board grid (transpose to print columns as text lines)
        board_array = np.array(self._board)
        transposed = board_array.T
        for row in transposed:
            line = ' '.join(f"{cell:2d}" for cell in row)
            lines.append(line)

        lines.append("=" * 38)
        return '\n'.join(lines)

    def close(self) -> None:
        """Close the environment and release resources."""
        super().close()
        self._board = []  # Clear board
        self._current_color = -1
        self._moves_used = 0
        self._solved = False


def main():
    """
    Test script that runs random actions until episode ends.

    Demonstrates:
    - Environment reset
    - Taking random steps
    - Win/loss handling
    """
    print("Flood-It Environment Test")
    print("=" * 40)
    print()

    # Create environment with human rendering mode
    env = FloodItEnv(render_mode="human")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Reset environment and get initial observation
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial board:\n{observation}")
    print(f"Info: moves={info['moves']}, moves_left={info['moves_left']}, solved={info['solved']}")
    print()

    # Take random actions until episode ends
    import random
    print("Taking random actions...")
    print()

    while True:
        # Random action
        action = random.randint(0, env.num_colors - 1)
        print(f"Action: {action}")

        observation, reward, terminated, truncated, info = env.step(action)
        
        # Render the board after each step (required for human mode)
        env.render()

        print(f"Reward: {reward:.2f}")
        print(f"Info: moves={info['moves']}, moves_left={info['moves_left']}, solved={info['solved']}")
        print()

        if terminated or truncated:
            break

    # Close environment
    env.close()
    print("Episode ended!")


if __name__ == "__main__":
    main()
