"""
Flood-It: Gymnasium Environment
================================

Wraps the terminal Flood-It game into a Gymnasium-compatible environment.
Uses existing floodit.py game logic with minimal modifications for importability.
"""

from typing import Any, Dict, Optional, Tuple, Union
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
)


class FloodItEnv(gym.Env):
    """
    Gymnasium environment for the Flood-It game.

    Game rules:
    - 14x14 grid with colors 0-5
    - Each turn, choose a color to flood from top-left region
    - Win when entire board becomes one color
    - Max 25 moves allowed
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str = None) -> None:
        """
        Initialize the Flood-It environment.

        Args:
            render_mode: How to render the environment. Options are:
                - None (no rendering)
                - "human" (render in terminal)
        """
        super().__init__()

        self.render_mode = render_mode

        # Observation space: flattened board as integer array
        # Shape: (BOARD_SIZE, BOARD_SIZE), dtype: int64
        observation_shape = (BOARD_SIZE, BOARD_SIZE)
        self.observation_space = spaces.Box(
            low=0,
            high=NUM_COLORS - 1,
            shape=observation_shape,
            dtype=np.int64,
        )

        # Action space: choose a color from 0 to NUM_COLORS-1
        self.action_space = spaces.Discrete(NUM_COLORS)

        # Store game state attributes (not in observation space)
        self._current_color: int = 0
        self._moves_used: int = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset the environment to initial state.

        Args:
            seed: Seed for random number generation (not used in this simple env)
            options: Additional options (not used currently)

        Returns:
            observation: Initial board state as numpy array
            info: Dictionary with metadata
        """
        super().reset(seed=seed, options=options)

        # Generate a new random board
        board = generate_board()

        # Store board and track game state
        self._board = board  # Keep as list for flood_fill_stack compatibility
        self._current_color = board[0][0]  # Color of top-left cell
        self._moves_used = 0
        self._solved = False

        # Convert to numpy array and return as observation
        observation = np.array(board, dtype=np.int64)

        info = {
            "moves": self._moves_used,
            "moves_left": MAX_MOVES - self._moves_used,
            "solved": False,
        }

        if seed is not None:
            np.random.seed(seed)

        return observation.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Args:
            action: Color to flood (0 to NUM_COLORS-1)

        Returns:
            observation: Updated board state
            reward: -1 for each move, +100 if solved
            terminated: True if game ended by winning
            truncated: True if max moves reached
            info: Dictionary with metadata
        """
        # Validate action
        if not isinstance(action, int):
            action = int(action)

        if action < 0 or action >= NUM_COLORS:
            raise ValueError(
                f"Invalid action {action}. Must be in range [0, {NUM_COLORS - 1}]"
            )

        # Flood fill with chosen color
        changed_cells = flood_fill_stack(self._board, 0, 0, action)

        if len(changed_cells) == 0:
            # No change occurred (could happen if trying to flood current color)
            reward = 0.0
        else:
            # Penalty for each move
            reward = -1.0
            self._moves_used += 1

        # Update current color of top-left region
        self._current_color = self._board[0][0]

        # Check termination conditions
        terminated = check_win(self._board)
        truncated = self._moves_used >= MAX_MOVES

        # Reward for solving game
        if terminated and not getattr(self, "_solved", False):  # Only give reward once
            reward = 100.0 + reward  # Combine win bonus with move penalty
            self._solved = True

        # Convert board to numpy array for observation
        observation = np.array(self._board, dtype=np.int64)

        info = {
            "moves": self._moves_used,
            "moves_left": MAX_MOVES - self._moves_used,
            "solved": terminated,
        }

        return observation.copy(), reward, terminated, truncated, info

    @property
    def board(self) -> np.ndarray:
        """Get the current board as a numpy array."""
        if not hasattr(self, "_board"):
            raise RuntimeError("Board not initialized. Call reset() first.")
        return self._board.copy()

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
            print(f"Moves left: {MAX_MOVES - self._moves_used}", end="\n\n")
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
        lines.append(f"Moves used: {self._moves_used}/{MAX_MOVES}")
        lines.append("=" * 38)

        # Board with color labels
        lines.append("Color key:")
        for c in range(NUM_COLORS):
            lines.append(f"  {c}: color {c}")
        lines.append("=" * 38)

        # Board grid (transpose to print columns as text lines)
        transposed = np.array(self._board).T
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
        action = random.randint(0, NUM_COLORS - 1)
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
