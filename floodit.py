"""
Flood-It: Terminal Version
==========================

A simple terminal-based implementation of the Flood-It game using only
Python standard libraries. No external dependencies required.

Game Rules:
- 14x14 grid with colors represented by integers 0-5
- Player starts controlling the top-left connected region
- Each turn, choose a new color to expand your region via flood fill
- Win when entire board becomes one color
- Lose after 25 moves

Controls:
- Enter a number [0-5] to flood with that color
- Type 'q' to quit anytime
"""

from typing import List, Set


# Constants
BOARD_SIZE = 14           # Grid dimensions (14x14)
NUM_COLORS = 6            # Colors are integers 0-5
MAX_MOVES = 25            # Game ends after this many moves


def generate_board() -> List[List[int]]:
    """
    Generate a random 14x14 board with colors 0-5.

    Returns:
        A 2D list representing the board, where each cell holds an integer color.
    """
    board = []
    import random
    for _ in range(BOARD_SIZE):
        row = [random.randint(0, NUM_COLORS - 1) for _ in range(BOARD_SIZE)]
        board.append(row)
    return board


def flood_fill_stack(board: List[List[int]], start_row: int, start_col: int,
                     new_color: int) -> Set[tuple]:
    """
    Perform iterative flood fill using a stack (not recursive).

    Flood fill changes all connected cells of the same color as the start cell
    to the new color, and returns the set of changed positions.

    Algorithm explanation (simple terms):
    1. Start at the given cell position.
    2. Push that position onto a stack.
    3. While the stack is not empty:
       - Pop a position off the stack.
       - If its color is different from start_color, stop exploring it (not connected).
       - If its color matches start_color:
         * Change its color to new_color (if already new_color, skip further work)
         * Push all adjacent positions (up/down/left/right) onto the stack if they
           haven't been visited yet and match the starting color.
    4. Return the set of all positions that were changed.

    Args:
        board: The current board state (modified in place).
        start_row: Starting row index for flood fill.
        start_col: Starting column index for flood fill.
        new_color: The color to fill with.

    Returns:
        A set of (row, col) tuples for all cells that were changed to new_color.
    """
    # Get the starting color (before we change it)
    start_color = board[start_row][start_col]

    if start_color == new_color:
        return frozenset()  # Nothing to flood if choosing current color

    # Track which cells have been visited during this flood fill
    visited: Set[tuple] = set()

    # Stack for iterative traversal (holds positions as (row, col) tuples)
    stack: List[tuple] = [(start_row, start_col)]

    changed_cells: Set[tuple] = set()  # Track cells that actually changed color

    while stack:
        r, c = stack.pop()

        if (r, c) in visited:
            continue

        visited.add((r, c))

        current_color = board[r][c]

        # If cell has already been flooded to new_color, just skip processing neighbors
        if current_color == new_color:
            continue

        # Change this cell's color
        board[r][c] = new_color
        changed_cells.add((r, c))

        # Add valid adjacent cells that still have start_color to the stack
        # Check all four directions: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            # Skip out-of-bounds cells
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue

            # Only traverse cells that have the starting color AND haven't been visited
            if board[nr][nc] == start_color and (nr, nc) not in visited:
                stack.append((nr, nc))

    return frozenset(changed_cells)


def get_connecting_region(board: List[List[int]], new_color: int) -> Set[tuple]:
    """
    Flood fill from top-left with the chosen color and get all changed cells.

    Args:
        board: The current board state (will be modified in place).
        new_color: The color to flood-fill with.

    Returns:
        A frozenset of (row, col) tuples for all cells that were changed.
    """
    return flood_fill_stack(board, 0, 0, new_color)


def check_win(board: List[List[int]]) -> bool:
    """
    Check if the entire board is now a single color (any color is fine).

    Args:
        board: The current board state.

    Returns:
        True if all cells have the same color, False otherwise.
    """
    if not board or not board[0]:
        return True  # Empty board is considered "won"

    target_color = board[0][0]

    for row in board:
        for cell in row:
            if cell != target_color:
                return False

    return True


def render_game(board: List[List[int]], moves_left: int) -> None:
    """
    Print the current board state and game info to the terminal.

    Args:
        board: The current board state.
        moves_left: Number of moves remaining for the player.
    """
    print(f"Moves left: {moves_left}\n")

    # Print board as a grid
    for row in board:
        # Format each number with fixed width for alignment
        line = ' '.join(f"{cell:2d}" for cell in row)
        print(line)

    print()  # Blank line after board


def get_connected_cells(board: List[List[int]], target_color: int,
                        start_row: int = 0, start_col: int = 0) -> Set[tuple]:
    """
    Get all cells connected to the starting cell (same color), without changing colors.

    This is used for validation - to see what region you're currently controlling.

    Args:
        board: The current board state.
        target_color: The color to check connectivity for.
        start_row, start_col: Starting position (defaults to top-left).

    Returns:
        A set of (row, col) tuples for all cells with target_color connected to start.
    """
    if board[start_row][start_col] != target_color:
        return frozenset()

    visited: Set[tuple] = set()
    stack: List[tuple] = [(start_row, start_col)]

    while stack:
        r, c = stack.pop()

        if (r, c) in visited:
            continue

        visited.add((r, c))

        # Check all four adjacent directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue

            if board[nr][nc] == target_color and (nr, nc) not in visited:
                stack.append((nr, nc))

    return frozenset(visited)


def main() -> None:
    """
    Main game loop for the terminal Flood-It game.

    Handles player input, state updates, win/loss conditions, and rendering.
    """
    import random

    print("=" * 40)
    print("Welcome to TERMINAL FLOOD-IT!")
    print("=" * 40)
    print(f"Grid size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Colors available: 0-{NUM_COLORS - 1}")
    print(f"Max moves allowed: {MAX_MOVES}")
    print("=" * 40)
    print()

    # Generate initial random board
    board = generate_board()

    # Track game state
    current_color = board[0][0]  # Color of top-left (starting region)
    moves_used = 0
    total_moves = MAX_MOVES

    # Render the initial board
    render_game(board, total_moves - moves_used)

    print(f"Your starting color is: {current_color}")
    print(f"Goal: Make the entire board one color!")
    print()
    print("Type a number [0-5] to flood with that color.")
    print("Type 'q' at any time to quit.\n")

    # Main game loop
    while moves_used < total_moves:
        # Show current state summary
        connected_count = len(get_connected_cells(board, current_color))
        total_unique_colors = len(set(cell for row in board for cell in row))

        print(f"Connected region size: {connected_count}")
        print(f"Unique colors remaining on board: {total_unique_colors}")
        print(f"Current color of your region: {current_color}")
        print("-" * 40)

        # Get player input
        choice = input("Choose color [0-5] or q: ").strip().lower()

        if choice == 'q':
            print("\nThanks for playing! Goodbye!")
            return

        if not choice.isdigit():
            print(f"Invalid input. Please enter a number [0-{NUM_COLORS - 1}] or 'q'.")
            continue

        selected_color = int(choice)

        if selected_color < 0 or selected_color >= NUM_COLORS:
            print(f"Error: Color must be between 0 and {NUM_COLORS - 1}.")
            continue

        # Check if selecting the current color (invalid move)
        if selected_color == current_color:
            print(f"You already control color {selected_color}. Choose a different color.")
            continue

        # Perform flood fill with chosen color
        changed_cells = flood_fill_stack(board, 0, 0, selected_color)

        if not changed_cells or len(changed_cells) == 0:
            # This can happen if somehow we chose current_color (already checked above)
            print("No change occurred.")
            continue

        moves_used += 1
        remaining_moves = total_moves - moves_used

        # Update current_color after flood fill
        current_color = board[0][0]

        # Check win condition after each move
        if check_win(board):
            render_game(board, remaining_moves)
            print("=" * 40)
            print("🎉 CONGRATULATIONS! YOU WON! 🎉")
            print(f"Final score: {total_moves - remaining_moves} moves used")
            print("=" * 40)
            break
        else:
            # Render updated board and continue
            render_game(board, remaining_moves)

        # Check for loss condition (moves exhausted)
        if remaining_moves <= 0:
            render_game(board, 0)
            print("=" * 40)
            print("💀 GAME OVER - You ran out of moves! 💀")
            print("=" * 40)
            return


if __name__ == "__main__":
    main()
