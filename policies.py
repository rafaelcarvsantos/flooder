"""
Action selection policies for Flood-It environment.

Each policy implements a simple interface:
    def __call__(self, obs) -> int

This separation allows easy experimentation with different strategies
without modifying the environment code.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class Policy(ABC):
    """Base class for all policies."""
    
    @abstractmethod
    def __call__(self, obs: Union[np.ndarray, list]) -> int:
        """Select an action (color) based on observation.
        
        Args:
            obs: Environment observation (14x14 numpy array of colors 0-5)
            
        Returns:
            Action index (0-5, representing a color)
        """
        pass


class RandomPolicy(Policy):
    """Select random color."""
    
    def __call__(self, obs: np.ndarray) -> int:
        return np.random.randint(0, 6)


class GreedyPolicy(Policy):
    """Greedy policy: pick the most frequent color (hardest to spread).
    
    Rationale: Flooding the most common color requires spreading across
    more connected regions, making progress easier.
    """
    
    def __call__(self, obs: np.ndarray) -> int:
        # Count occurrences of each color
        counts = np.array([np.sum(obs == i) for i in range(6)])
        
        # Find color with maximum count (hardest to change)
        most_common_color = np.argmax(counts)
        return int(most_common_color)


class ColorChangePolicy(Policy):
    """Always pick a different color from current flood."""
    
    def __call__(self, obs: np.ndarray) -> int:
        # Find the dominant (flood) color
        total = obs.sum()
        counts = {i: np.sum(obs == i) for i in range(6)}
        
        if total == 0:
            return 0
        
        max_count = max(counts.values())
        flood_colors = [c for c, cnt in counts.items() if cnt == max_count]
        
        # Pick any color different from the flood
        for c in range(6):
            if c not in flood_colors:
                return int(c)
        
        # Fallback: if all colors are equally dominant, pick random
        return int(np.random.choice(flood_colors))


class FirstRowGreedyPolicy(Policy):
    """Pick color that covers most of the first row."""
    
    def __call__(self, obs: np.ndarray) -> int:
        # Count colors in the first row (excluding flood color)
        first_row = obs[0]
        
        # Find dominant color on board
        counts = {i: int(np.sum(obs == i)) for i in range(6)}
        total = obs.sum()
        
        if total == 0:
            return 0
        
        max_count = max(counts.values())
        flood_colors = [c for c, cnt in counts.items() if cnt == max_count]
        
        # Count how many cells of each non-flood color are in row 0
        best_color = -1
        best_coverage = -1
        
        for c in range(6):
            if c not in flood_colors:
                coverage = np.sum(first_row == c)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_color = c
        
        # If no color covers anything, pick random non-flood color
        if best_color < 0:
            available = [c for c in range(6) if c not in flood_colors]
            return int(np.random.choice(available))
        
        return int(best_color)


class EdgeExpansionPolicy(Policy):
    """Pick color that would expand along an edge boundary.
    
    Heuristic: Find boundary between two dominant regions and pick 
    a color different from the flood to potentially cross it.
    """
    
    def __call__(self, obs: np.ndarray) -> int:
        # Simple heuristic: alternate colors cyclically based on step count
        # This is just a placeholder - proper implementation needs more analysis
        
        # For now, pick second-most-common color (often creates better splits)
        counts = {i: int(np.sum(obs == i)) for i in range(6)}
        sorted_colors = sorted(counts.keys(), key=lambda c: counts[c])
        
        if len(sorted_colors) >= 2:
            # Return the second-most-common color
            return int(sorted_colors[-2])
        
        return int(np.random.randint(0, 6))


class EntropyMaximizePolicy(Policy):
    """Pick color that maximizes entropy (creates most diverse outcome).
    
    This is a proxy for "which move will create the most new cells?"
    but simplified to just pick least common color.
    """
    
    def __call__(self, obs: np.ndarray) -> int:
        counts = {i: int(np.sum(obs == i)) for i in range(6)}
        
        # Pick the least common color (creates most diversity)
        min_count = min(counts.values())
        if min_count > 0:
            colors_with_min = [c for c, cnt in counts.items() if cnt == min_count]
            return int(np.random.choice(colors_with_min))
        
        # Fallback to random if all tied
        return int(np.random.randint(0, 6))


# Convenience function to get a policy by name
def get_policy(name: str) -> Policy:
    """Get a policy instance by name."""
    policies = {
        "random": RandomPolicy(),
        "greedy": GreedyPolicy(),
        "color_change": ColorChangePolicy(),
        "first_row": FirstRowGreedyPolicy(),
        "edge": EdgeExpansionPolicy(),
        "entropy": EntropyMaximizePolicy(),
    }
    
    if name not in policies:
        available = ", ".join(policies.keys())
        raise ValueError(f"Unknown policy '{name}'. Available: {available}")
    
    return policies[name]
