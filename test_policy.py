"""Test different policies on the Flood-It environment."""

from floodit_env import FloodItEnv, NUM_COLORS
from policies import (
    RandomPolicy,
    GreedyPolicy,
    ColorChangePolicy,
    FirstRowGreedyPolicy,
    EdgeExpansionPolicy,
    EntropyMaximizePolicy,
)


def run_policy(policy: object, num_steps: int = 5) -> None:
    """Run a policy for a few steps and show results."""
    env = FloodItEnv(render_mode="human")
    obs, _ = env.reset()
    print(f"\n{'='*40}")
    print(f"Testing: {policy.__class__.__name__}")
    print("=" * 40)

    for step in range(num_steps):
        # Get action from policy
        action = policy(obs)
        obs, reward, term, trunc, info = env.step(action)
        env.render()

        if term or trunc:
            break


def main():
    policies = [
        ("Random", RandomPolicy()),
        ("Greedy (most common)", GreedyPolicy()),
        ("ColorChange (avoid flood)", ColorChangePolicy()),
        ("FirstRowGreedy", FirstRowGreedyPolicy()),
        ("EdgeExpansion", EdgeExpansionPolicy()),
        ("EntropyMaximize", EntropyMaximizePolicy()),
    ]

    for name, policy in policies:
        run_policy(policy, num_steps=5)


if __name__ == "__main__":
    main()
