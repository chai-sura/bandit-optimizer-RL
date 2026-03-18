# THE POLICY.
# This is the reinforcement learning algorithm.
# It makes exactly one decision: which strategy to use next.

import random
import math
from memory import BanditMemory
from strategies import all_strategy_names

class EpsilonGreedyBandit:
    """
    The simplest RL algorithm.

    Rule:
      - With probability epsilon  → EXPLORE (random strategy)
      - With probability 1-epsilon → EXPLOIT (best known strategy)

    Over time, as the agent sees more data, it learns which
    strategy has the highest average reward and exploits it more.
    """

    def __init__(self, memory: BanditMemory, epsilon: float = 0.15):
        self.memory = memory
        self.epsilon = epsilon
        self.strategies = all_strategy_names()

    def select(self) -> tuple[str, bool]:
        """
        Choose which strategy to use.
        Returns: (strategy_name, was_this_an_explore)

        This single function IS the policy.
        """
        if random.random() < self.epsilon:
            # EXPLORE: ignore what we know, try something random
            # Why? Because our current best might not actually be best.
            # We might just not have tried the others enough.
            chosen = random.choice(self.strategies)
            return chosen, True  # True = this was exploration

        else:
            # EXPLOIT: use the strategy with best average reward so far
            stats = self.memory.get_stats()

            # Handle cold start — if nothing has been tried yet
            # use uniform random (same as explore)
            untried = [s for s in stats if s["total_runs"] == 0]
            if untried:
                chosen = random.choice(untried)["strategy"]
                return chosen, True

            # Pick strategy with highest average reward
            best = max(stats, key=lambda s: s["avg_reward"])
            return best["strategy"], False  # False = this was exploitation

    def best_strategy(self) -> str:
        """What does the agent currently think is best?"""
        stats = self.memory.get_stats()
        tried = [s for s in stats if s["total_runs"] > 0]
        if not tried:
            return "not enough data yet"
        return max(tried, key=lambda s: s["avg_reward"])["strategy"]


class UCBBandit:
    """
    Upper Confidence Bound — the smarter version.

    Instead of random exploration, it explores strategically:
    picks strategies it's UNCERTAIN about, not just random ones.

    Formula: score = avg_reward + sqrt(2 * log(total) / n_tries)
                     ↑ exploitation      ↑ exploration bonus
                                           (big when rarely tried)

    Use this after you understand epsilon-greedy.
    Swap it into run.py as a drop-in replacement.
    """

    def __init__(self, memory: BanditMemory):
        self.memory = memory
        self.strategies = all_strategy_names()

    def select(self) -> tuple[str, bool]:
        stats = self.memory.get_stats()
        total = self.memory.total_episodes()

        # Cold start — try everything once first
        untried = [s for s in stats if s["total_runs"] == 0]
        if untried:
            return untried[0]["strategy"], True

        # UCB score for each strategy
        ucb_scores = {}
        for s in stats:
            exploitation = s["avg_reward"]
            # Exploration bonus — gets smaller as strategy gets tried more
            exploration  = math.sqrt(2 * math.log(total + 1) / s["total_runs"])
            ucb_scores[s["strategy"]] = exploitation + exploration

        best = max(ucb_scores, key=ucb_scores.get)

        # If exploration bonus dominated, it was effectively exploring
        avg_rewards = {s["strategy"]: s["avg_reward"] for s in stats}
        was_explore = ucb_scores[best] - avg_rewards[best] > avg_rewards[best]

        return best, was_explore

    def best_strategy(self) -> str:
        stats = self.memory.get_stats()
        tried = [s for s in stats if s["total_runs"] > 0]
        if not tried:
            return "not enough data yet"
        return max(tried, key=lambda s: s["avg_reward"])["strategy"]
