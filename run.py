# THE TRAINING LOOP.
# This is where State → Action → Reward → Memory → Repeat happens.

from langchain_openai import ChatOpenAI
from strategies import get_strategy
from evaluator import evaluate_response
from memory import BanditMemory
from bandit import EpsilonGreedyBandit   # swap to UCBBandit when ready

import os
import time

# Your test questions — the tasks the agent will practice on
QUESTIONS = [
    "What is the capital of Australia?",
    "Explain what an API is in one sentence.",
    "What is 17% of 340?",
    "What does SQL stand for and what is it used for?",
    "What is the difference between a list and a tuple in Python?",
    "Explain machine learning in simple terms.",
    "What is the time complexity of binary search?",
    "What does REST stand for in REST API?",
    "What is a foreign key in a database?",
    "Explain what recursion is with a simple example.",
    "What is the difference between GET and POST requests?",
    "What does ACID stand for in databases?",
    "What is a neural network in simple terms?",
    "What is the difference between SQL and NoSQL?",
    "Explain what a webhook is.",
]

def run_episode(question: str, bandit, memory: BanditMemory, verbose=True):
    """
    ONE complete RL loop:
    1. Agent observes state (the question)
    2. Agent selects action (which strategy to use)
    3. Agent takes action (generates a response)
    4. Environment returns reward (evaluator scores it)
    5. Agent records experience (memory stores it)
    """

    # STEP 1: SELECT ACTION
    strategy_name, was_explore = bandit.select()
    strategy = get_strategy(strategy_name)

    if verbose:
        mode = "EXPLORE" if was_explore else "EXPLOIT"
        print(f"\n[{mode}] Strategy: {strategy_name}")
        print(f"Question: {question[:60]}...")

    # STEP 2: TAKE ACTION — generate response using selected strategy
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=strategy["temperature"]
    )

    messages = [
        {"role": "system", "content": strategy["system"]},
        {"role": "user",   "content": strategy["template"].format(question=question)}
    ]

    response = llm.invoke(messages).content.strip()

    if verbose:
        print(f"Response: {response[:80]}...")

    # STEP 3: GET REWARD — evaluate the response
    evaluation = evaluate_response(question, response, strategy_name)
    reward  = evaluation["reward"]
    success = evaluation["success"]
    reason  = evaluation["reason"]

    if verbose:
        print(f"Reward: {reward:.2f} | Success: {success} | {reason}")

    # STEP 4: STORE EXPERIENCE
    memory.record(
        question    = question,
        strategy    = strategy_name,
        response    = response,
        reward      = reward,
        raw_score   = evaluation["raw_score"],
        reason      = reason,
        success     = success,
        was_explore = was_explore
    )

    return {"strategy": strategy_name, "reward": reward, "success": success}


def print_stats(memory: BanditMemory, bandit):
    """Print current state of the policy."""
    print("\n" + "="*50)
    print(f"POLICY STATE after {memory.total_episodes()} episodes")
    print(f"Recent success rate: {memory.recent_success_rate()*100:.0f}%")
    print(f"Agent currently thinks best strategy: {bandit.best_strategy()}")
    print("\nStrategy leaderboard:")
    for s in memory.get_stats():
        bar = "█" * int(s["avg_reward"] * 20)
        print(f"  {s['strategy']:20} {bar:20} avg={s['avg_reward']:.3f}  runs={s['total_runs']}  wins={s['total_wins']}")
    print("="*50)


def main(n_episodes=50):
    memory = BanditMemory()
    bandit = EpsilonGreedyBandit(memory, epsilon=0.15)

    print(f"Starting bandit optimizer — {n_episodes} episodes")
    print(f"Strategies: {bandit.strategies}")
    print(f"Epsilon: {bandit.epsilon} (15% exploration)")

    for i in range(n_episodes):
        # Cycle through questions
        question = QUESTIONS[i % len(QUESTIONS)]

        run_episode(question, bandit, memory, verbose=True)

        # Print stats every 10 episodes — watch the policy shift
        if (i + 1) % 10 == 0:
            print_stats(memory, bandit)

        time.sleep(0.3)  # be gentle on the API

    print("\nFINAL RESULTS")
    print_stats(memory, bandit)


if __name__ == "__main__":
    main(n_episodes=50)
