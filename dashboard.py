# Run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import sqlite3
from memory import BanditMemory
from bandit import EpsilonGreedyBandit
from run import run_episode, QUESTIONS
import random

st.set_page_config(page_title="Bandit Optimizer", layout="wide")
st.title("Multi-Armed Bandit — watching RL learn")

memory = BanditMemory()
bandit = EpsilonGreedyBandit(memory, epsilon=0.15)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
total = memory.total_episodes()
col1.metric("Total episodes", total)
col2.metric("Recent success rate", f"{memory.recent_success_rate()*100:.0f}%")
col3.metric("Best strategy", bandit.best_strategy())
col4.metric("Exploration rate", "15%")

# Strategy leaderboard
st.subheader("Strategy leaderboard — the policy")
stats = memory.get_stats()
if stats:
    df = pd.DataFrame(stats)
    df = df[["strategy","total_runs","total_wins","win_rate","avg_reward"]]
    df.columns = ["Strategy","Runs","Wins","Win rate %","Avg reward"]
    st.dataframe(df, use_container_width=True, hide_index=True)

# Learning curve
st.subheader("Learning curve — is the agent improving?")
history = memory.get_history()
if len(history) > 5:
    df_h = pd.DataFrame(history)
    df_h["rolling_success"] = (
        df_h["success"]
        .astype(float)
        .rolling(window=10, min_periods=1)
        .mean()
    )
    st.line_chart(df_h.set_index("id")["rolling_success"],
                  y_label="Success rate (rolling 10)")

# Strategy usage over time
if len(history) > 5:
    st.subheader("Which strategy got picked over time")
    df_h2 = pd.DataFrame(history)
    strategy_counts = (
        df_h2.groupby(["id","strategy"])
        .size().unstack(fill_value=0)
        .rolling(10, min_periods=1).sum()
    )
    st.area_chart(strategy_counts)

# Run one episode live
st.subheader("Run a live episode")
question = st.text_input(
    "Question:",
    value=random.choice(QUESTIONS)
)

if st.button("Run episode"):
    with st.spinner("Agent deciding and running..."):
        result = run_episode(question, bandit, memory, verbose=False)

    st.success(
        f"Strategy: **{result['strategy']}** | "
        f"Reward: **{result['reward']:.2f}** | "
        f"Success: **{result['success']}**"
    )
    st.rerun()
