# The agent's experience log.
# Every (action, reward) pair ever seen is stored here.
# This is what the agent learns from.

import sqlite3
import json
from datetime import datetime
from pathlib import Path

class BanditMemory:
    def __init__(self, db_path="bandit_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Every episode (one run of the loop) gets recorded
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT,
                    question      TEXT,
                    strategy      TEXT,
                    response      TEXT,
                    reward        REAL,
                    raw_score     INTEGER,
                    reason        TEXT,
                    success       BOOLEAN,
                    was_explore   BOOLEAN
                )
            """)
            # Running totals per strategy — updated after each episode
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_stats (
                    strategy      TEXT PRIMARY KEY,
                    total_reward  REAL DEFAULT 0.0,
                    total_runs    INTEGER DEFAULT 0,
                    total_wins    INTEGER DEFAULT 0,
                    avg_reward    REAL DEFAULT 0.0
                )
            """)
            # Seed the stats table with all strategies at zero
            from strategies import all_strategy_names
            for name in all_strategy_names():
                conn.execute("""
                    INSERT OR IGNORE INTO strategy_stats
                    (strategy, total_reward, total_runs, total_wins, avg_reward)
                    VALUES (?, 0.0, 0, 0, 0.0)
                """, (name,))

    def record(self, question, strategy, response,
               reward, raw_score, reason, success, was_explore):
        """Save one episode and update running stats."""
        with sqlite3.connect(self.db_path) as conn:
            # Log the full episode
            conn.execute("""
                INSERT INTO episodes
                (timestamp, question, strategy, response,
                 reward, raw_score, reason, success, was_explore)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), question, strategy,
                  response, reward, raw_score, reason, success, was_explore))

            # Update running averages
            conn.execute("""
                UPDATE strategy_stats
                SET total_reward = total_reward + ?,
                    total_runs   = total_runs + 1,
                    total_wins   = total_wins + ?,
                    avg_reward   = (total_reward + ?) / (total_runs + 1)
                WHERE strategy = ?
            """, (reward, 1 if success else 0, reward, strategy))

    def get_stats(self) -> list:
        """Get current performance stats for all strategies."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT strategy, total_reward, total_runs,
                       total_wins, avg_reward
                FROM strategy_stats
                ORDER BY avg_reward DESC
            """).fetchall()
        return [
            {
                "strategy":     r[0],
                "total_reward": round(r[1], 3),
                "total_runs":   r[2],
                "total_wins":   r[3],
                "avg_reward":   round(r[4], 3),
                "win_rate":     round(r[3] / r[2] * 100, 1) if r[2] > 0 else 0
            }
            for r in rows
        ]

    def get_history(self) -> list:
        """Get all episodes for plotting the learning curve."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT id, strategy, reward, success, was_explore
                FROM episodes ORDER BY id
            """).fetchall()
        return [
            {"id": r[0], "strategy": r[1], "reward": r[2],
             "success": r[3], "was_explore": r[4]}
            for r in rows
        ]

    def total_episodes(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]

    def recent_success_rate(self, n=20) -> float:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT AVG(CAST(success AS FLOAT))
                FROM (SELECT success FROM episodes
                      ORDER BY id DESC LIMIT ?)
            """, (n,)).fetchone()[0]
        return round(result or 0.0, 3)

    def reset(self):
        """Wipe memory — useful for experiments."""
        Path(self.db_path).unlink(missing_ok=True)
        self._init_db()
        print("Memory wiped. Starting fresh.")
