# The REWARD FUNCTION.
# This is where definition of "good" gets encoded into math.
# The agent will optimize for exactly this — nothing more, nothing less.

from langchain_openai import ChatOpenAI

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def evaluate_response(question: str, response: str, strategy_name: str) -> dict:
    """
    Score a response from 0.0 to 1.0.
    Returns both the score and the reason — important for debugging.

    In RL terms: this function IS the environment.
    It receives an action (response) and returns a reward (score).
    """

    prompt = f"""You are a strict but fair evaluator.

Question: {question}
Response: {response}

Score this response from 0 to 10 on these criteria:
- Accuracy: Is the answer correct?
- Completeness: Does it fully answer the question?
- Clarity: Is it easy to understand?
- Conciseness: No unnecessary padding?

Return ONLY a JSON object like this:
{{"score": 7, "reason": "Accurate but slightly verbose"}}

Return nothing else. Just the JSON."""

    try:
        result = judge_llm.invoke(prompt).content.strip()
        # Clean markdown if LLM wraps in backticks
        result = result.replace("```json", "").replace("```", "").strip()
        import json
        parsed = json.loads(result)
        raw_score = int(parsed["score"])
        reason = parsed.get("reason", "")

        # Normalize 0-10 to 0.0-1.0
        # This is your reward signal
        normalized = raw_score / 10.0

        return {
            "reward": normalized,
            "raw_score": raw_score,
            "reason": reason,
            "success": normalized >= 0.7  # threshold for "win"
        }

    except Exception as e:
        # If evaluation fails, penalize slightly
        # Don't return 0 — that's too harsh for a parsing error
        return {
            "reward": 0.3,
            "raw_score": 3,
            "reason": f"Evaluation failed: {e}",
            "success": False
        }


# Alternative: use a deterministic reward for verifiable tasks
# This is MUCH better when possible — no LLM judge needed
def evaluate_sql_response(question: str, response: str, expected_answer=None) -> dict:
    """
    For tasks with verifiable answers (math, SQL, code),
    you don't need an LLM judge.
    Just check if the answer is correct.
    This gives a much cleaner reward signal.
    """
    if expected_answer and str(expected_answer).strip() in response:
        return {"reward": 1.0, "reason": "Correct answer", "success": True}
    elif expected_answer:
        return {"reward": 0.0, "reason": "Wrong answer", "success": False}
    else:
        # Fall back to LLM judge if no ground truth
        return evaluate_response(question, response, "")
