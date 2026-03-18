# These are the ACTIONS in RL system.
# The agent will learn which one works best for which task.

STRATEGIES = {
    "direct": {
        "name": "direct",
        "temperature": 0.0,
        "system": "You are a precise assistant. Answer directly and concisely.",
        "template": "{question}",
        "description": "Straight to the answer, no fluff"
    },

    "chain_of_thought": {
        "name": "chain_of_thought",
        "temperature": 0.2,
        "system": "You are a careful assistant. Always think step by step.",
        "template": "Think through this step by step, then give your answer.\n\n{question}",
        "description": "Reasons before answering"
    },

    "few_shot": {
        "name": "few_shot",
        "temperature": 0.0,
        "system": "You are a helpful assistant. Follow the pattern of examples given.",
        "template": """Here are some examples of good answers:

Example 1:
Q: What is 15% of 200?
A: 30

Example 2:
Q: If a train travels 60mph for 2 hours, how far does it go?
A: 120 miles

Now answer this:
Q: {question}
A:""",
        "description": "Learns from examples"
    },

    "expert": {
        "name": "expert",
        "temperature": 0.1,
        "system": "You are a world-class expert. Give authoritative, accurate answers.",
        "template": "As an expert, answer this precisely: {question}",
        "description": "Expert framing"
    }
}

def get_strategy(name: str) -> dict:
    return STRATEGIES[name]

def all_strategy_names() -> list:
    return list(STRATEGIES.keys())
