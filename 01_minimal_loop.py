"""
Lesson 1 - The Minimal Agent Loop

The whole idea in one sentence:
    Ask the model. It either answers OR asks for a tool.
    You run the tool, hand the result back, repeat until it answers.

This is ~90% of what frameworks like LangChain actually do.
Everything else is plumbing on top of this while-loop.

Run me:
    python 01_minimal_loop.py --demo   (offline, no API key needed)
    python 01_minimal_loop.py          (needs DEEPSEEK_API_KEY in env/.env)
"""

import json
import os
import sys

USE_DEMO = "--demo" in sys.argv

MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ------------------------------------------------------------- the tool

WEATHER_DB = {
    "manila": "33C, humid, rain later",
    "tokyo": "18C, clear",
    "london": "11C, drizzle",
}


def get_weather(city: str) -> str:
    report = WEATHER_DB.get(city.strip().lower())
    return report or f"No weather data for '{city}'"


WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
    },
}

# ------------------------------------------------------------ the brain


def real_llm(messages: list) -> dict:
    """One HTTP round-trip to any OpenAI-compatible chat API."""
    import requests

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": messages,
            "tools": [WEATHER_SCHEMA],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def demo_llm(messages: list) -> dict:
    """A scripted stand-in for the model so you can watch the loop offline."""
    already_used_tool = any(m.get("role") == "tool" for m in messages)
    if already_used_tool:
        observation = messages[-1]["content"]
        return {
            "role": "assistant",
            "content": f"Weather check complete: {observation}",
        }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "demo_call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Manila"}),
                },
            }
        ],
    }


llm = demo_llm if USE_DEMO else real_llm

# ----------------------------------------------------------------- run

if not USE_DEMO and not API_KEY:
    sys.exit("No DEEPSEEK_API_KEY set. Try: python 01_minimal_loop.py --demo")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Manila right now?"},
]

MAX_STEPS = 5  # hard cap: a runaway loop burns money

print(f"GOAL: {messages[-1]['content']}")

for step in range(1, MAX_STEPS + 1):
    print(f"\n--- STEP {step}: ask the model ---")
    reply = llm(messages)

    if not reply.get("tool_calls"):          # no tool request -> we're done
        print(f"\nFINAL ANSWER:\n{reply['content']}")
        break

    messages.append(reply)                   # remember the model's move
    for call in reply["tool_calls"]:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])
        print(f"AGENT WANTS: {name}({args})")

        result = get_weather(**args)         # <- actually do the work here
        print(f"TOOL SAID:   {result}")

        messages.append({                    # feed the observation back
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })
else:
    print("\nStopped: step budget exhausted (this is a real bug class in prod agents)")
