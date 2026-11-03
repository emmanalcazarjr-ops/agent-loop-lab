"""
Lesson 2 - The Tool Belt (registry pattern)

The insight that separates loop engineers from framework users:
    THE LOOP NEVER CHANGES AGAIN.
Adding a capability = write ONE function + ONE schema, decorate it.
The model discovers tools through their descriptions alone,
so description quality decides how well it uses them.

Run me:
    python 02_tool_belt.py --demo
"""

import ast
import json
import operator
import os
import sys
from datetime import datetime

USE_DEMO = "--demo" in sys.argv

MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ------------------------------------------------------- the registry

REGISTRY = {}   # tool name -> python function
SCHEMAS = []    # OpenAI-style schemas sent to the API


def tool(schema):
    """Decorator: register a function so the loop can find it."""
    def register(fn):
        REGISTRY[schema["function"]["name"]] = fn
        SCHEMAS.append(schema)
        return fn
    return register


# --------------------------------------------- capability: calculator

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate arithmetic with + - * / and parentheses.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"],
        },
    },
}


@tool(CALCULATOR_SCHEMA)
def calculator(expression: str) -> str:
    """Safe math: walks the syntax tree instead of eval()ing raw input."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        raise ValueError("only numbers, + - * / and parentheses are allowed")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(walk(tree.body))
    except Exception as exc:
        return f"error: {exc}"


# ---------------------------------------------- capability: notes

NOTES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes.txt")

SAVE_NOTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Save a short note for the user. Use for reminders or facts worth keeping.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"],
        },
    },
}


@tool(SAVE_NOTE_SCHEMA)
def save_note(text: str) -> str:
    with open(NOTES_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"- {text}\n")
    return f"note saved: {text}"


# ---------------------------------------------------------- the brain


def real_llm(messages: list) -> dict:
    import requests

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "tools": SCHEMAS},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def tool_call_msg(name: str, args: dict, call_id: str) -> dict:
    """Build the assistant message requesting one tool use (for demo mode)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


DEMO_SCRIPT = [
    ("calculator", {"expression": "24*7+15"}),
    ("save_note", {"text": "Report draft due Friday"}),
]


def demo_llm(messages: list) -> dict:
    results = [m["content"] for m in messages if m.get("role") == "tool"]
    if len(results) < len(DEMO_SCRIPT):
        name, args = DEMO_SCRIPT[len(results)]
        return tool_call_msg(name, args, f"demo_call_{len(results) + 1}")
    return {
        "role": "assistant",
        "content": f"All done. 24*7+15 = {results[0]}. Also: {results[1]}",
    }


llm = demo_llm if USE_DEMO else real_llm

# ----------------------------------------------------------------- run

if not USE_DEMO and not API_KEY:
    sys.exit("No DEEPSEEK_API_KEY set. Try: python 02_tool_belt.py --demo")

messages = [
    {"role": "system", "content":
        "You are a precise assistant. Use the provided tools when they help."},
    {"role": "user", "content": "What is 24*7+15, and remind me the report draft is due Friday."},
]

MAX_STEPS = 8

print(f"GOAL: {messages[-1]['content']}")
print(f"TOOLS AVAILABLE: {', '.join(REGISTRY)}")

for step in range(1, MAX_STEPS + 1):
    print(f"\n--- STEP {step} ---")
    reply = llm(messages)

    if not reply.get("tool_calls"):
        print(f"\nFINAL ANSWER:\n{reply['content']}")
        break

    messages.append(reply)
    for call in reply["tool_calls"]:
        name = call["function"]["name"]
        print(f"AGENT WANTS: {name}")

        try:
            args = json.loads(call["function"]["arguments"])
        except Exception as exc:
            result = f"error: could not parse arguments ({exc})"
        else:
            fn = REGISTRY.get(name)
            if fn is None:
                result = f"error: unknown tool '{name}'. Available: {', '.join(REGISTRY)}"
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    result = f"error: {type(exc).__name__}: {exc}"

        print(f"TOOL SAID:   {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })
else:
    print("\nStopped: step budget exhausted")
