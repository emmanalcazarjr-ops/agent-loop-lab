"""
Lesson 3 - Guardrails (what separates toys from production)

Three failure modes every real loop must survive:
    1. Runaway loops        -> step budget + forced final answer
    2. Tool errors          -> errors become observations, never crashes
    3. Repeated bad calls   -> detect identical repeats, warn the model

Rule: the loop is allowed to fail. The loop is NOT allowed to crash
or burn money silently.

Run me:
    python 03_guardrails.py --demo   (scripted to fail, then recover)
"""

import ast
import json
import operator
import os
import sys

USE_DEMO = "--demo" in sys.argv

MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MAX_STEPS = 6          # hard money/time budget
MAX_IDENTICAL = 1      # same failing call twice -> warn

# ------------------------------------------------------------- tools

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
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


def calculator(expression: str) -> str:
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


REGISTRY = {"calculator": calculator}
SCHEMAS = [CALCULATOR_SCHEMA]

# ------------------------------------------------------- llm clients


def real_llm(messages: list, use_tools: bool = True) -> dict:
    import requests

    body = {"model": MODEL, "messages": messages}
    if use_tools:
        body["tools"] = SCHEMAS
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def tool_call_msg(name: str, args: dict, call_id: str) -> dict:
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
    ("calculator", {"expression": "open the pod bay doors"}),   # will error
    ("calculator", {"expression": "open the pod bay doors"}),   # repeat -> warning
    ("calculator", {"expression": "1984 / 4"}),                 # recovers
]


def demo_llm(messages: list, use_tools: bool = True) -> dict:
    results = [m["content"] for m in messages if m.get("role") == "tool"]
    if not use_tools or len(results) >= len(DEMO_SCRIPT):
        summary = "; ".join(results) if results else "nothing learned"
        return {"role": "assistant",
                "content": f"Final answer based on what I have ({summary})"}
    name, args = DEMO_SCRIPT[len(results)]
    return tool_call_msg(name, args, f"demo_call_{len(results) + 1}")


if USE_DEMO:
    def llm(messages, use_tools=True):
        return demo_llm(messages, use_tools)
else:
    def llm(messages, use_tools=True):
        return real_llm(messages, use_tools)

# ----------------------------------------------------------------- run

if not USE_DEMO and not API_KEY:
    sys.exit("No DEEPSEEK_API_KEY set. Try: python 03_guardrails.py --demo")

messages = [
    {"role": "system", "content":
        "You are a careful assistant. If a tool fails, read the error "
        "and try a different approach instead of repeating yourself."},
    {"role": "user", "content": "Divide 1984 by 4 using the calculator."},
]

steps_used = 0
last_signature = None
already_warned = False

print(f"GOAL: {messages[-1]['content']}")
print(f"BUDGET: {MAX_STEPS} steps\n")

while True:
    if steps_used >= MAX_STEPS:
        print(f"\n!! BUDGET HIT after {steps_used} steps - forcing final answer")
        messages.append({
            "role": "user",
            "content": "Step budget exhausted. Give your best answer NOW without tools.",
        })
        reply = llm(messages, use_tools=False)
        print(f"FORCED FINAL ANSWER:\n{reply['content']}")
        break

    steps_used += 1
    print(f"--- STEP {steps_used} ---")
    reply = llm(messages)

    calls = reply.get("tool_calls") or []
    if not calls:
        print(f"\nFINAL ANSWER:\n{reply['content']}")
        break

    signature_parts = []
    messages.append(reply)

    for call in calls:
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception as exc:
            result = f"error: could not parse arguments ({exc})"
            args = {}
        else:
            fn = REGISTRY.get(name)
            if fn is None:
                result = f"error: unknown tool '{name}'"
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    result = f"error: {type(exc).__name__}: {exc}"

        signature_parts.append(f"{name}:{json.dumps(args, sort_keys=True)}")
        print(f"AGENT WANTS: {name}({args})")
        print(f"TOOL SAID:   {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })

    signature = "|".join(signature_parts)
    if signature == last_signature and not already_warned:
        already_warned = True
        messages.append({
            "role": "user",
            "content": "WARNING: you just repeated an identical call that failed "
                       "the same way. Change your approach or answer directly.",
        })
        print(">> LOOP DETECTED - warning injected")
    last_signature = signature
