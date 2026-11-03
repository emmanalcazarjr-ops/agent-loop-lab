# Agent Loop Lab

Learn **loop engineering** by building the core of every AI agent from scratch.
No LangChain, no framework. ~120 lines per lesson. Python 3.11.

**Live demo:** [emmanalcazarjr.com/agentloop](https://emmanalcazarjr.com/agentloop/) runs the same bounded loop in the browser, step by step, with a step budget you can change.

## The loop (the whole secret)

```
        ┌──────────────────────────────────────────────┐
        │                  THE LOOP                    │
        │                                              │
 goal   │   ┌───────┐  "call get_weather"   ┌────────┐ │
  ────▶ │   │  LLM  │ ────────────────────▶ │ tools  │ │
        │   │       │ ◀──────────────────── │ (you)  │ │
        │   └───────┘    observation/result └────────┘ │
        │       │                                      │
        │       ▼  no more tool calls                  │
        │   final answer                               │
        └──────────────────────────────────────────────┘
```

The model never *runs* anything. It only **asks**. Your loop runs the tool
and feeds the result back as a message with `role: "tool"`. Repeat until
the model replies without tool calls.

## Lessons

| File | What you learn |
|---|---|
| `01_minimal_loop.py` | The bare loop: ask → tool call → observe → repeat → answer |
| `02_tool_belt.py` | Registry pattern: add capabilities without ever touching the loop again |
| `03_guardrails.py` | Step budgets, errors-as-observations, repeat detection, forced final answer |

## Run it

```powershell
cd agent-loop-lab

# offline demo mode - watch the loop work, costs nothing:
python 01_minimal_loop.py --demo
python 02_tool_belt.py --demo
python 03_guardrails.py --demo     # scripted to fail, then recover
```

Real mode (any OpenAI-compatible provider — DeepSeek, OpenAI, Groq, Ollama):

```powershell
pip install requests
$env:DEEPSEEK_API_KEY = "sk-..."          # or OPENAI-compatible key
$env:LLM_BASE_URL = "https://api.deepseek.com"   # default; Groq: https://api.groq.com/openai/v1
$env:LLM_MODEL = "deepseek-chat"
python 01_minimal_loop.py
```

## The one insight

> **The loop is invariant. Capabilities are data.**

Once lesson 1 works, you never rewrite the loop. A new skill is one function
+ one schema dict. The model only knows tools by their `description` strings,
so writing good descriptions *is* prompt engineering for agents.

## Where this goes next (homework)

1. **Context management** — print `sum(len(str(m)) for m in messages)` each
   step and watch history grow. Then trim old tool observations once over a
   budget. That's context engineering in one afternoon.
2. **A real tool** — swap fake weather for a real HTTP API.
3. **Wire into Rush** — same pattern powers `@RushDailyBot`'s intent router;
   a food-photo → calorie-lookup tool would be one more registry entry.
4. **Sub-agents** — add a tool whose implementation *is another loop*.
   That's how deep research agents are born.
