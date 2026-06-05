"""The Decision role: pick the next action for ONE goal.

Decision is deliberately narrow. It sees one goal, the relevant memory hits,
optional artifact bytes attached by Perception, the recent history, and the
list of available MCP tools. It returns EXACTLY ONE of:
  - a final ``answer`` (plain text)
  - a single ``tool_call``
"""
from __future__ import annotations

import sys
from pathlib import Path
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall

# Insert gateway client path
sys.path.insert(0, str(Path(__file__).parent / "llm_gatewayV3"))
from client import LLM

DECISION_SYSTEM_PROMPT = """
You are the Decision role in an agentic cognitive loop. Your job is to decide the next action to satisfy a single CURRENT GOAL.

You will receive:
1. The CURRENT GOAL to satisfy.
2. Relevant memory hits.
3. Attached artifacts (raw bytes/text).
4. Run history.

Your obligations:
1. SELECT ONE RESPONSE MODE: You must respond by either calling an available tool (using the provided tool definitions) or producing a final plain text answer. Never do both or neither.
2. ARTIFACT HANDLE GUARD: Do NOT pass any string starting with "art:" as a parameter to any tool (like fetch_url or read_file). These are internal artifact handles, not real files or URLs. If the goal requires the bytes of an artifact, they are already provided to you in the prompt under "ATTACHED ARTIFACTS:". Read and use that content directly.
3. SUBSTANTIVE ANSWERS: If producing a final answer, provide a substantive response (at least 3 sentences or a detailed list of points). Do not output brief or non-committal meta-answers like "I have fetched the page, how should we proceed?". Answer the goal directly using the available data.
4. TOOL CALLS: Choose the appropriate tool call if the goal is not yet satisfied and requires further information or execution.
"""

def _build_user_message(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
) -> str:
    # Format memory hits
    memory_hits_str = ""
    for h in hits:
        memory_hits_str += f"- Kind: {h.kind}, Descriptor: {h.descriptor}, Value: {h.value}\n"
    if not memory_hits_str:
        memory_hits_str = "None\n"

    # Format attached artifacts
    attached_str = ""
    for art_id, blob in attached:
        try:
            content = blob.decode("utf-8")
        except UnicodeDecodeError:
            content = f"<binary blob: {len(blob)} bytes>"
        attached_str += f"=== ATTACHED ARTIFACT {art_id} ===\n{content}\n==================================\n"
    if not attached_str:
        attached_str = "None\n"

    # Format history
    history_str = ""
    for ev in history:
        history_str += f"- Iter {ev['iter']}, Kind: {ev['kind']}\n"
        if ev['kind'] == "answer":
            history_str += f"  Answer: {ev['text']}\n"
        elif ev['kind'] == "action":
            history_str += f"  Tool: {ev['tool']}({ev['arguments']}) -> {ev['result_descriptor']}\n"
    if not history_str:
        history_str = "No events in history yet.\n"

    msg = f"""CURRENT GOAL: {goal.text} (Goal ID: {goal.id})

RELEVANT MEMORY HITS:
{memory_hits_str}

ATTACHED ARTIFACTS:
{attached_str}

RUN HISTORY:
{history_str}
"""
    return msg

async def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    """Decision.next_step — one LLM call when there is an open goal."""
    llm = LLM()
    provider = "g" if attached else None
    auto_route = None if provider else "decision"
    
    reply = llm.chat(
        provider=provider,
        auto_route=auto_route,
        system=DECISION_SYSTEM_PROMPT,
        prompt=_build_user_message(goal, hits, attached, history),
        tools=mcp_tools,
        tool_choice="auto",
        temperature=0.2,
    )

    if reply.get("tool_calls"):
        tc = reply["tool_calls"][0]
        return DecisionOutput(
            tool_call=ToolCall(name=tc["name"], arguments=tc.get("arguments") or {})
        )
    return DecisionOutput(answer=reply.get("text") or "")
