"""The Perception role: orchestrator.

Runs every iteration of the loop. Reads the user query, current memory hits,
run history so far, and the prior goal list. Emits a fresh ``Observation``
containing the current goal list with ``done`` flags and optional artifact
attachments on the next unfinished goal.

Pinned to Gemini via ``provider="g"`` (when the real call is wired up) because
the multi-step procedure isn't reliable on smaller models.
"""
from __future__ import annotations

import sys
from pathlib import Path
from schemas import Goal, MemoryItem, Observation

# Insert gateway client path
sys.path.insert(0, str(Path(__file__).parent / "llm_gatewayV3"))
from client import LLM

PERCEPTION_SYSTEM_PROMPT = """
You are the Perception role in an agentic cognitive loop. Your job is to observe the current execution state and manage a stable plan (a list of Goals) to satisfy the user's query.

You will receive:
1. The user's original query.
2. The list of prior goals (which is empty on the first turn).
3. The memory hits retrieved from memory.
4. The history of actions and outcomes from the current run.

Your obligations:
1. DECOMPOSE ON FIRST TURN: If the list of prior goals is empty, decompose the user's query into a list of sequential, bounded goals.
   - Each goal must be a short, direct imperative statement (e.g. "Search for 'Python asyncio best practices'", "Fetch the URL 'https://example.com'", "Extract birth and death dates of Claude Shannon").
   - Give each goal a unique ID starting with 'g1', 'g2', etc.
   - Set 'done' to false, and 'attach_artifact_id' to null for all goals.
2. RE-JUDGE DONE FLAGS: On subsequent calls, review the prior goals. For each goal, read the history of actions. If the history shows that the goal has been successfully completed, set 'done' to true. Once a goal is marked done, it MUST stay done. Do not unmark it as done.
3. PRESERVE GOAL LIST INTEGRITY: Keep the goal list stable. Do not reorder, delete, insert new goals in the middle, or modify the text of existing goals. Only update their 'done' and 'attach_artifact_id' attributes.
4. ATTACH ARTIFACTS: For the FIRST unfinished goal in the list, check if it needs the raw content of a previously fetched web page/file (an artifact) to be solved.
   - In the prompt, you will see a list of MEMORY HITS, some of which are marked with an attachment index (e.g., "[ATTACHMENT INDEX: 0]").
   - If the first unfinished goal requires the bytes of an artifact, set 'attach_artifact_id' to the string index (e.g., "0", "1") of the memory hit that holds the corresponding artifact.
   - If no artifact is needed for the current goal, or for all other goals, set 'attach_artifact_id' to null.
   - Only attach an artifact to the FIRST unfinished goal. All other goals must have 'attach_artifact_id' set to null.

You must output a JSON object matching the Observation schema:
{
  "goals": [
    {
      "id": "g1",
      "text": "Goal description",
      "done": false,
      "attach_artifact_id": null
    }
  ]
}
"""

def _build_user_message(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
) -> str:
    # 1. Format prior goals
    if prior_goals:
        goals_section = "\n".join(
            f"- ID: {g.id}, Done: {g.done}, Attached Artifact: {g.attach_artifact_id or 'None'}, Text: {g.text}"
            for g in prior_goals
        )
    else:
        goals_section = "NO PRIOR GOALS (First Turn)"

    # 2. Format memory hits and map attachment indices
    memory_hits_str = ""
    art_index = 0
    for h in hits:
        art_info = ""
        if h.artifact_id:
            art_info = f" [ATTACHMENT INDEX: {art_index}]"
            art_index += 1
        memory_hits_str += f"- Kind: {h.kind}, Descriptor: {h.descriptor}{art_info}\n"
    if not memory_hits_str:
        memory_hits_str = "None\n"

    # 3. Format history
    history_str = ""
    for ev in history:
        history_str += f"- Iter {ev['iter']}, Kind: {ev['kind']}\n"
        if ev['kind'] == "answer":
            history_str += f"  Answer: {ev['text']}\n"
        elif ev['kind'] == "action":
            history_str += f"  Tool: {ev['tool']}({ev['arguments']}) -> {ev['result_descriptor']}\n"
            if ev.get("artifact_id"):
                history_str += f"  Created Artifact ID: {ev['artifact_id']}\n"
    if not history_str:
        history_str = "No events in history yet.\n"

    # 4. Construct user message
    msg = f"""USER QUERY: {query}

PRIOR GOALS:
{goals_section}

MEMORY HITS:
{memory_hits_str}

RUN HISTORY:
{history_str}
"""
    return msg

async def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    """Perception.observe — one LLM call per iteration (pinned to Gemini)."""
    # Build mapping from positional attachment index string to actual artifact_id
    art_map = {}
    art_index = 0
    for h in hits:
        if h.artifact_id:
            art_map[str(art_index)] = h.artifact_id
            art_index += 1

    llm = LLM()
    reply = llm.chat(
        provider="g",  # pinned to Gemini
        system=PERCEPTION_SYSTEM_PROMPT,
        prompt=_build_user_message(query, hits, history, prior_goals),
        response_format={
            "type": "json_schema",
            "schema": Observation.model_json_schema(),
            "name": "Observation",
            "strict": True,
        },
        temperature=1.0,  # avoids low-temp Gemini looping
    )

    parsed = reply.get("parsed")
    if not parsed:
        # If parsing failed, fallback or raise error
        raise ValueError(f"Perception failed to parse output: {reply.get('text')}")

    obs = Observation.model_validate(parsed)

    # Map positional indexes back to real artifact_ids
    for goal in obs.goals:
        if goal.attach_artifact_id:
            val = str(goal.attach_artifact_id).strip()
            if val in art_map:
                goal.attach_artifact_id = art_map[val]
            elif val.startswith("art:"):
                # LLM outputted actual handle directly
                pass
            else:
                goal.attach_artifact_id = None

    return obs
