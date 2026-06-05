"""The agent6 loop — wires Memory, Perception, Decision, Action together.

Iterates the fixed five-step rhythm:
    memory.read → perception.observe → all_done?
                                  ↓ no
                       decision.next_step → (answer? loop)
                                  ↓ tool call
                       action.execute → memory.record_outcome → loop
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import action
import decision
import perception
from artifacts import ArtifactStore
from memory import Memory
from schemas import DecisionOutput, Goal


STATE_DIR = Path(__file__).parent / "state"
MCP_SERVER = Path(__file__).parent / "mcp_server.py"
MAX_ITERATIONS = 15


async def run(query: str) -> str:
    """Run one full agent loop for ``query``. Returns the final answer text."""
    memory = Memory(STATE_DIR / "memory.json")
    artifacts = ArtifactStore(STATE_DIR / "artifacts")
    run_id = uuid.uuid4().hex[:8]

    # Setup: classify the user's query into Memory so durable facts survive.
    memory.remember(query, source="user_query", run_id=run_id)

    history: list[dict] = []
    prior_goals: list[Goal] = []

    print("=" * 78)
    print(f"agent6.py  run_id={run_id}")
    print(f"query: {query!r}")
    print("=" * 78)

    params = StdioServerParameters(
        command=sys.executable, args=[str(MCP_SERVER)]
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools_raw = (await session.list_tools()).tools
            mcp_tools = [_mcp_tool_to_dict(t) for t in mcp_tools_raw]
            print(f"[mcp] tools: {[t['name'] for t in mcp_tools]}")

            for it in range(1, MAX_ITERATIONS + 1):
                print(f"\n--- iter {it} ---")

                # 1. Memory.read
                hits = memory.read(query, history)
                print(f"[memory.read] {len(hits)} hits")

                # 2. Perception.observe
                obs = await perception.observe(
                    query, hits, history, prior_goals, run_id
                )
                prior_goals = obs.goals
                for g in obs.goals:
                    flag = "done" if g.done else "open"
                    attach = (
                        f"  attach={g.attach_artifact_id}"
                        if g.attach_artifact_id else ""
                    )
                    print(f"[perception] [{flag}] {g.text}{attach}")

                # 3. Loop check
                if obs.all_done():
                    print(f"[done] all {len(obs.goals)} goals satisfied")
                    break

                goal = obs.next_unfinished()

                # 4. Optional artifact attachment
                attached: list[tuple[str, bytes]] = []
                if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                    blob = artifacts.get_bytes(goal.attach_artifact_id)
                    attached.append((goal.attach_artifact_id, blob))
                    print(f"[attach] {goal.attach_artifact_id} ({len(blob)} bytes)")

                # 5. Decision.next_step
                out: DecisionOutput = await decision.next_step(
                    goal, hits, attached, history, mcp_tools
                )

                if out.is_answer:
                    print(f"[decision] ANSWER: {out.answer[:120]}")
                    history.append({
                        "iter": it, "kind": "answer",
                        "goal_id": goal.id, "text": out.answer,
                    })
                    continue

                # 6. Action.execute
                tc = out.tool_call
                print(f"[decision] TOOL_CALL: {tc.name}({tc.arguments})")
                result_text, art_id = await action.execute(
                    session, tc, artifacts
                )
                print(f"[action] -> {result_text[:120]}")

                # 7. Memory.record_outcome
                memory.record_outcome(
                    tool_call=tc, result_text=result_text,
                    artifact_id=art_id, run_id=run_id, goal_id=goal.id,
                )
                history.append({
                    "iter": it, "kind": "action",
                    "goal_id": goal.id, "tool": tc.name,
                    "arguments": tc.arguments,
                    "result_descriptor": result_text[:300],
                    "artifact_id": art_id,
                })
            else:
                print(f"\n[max-iter] hit MAX_ITERATIONS={MAX_ITERATIONS}")

    for ev in reversed(history):
        if ev.get("kind") == "answer":
            return ev["text"]
    return "no answer produced"


def _mcp_tool_to_dict(t) -> dict:
    """Reshape an MCP tool description into the gateway's expected envelope."""
    return {
        "name": t.name,
        "description": t.description or "",
        "input_schema": t.inputSchema or {"type": "object", "properties": {}},
    }


def main() -> None:
    query = " ".join(sys.argv[1:]) or "Hello, world."
    final = asyncio.run(run(query))
    print("\n" + "=" * 78)
    print(f"FINAL: {final}")
    print("=" * 78)


if __name__ == "__main__":
    main()
