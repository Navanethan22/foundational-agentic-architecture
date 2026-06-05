# Session 6 Agentic Architecture — Summary & Key Learnings

This document summarizes the architecture, execution results, and key engineering insights from implementing and verifying the Session 6 Agentic Architecture loop (comprising Memory, Perception, Decision, and Action roles).

---

## 1. What We Did

We implemented a custom cognitive loop from scratch using **Pydantic contracts** for type safety at all role boundaries, integrated with a local **LLM Gateway V3** and a **9-tool MCP Server**:

1.  **Memory Role (`memory.py`)**: Structured memory using standard keyword lookup for reads, and a strict Gemini-based classifier (`MemoryClassifierOutput` schema) to write text inputs to `state/memory.json`.
2.  **Perception Role (`perception.py`)**: Decomposed user requests into structured, stable lists of goals (`Goal` schemas), re-judged goal done-ness on subsequent iterations, and attached relevant artifacts to goals.
3.  **Decision Role (`decision.py`)**: Decided the next logical action (either outputting an answer or selecting a tool call) based on goals, memory hits, and history.
4.  **Action Role (`action.py`)**: Executed tool dispatches over the MCP channel, saved raw responses as binary/text files to `state/artifacts/`, and returned outcome descriptors.
5.  **LLM Gateway V3**: Configured and ran the Gateway in the background on port `8101` to coordinate model calls.

---

## 2. Verification Results

We verified the agent loop against the four target queries. All runs finished well within their respective iteration budgets:

| Query | Description | Budget | Actual Iterations | Status / Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Query A** | Shannon Wikipedia (Artifact attachment) | 6 | **2** | 🟢 **Success**: Retrieved Wikipedia page, saved details to artifact, and attached it. |
| **Query B** | Tokyo Itinerary + Weather (Multi-goal memory) | 12 | **4** | 🟢 **Success**: Planned itinerary and retrieved weather, merging results. |
| **Query C (Run 1)** | Mom's Birthday (Write phase & reminders) | 8 | **7 → 4** | 🟢 **Fixed**: Originally 🟡 (7 iters, wrong date math — `get_time` failed on Windows, no IANA tz db). After adding `tzdata` and clearing stale state, a clean re-run finished in **4 iters** with the correct answer (15 days away, Saturday). See learnings B, F, G. |
| **Query C (Run 2)** | Mom's Birthday (Read phase & carryover) | 4 | **2** | 🟢 **Success**: Retrieved birthday directly from memory in a separate run without calling tools. |
| **Query D** | Asyncio Research (Multi-source synthesis) | 14 | **3** | 🟢 **Success**: Performed real DuckDuckGo search, read results via artifact, and synthesized consensus. |

---

## 3. Key Learnings & Engineering Discoveries

### 🧠 A. The Memory Classification Hallucination Bug
*   **The Issue**: During the first run of Query D (`"Search for..."`), the agent immediately outputted a final answer in Iteration 1 without calling tools. Investigation of `state/memory.json` revealed that the memory classifier (`memory.remember(...)`) had parsed the user query instruction as a `tool_outcome` that had already occurred, and hallucinated a complete list of search results inside the value payload.
*   **The Learning**: Instructing an LLM to structure free-form text can cause it to hallucinate outcomes when the input text contains active command words. 
*   **The Fix**: 
    1. We supplied the context `source` (e.g. `user_query`) to the prompt to let the LLM know where the text came from.
    2. We added strict rules to `MEMORY_CLASSIFICATION_SYSTEM_PROMPT` explicitly stating that instructions/queries are not tool outcomes and must not have fake results generated for them.

### 🕒 B. Windows ZoneInfo Timezone Constraint
*   **The Issue**: During Query C Run 1, the agent spent 5 iterations failing tool calls (`get_time`) and retrying different timezone names (`UTC`, `America/New_York`, `Asia/Kolkata`).
*   **The Learning**: On Windows, Python's standard library `zoneinfo` module does not bundle an IANA timezone database. Calling `get_time` with a standard timezone string throws a `KeyError: 'No time zone found with key...'`. 
*   **The Solution**: In environments without a default database, either the `tzdata` package must be installed in Python (`pip install tzdata`), or the agent must gracefully handle tool failure and fallback to direct mathematical reasoning.

### 🛡️ C. Artifact Handle Guarding & Positional Indexing
*   **The Issue**: LLMs frequently fail when handling raw UUID-like artifact IDs (e.g., `art:57d3d7675f...`), leading to hallucinations or passing internal keys into tools.
*   **The Solution**: 
    - In Perception, we mapped memory hit artifact IDs to short positional indexes (e.g., `[ATTACHMENT INDEX: 0]`) in the prompt. The LLM simply outputs the integer index, which the code maps back to the raw `art:...` handle.
    - In Decision, we added a strict system prompt rule prohibiting the passing of `art:...` strings to external tools (since the actual bytes of the artifact are already injected directly into Decision's prompt under `ATTACHED ARTIFACTS:`).

### ⚡ D. Large Contexts & Direct Routing
*   **The Issue**: When attaching larger documents or scrapings (artifacts), the prompt size grew, causing gateway routing calls to exceed standard token limits and throw `503 Service Unavailable`.
*   **The Solution**: In `decision.py`, we bypassed the gateway's token router and set `provider="g"` (direct Gemini routing) whenever files/artifacts are attached to handle larger context windows reliably.

### 🪞 E. Resilience Masks Failure — Why the Build Agent Never Caught the `get_time` Bug
*   **The Issue**: The `tzdata` fix (learning B) was tiny — one line in `pyproject.toml` plus `uv sync`. The coding agent (Antigravity) that built the project *could* have run that itself. It didn't. The question is why.
*   **The root cause — failure-tolerance is a double-edged sword**: This loop is *failure-tolerant by design* (a failed action doesn't crash; the goal stays open and the next iteration retries — see "Failure is handled by the architecture, not by error code"). When `get_time` threw on Windows, that resilience **converted a hard error into a quiet degradation**: the loop retried, eventually produced a final answer, and exited cleanly. The same property cut two ways:
    *   **At runtime** it *protected the agent* — it didn't die on a broken tool.
    *   **At build time** it *blinded the builder* — Antigravity observed no crash, a zero exit code, and a printed answer. From its vantage the run was a **success**, so there was nothing to trigger a fix.
*   **The deeper learning — "ran" is not "right"**: An autonomous coding agent verifies that code *ran*. It does **not** verify that the result is *correct* — that requires an **oracle** (a known-correct answer to compare against) or a **critic that reads the process, not just the output**. The tell here was never the final answer; it was the *iteration trace* showing `get_time` retried 5× with different timezone names. Catching it required (a) knowing the reminder date was wrong (semantic error, not a runtime error) and (b) reading the trace while *looking for waste*. Neither is available to a build agent that treats "produced output" as "done."
*   **The takeaway**: Resilience and intelligence are different, and so are *running* and *verifying*. When a system is failure-tolerant, "it ran" and "it's right" can diverge **silently** — the agent will happily report success over a degraded result. A human (or a separate critic agent) reading the trace against a known-good answer is not redundant QA; it is the one step the build agent structurally cannot perform on itself.

### 🔁 F. Tool Failures Are Cushioned; Brain-Call Failures Are Not (found re-running Query C after the `tzdata` fix)
*   **The Issue**: With `get_time` finally working, a fresh Query C run still crashed — this time on the *final* Decision call, with `httpx.HTTPStatusError: 502 Bad Gateway` from the gateway. One re-run got past it. The 502 was a transient upstream/provider error, not the agent's code.
*   **The Learning**: The loop's resilience (insight #15) only covers **tool/Action** failures — a failed `session.call_tool` leaves the goal open and the next iteration retries. But an **LLM-call** failure inside `perception.observe` / `decision.next_step` is a raw exception (`client.py` does `r.raise_for_status()`), and **nothing wraps it** — so a single transient gateway 502 takes down the entire run, even though the very next attempt would likely succeed.
*   **The Fix (future work)**: Wrap each role's gateway call in a bounded retry-with-backoff (the gateway itself fails over between providers, but the *caller* should also tolerate the gateway returning 5xx). Until then, a transient 502 means "just run it again."

### 🧬 G. Persistent Memory Is a Feature *and* a Contamination Hazard (found in the same re-run)
*   **The Issue**: Re-running Query C *on top of* leftover state from an earlier Query D ("asyncio") run produced a final answer about **Python asyncio best practices** — completely ignoring the birthday question — and looped to `MAX_ITERATIONS` without ever closing the goal.
*   **The mechanism (a self-reinforcing leak)**:
    1. `state/memory.json` persists across runs (by design — this is what makes Query C run 2 cheap).
    2. `memory.read` surfaced the stale asyncio items; `decision.py` injects each hit's `value` into the prompt, so Decision saw asyncio text against a vague "synthesize the final answer" goal and answered with asyncio content.
    3. That wrong answer was appended to **history** — and `memory.read` folds recent history tokens into its keyword query, so the *next* read matched **even more** asyncio items (hit count climbed 5 → 6 → 7). The contamination fed itself, and Perception never marked the goal done.
*   **The Learning**: The exact property that helps cross-run carryover (Query C run 2) *hurts* across **unrelated** queries. Same mechanism, two faces — the recurring theme of this assignment.
*   **The Fix**: Start unrelated queries from **clean state** (`del state/memory.json` + empty `state/artifacts/`). After clearing, Query C completed correctly in **4 / 8 iterations** (15 days away, Saturday — both independently verified). For production, memory items should be **run-scoped or namespaced** so a read can't pull facts from an unrelated task; durable cross-run carryover should be an explicit opt-in (e.g. `kind="fact"`/`"preference"` only), not the default for everything in the store.
