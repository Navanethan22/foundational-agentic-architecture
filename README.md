# EAG V3 — Session 6: Agentic Architecture

A goal-driven agent built from **four cognitive roles** — Memory, Perception,
Decision, Action — communicating through **typed Pydantic contracts** at every
boundary, backed by two support stores (Artifacts, the LLM Gateway V3) and a
9-tool MCP server.

The thesis of S6: a single LLM brain that must *decompose the goal, pick the
next action, and judge done-ness all at once* hits a ceiling on multi-goal
queries. S6 splits that one overloaded call into narrow roles, each with a typed
input, a typed output, and exactly one job.

## What's in this repo

**My work** — the agent itself: `schemas.py`, `memory.py`, `perception.py`,
`decision.py`, `action.py`, `artifacts.py`, and the `agent6.py` loop.

**Two external dependencies, not included here:**
- **`llm_gatewayV3/`** — a multi-provider LLM gateway provided by the course.
  Runs as a separate service on port 8101 (see below). Not redistributed in this
  repo; it is not my code.
- **`mcp_server.py`** — the 9-tool MCP server *is* included for runnability, but
  it too is course-provided scaffolding, not part of my contribution.

## Architecture

| File | Role | LLM calls | One job |
| :--- | :--- | :---: | :--- |
| `schemas.py` | Contracts | — | One source of truth: `MemoryItem`, `Artifact`, `Goal`, `Observation`, `ToolCall`, `DecisionOutput`. |
| `memory.py` | **Memory** | reads 0 / writes 1 | Typed store. `read()` is pure keyword search (no LLM); `remember()` pays one classification call once, then reads are free forever. Persists to `state/memory.json`. |
| `perception.py` | **Perception** | 1 / iter | The orchestrator. Decomposes the query into goals, re-judges `done` flags every turn, attaches artifacts. Pinned to Gemini (`provider="g"`) for reliability. |
| `decision.py` | **Decision** | 1 when a goal is open | Picks the next action for **one** goal. Returns *either* an answer *or* one tool call, never both. Blinded to other goals. |
| `action.py` | **Action** | 0 | Pure MCP dispatch. Spills results >4 KB to Artifacts, guards against `art:` handles, returns a descriptor. |
| `artifacts.py` | Artifacts | 0 | Content-addressable (SHA-256) blob store under `state/artifacts/`. Memory holds the address; Artifacts holds the bytes. |
| `agent6.py` | The loop | — | Wires the roles together in the fixed 5-step rhythm. |

**The iteration rhythm** (same whether the query has 1 goal or 10):

```
memory.read ─▶ perception.observe ─▶ all_done? ──yes──▶ exit
                                        │no
                                        ▼
                              decision.next_step
                                  │          │
                              answer?     tool_call?
                                  │          ▼
                                  │   action.execute ─▶ memory.record_outcome
                                  └──────────┴──▶ append history ─▶ loop
```

## Setup

```bash
cd eagv3_assignment
uv sync                              # installs deps incl. tzdata (Windows get_time)
cp .env.example .env                 # then fill in your own API keys
```

`.env` needs at minimum `GEMINI_API_KEY` (Perception is pinned to Gemini) plus
any of `CEREBRAS_API_KEY` / `GROQ_API_KEY` / `NVIDIA_API_KEY` / `GITHUB_ACCESS_TOKEN`
for the router pool, and `TAVILY_API_KEY` for web search (DuckDuckGo is the
no-key fallback).

## Running a query

**1. Start the gateway** in its own terminal (listens on port 8101). The gateway
is **not bundled in this repo** (see "What's in this repo") — obtain the EAG V3
Gateway V3, drop it in as `llm_gatewayV3/`, give it its own `.env`, then:

```bash
cd llm_gatewayV3 && uv run uvicorn main:app --port 8101
```

The agent only needs *something* answering the gateway contract on
`http://localhost:8101/v1/chat` (override with `LLM_GATEWAY_V3_URL`).

**2. Run the agent** in another terminal. `agent6.py` spawns the MCP server
itself over stdio — you do **not** start `mcp_server.py` separately:

```bash
uv run python agent6.py "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."
```

The loop prints each iteration: memory hits, Perception's goal list with
`done`/`open` flags, Decision's answer or tool call, and Action's result.

## Verification — the four target queries

All four queries finished **well within** their iteration budgets
(the rubric allows up to 2× the documented count). Full write-up of the runs and
the engineering discoveries behind them is in
[`session_6_learnings.md`](session_6_learnings.md).

| Query | Tests | Budget | Actual | Status |
| :--- | :--- | :---: | :---: | :--- |
| **A** — Shannon Wikipedia | Artifact attach (fetch → extract) | 6 | **2** | 🟢 Success |
| **B** — Tokyo activities + weather | Multi-goal, results merge | 12 | **4** | 🟢 Success |
| **C** — Mom's birthday (run 1) | Durable write + reminder math | 8 | **7 → 4** | 🟢 Fixed: `get_time` failed on Windows (no IANA tz db); after adding `tzdata` a clean re-run finished in 4 iters with correct date math |
| **C** — Mom's birthday (run 2) | Cross-run memory carryover | 4 | **2** | 🟢 Answered from memory, no tools |
| **D** — Asyncio research | Multi-source synthesis | 14 | **3** | 🟢 Success |

## Design decisions worth calling out

- **Exactly-one in `DecisionOutput`** — a `model_validator` rejects any output
  that sets both `answer` and `tool_call`, or neither. The contract enforces the
  "answer XOR act" rule so the loop never has to.
- **Positional artifact indices** — Perception sees `[ATTACHMENT INDEX: 0]`
  rather than raw `art:57d3d76…` handles, and emits the integer; code maps it
  back. LLMs handle small indices far more reliably than long hashes.
- **`art:` guard in Action** — refuses any tool argument starting with `art:`,
  so a hallucinating Decision can't feed an internal handle to `fetch_url` as if
  it were a URL.
- **Direct Gemini routing on attachment** — when Decision receives artifact
  bytes, it bypasses the token router (`provider="g"`) to avoid 503s on large
  contexts.

## State

- `state/memory.json` — persistent typed memory (this is what makes Query C
  run 2 cheap).
- `state/artifacts/<digest>.bin` + `.json` — spilled blobs and their metadata.

Both are gitignored; delete `state/` to start fresh.
