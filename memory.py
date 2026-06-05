"""The Memory service: typed store of facts, preferences, outcomes, scratchpad.

Reads are cheap (pure keyword search, no LLM).
Writes for ambiguous content are expensive (one LLM call to classify).
Persists to ``state/memory.json`` so memory survives across runs.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field

from schemas import MemoryItem, ToolCall


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
    "and", "or", "but", "if", "then", "for", "as", "with", "at", "by",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "my", "your", "his", "her", "its", "our", "their", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "me", "us", "them", "what",
    "how", "when", "where", "why", "will", "would", "should", "could",
}


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens minus stopwords."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


class MemoryClassifierOutput(BaseModel):
    """Structured classification output for MemoryItem components."""
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str] = Field(default_factory=list)
    descriptor: str
    value: dict[str, Any] = Field(default_factory=dict)


MEMORY_CLASSIFICATION_SYSTEM_PROMPT = """
You are a memory classification system. Your job is to classify and structure free-form text input into a structured memory object.

You must output a JSON object matching this schema:
{
  "kind": "fact" | "preference" | "tool_outcome" | "scratchpad",
  "keywords": ["keyword1", "keyword2", ...],
  "descriptor": "One short human-readable summary line",
  "value": { ... structured key-value payload ... }
}

Guidelines for 'kind':
- 'fact': A durable truth or factual statement (e.g. "Mom's birthday is 15 May 2026").
- 'preference': A user preference or styling/behavioral choice (e.g. "User prefers detailed summaries").
- 'tool_outcome': Record of a tool execution (usually contains tool name, arguments, and result details).
- 'scratchpad': Run-scoped intermediate state or note that does not represent a durable fact or preference.

Guidelines for 'keywords':
- Provide a list of 3-10 lowercase token keywords that represent the core subjects, entities, dates, or topics in the input (e.g., ["mom", "birthday", "may", "2026"]).

Guidelines for 'descriptor':
- A brief (one line, under 120 characters) human-readable description of the item.

Guidelines for 'value':
- A structured dictionary containing key-value attributes extracted from the text (e.g., {"entity": "mom", "attribute": "birthday", "value": "2026-05-15"}).

CRITICAL SAFETY RULES:
1. DO NOT fabricate, hallucinate, mock, or assume any facts, search summaries, web page contents, or tool execution outcomes that are not explicitly stated in the input text.
2. If the input text is a user query/instruction requesting that an action be performed (e.g., "Search for...", "Read...", "Verify..."), it is NOT a 'tool_outcome' because the action has not occurred yet. Classify it as a 'scratchpad' or 'fact' detailing the user's request, and leave the 'value' payload empty (e.g., {}) or only containing the raw query text. Never generate fake results.
"""


class Memory:
    """Persistent typed memory. Lives in ``state/memory.json``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._items: list[MemoryItem] = []
        if self.path.exists():
            raw = self.path.read_text().strip() or "[]"
            self._items = [MemoryItem.model_validate(d) for d in json.loads(raw)]

    # ── reads ────────────────────────────────────────────────────────────

    def read(
        self,
        query: str,
        history: list[dict] | None = None,
        kinds: Optional[list[str]] = None,
        top_k: int = 8,
    ) -> list[MemoryItem]:
        """Pure keyword search. No LLM. Called at the top of every iteration."""
        q_tokens = _tokens(query)
        if history:
            for ev in history[-5:]:
                q_tokens |= _tokens(str(ev))
        candidates = (
            self._items if kinds is None
            else [it for it in self._items if it.kind in kinds]
        )
        scored: list[tuple[int, MemoryItem]] = []
        for item in candidates:
            item_tokens = {t.lower() for t in item.keywords} | _tokens(item.descriptor)
            score = len(q_tokens & item_tokens)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda s: -s[0])
        return [it for _, it in scored[:top_k]]

    def filter(
        self,
        kinds: Optional[list[str]] = None,
        goal_id: Optional[str] = None,
        recent: Optional[int] = None,
    ) -> list[MemoryItem]:
        """Structured filter — no LLM."""
        items = self._items
        if kinds is not None:
            items = [it for it in items if it.kind in kinds]
        if goal_id is not None:
            items = [it for it in items if it.goal_id == goal_id]
        if recent is not None:
            items = sorted(items, key=lambda it: it.created_at, reverse=True)[:recent]
        return items

    def relevant(
        self,
        query: str,
        kinds: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """LLM-scored relevance over a kind-filtered candidate pool."""
        candidates = (
            self._items if kinds is None
            else [it for it in self._items if it.kind in kinds]
        )
        if not candidates:
            return []

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "llm_gatewayV3"))
        from client import LLM

        candidate_list_str = ""
        for i, item in enumerate(candidates):
            candidate_list_str += f"[{i}] Kind: {item.kind}, Descriptor: {item.descriptor}\n"

        prompt = f"""You are a relevance scoring system. Given a query and a list of candidate memory items, score which candidates are relevant to the query.
Return the indices of the top {top_k} most relevant candidates, in descending order of relevance.

Query: {query}

Candidates:
{candidate_list_str}

Respond with a JSON object containing a list of integer indices:
{{
  "indices": [2, 0, 1]
}}
"""
        from pydantic import BaseModel
        class RelevanceOutput(BaseModel):
            indices: list[int]

        llm = LLM()
        try:
            reply = llm.chat(
                auto_route="memory",
                system="You are a relevance scoring helper.",
                prompt=prompt,
                response_format={
                    "type": "json_schema",
                    "schema": RelevanceOutput.model_json_schema(),
                    "name": "RelevanceOutput",
                    "strict": True,
                },
                temperature=0.0,
            )
            parsed = reply.get("parsed")
            if parsed:
                out = RelevanceOutput.model_validate(parsed)
                scored_items = []
                for idx in out.indices:
                    if 0 <= idx < len(candidates):
                        scored_items.append(candidates[idx])
                return scored_items[:top_k]
        except Exception:
            pass

        # Fallback keyword match
        q_tokens = _tokens(query)
        scored = []
        for item in candidates:
            item_tokens = {t.lower() for t in item.keywords} | _tokens(item.descriptor)
            score = len(q_tokens & item_tokens)
            scored.append((score, item))
        scored.sort(key=lambda s: -s[0])
        return [it for _, it in scored[:top_k]]

    # ── writes ───────────────────────────────────────────────────────────

    def remember(
        self,
        raw_text: str,
        source: str,
        run_id: str,
        goal_id: Optional[str] = None,
    ) -> MemoryItem:
        """Classify free-form text into a typed item."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "llm_gatewayV3"))
        from client import LLM

        llm = LLM()
        prompt_text = f"Source Context: {source}\nInput to classify:\n{raw_text}"
        reply = llm.chat(
            auto_route="memory",
            system=MEMORY_CLASSIFICATION_SYSTEM_PROMPT,
            prompt=prompt_text,
            response_format={
                "type": "json_schema",
                "schema": MemoryClassifierOutput.model_json_schema(),
                "name": "MemoryClassifierOutput",
                "strict": True,
            },
            temperature=0.0,
        )

        parsed = reply.get("parsed")
        if not parsed:
            kind = "scratchpad"
            keywords = list(_tokens(raw_text))[:10]
            descriptor = raw_text[:120]
            value = {"raw": raw_text}
        else:
            try:
                out = MemoryClassifierOutput.model_validate(parsed)
                kind = out.kind
                keywords = out.keywords
                descriptor = out.descriptor
                value = out.value
            except Exception:
                kind = "scratchpad"
                keywords = list(_tokens(raw_text))[:10]
                descriptor = raw_text[:120]
                value = {"raw": raw_text}

        item = MemoryItem(
            id=uuid.uuid4().hex[:8],
            kind=kind,
            keywords=keywords,
            descriptor=descriptor,
            value=value,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
        )
        self._items.append(item)
        self._save()
        return item

    def record_outcome(
        self,
        tool_call: ToolCall,
        result_text: str,
        artifact_id: Optional[str],
        run_id: str,
        goal_id: Optional[str] = None,
    ) -> MemoryItem:
        """Record one MCP dispatch outcome — no LLM call."""
        item = MemoryItem(
            id=uuid.uuid4().hex[:8],
            kind="tool_outcome",
            keywords=[tool_call.name] + list(_tokens(json.dumps(tool_call.arguments))),
            descriptor=(
                f"{tool_call.name}({json.dumps(tool_call.arguments)}) → "
                f"{result_text[:120]}"
            ),
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_descriptor": result_text[:300],
            },
            artifact_id=artifact_id,
            source=f"tool:{tool_call.name}",
            run_id=run_id,
            goal_id=goal_id,
        )
        self._items.append(item)
        self._save()
        return item

    # ── persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                [it.model_dump(mode="json") for it in self._items],
                indent=2,
            )
        )
