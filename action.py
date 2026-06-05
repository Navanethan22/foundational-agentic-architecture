"""The Action role: dispatch a tool call, push large results to Artifacts.

Action contains NO LLM call. ~30 lines of pure dispatch.
"""
from artifacts import ArtifactStore
from schemas import ToolCall

ARTIFACT_THRESHOLD_BYTES = 4096
PREVIEW_LEN = 200


async def execute(session, tool_call: ToolCall,
                  artifacts: ArtifactStore) -> tuple[str, str | None]:
    """Run one tool call via MCP. Returns ``(descriptor, optional artifact_id)``.

    Three behaviours:
      1. Refuse args that begin with ``art:`` — those are internal handles,
         not real paths or URLs. Decision-side hallucination guard.
      2. Push results larger than ``ARTIFACT_THRESHOLD_BYTES`` into Artifacts;
         return a short descriptor that names the handle.
      3. Otherwise return the raw text as the descriptor.
    """
    # 1. Artifact-handle guard
    for v in tool_call.arguments.values():
        if isinstance(v, str) and v.startswith("art:"):
            return (
                f"ERROR: argument starts with 'art:' — that is an internal "
                f"artifact handle, not a real path or URL: {v!r}",
                None,
            )

    # 2. MCP dispatch
    result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)
    text = "".join(
        getattr(block, "text", "") for block in (result.content or [])
    )

    # 3. Threshold check
    if len(text.encode("utf-8")) > ARTIFACT_THRESHOLD_BYTES:
        preview = text[:PREVIEW_LEN].replace("\n", " ")
        artifact_id = artifacts.put(
            text.encode("utf-8"),
            content_type="text/plain",
            source=tool_call.name,
            descriptor=preview,
        )
        descriptor = (
            f"[artifact {artifact_id}, {len(text)} bytes] preview: {preview}"
        )
        return descriptor, artifact_id

    return text, None
