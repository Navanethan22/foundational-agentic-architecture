"""Content-addressable store for raw bytes. Parallel to Memory.

Memory holds the address (a ``MemoryItem.artifact_id`` like ``art:09ff...``).
ArtifactStore holds the actual bytes on disk under ``state/artifacts/``.
"""
import hashlib
from pathlib import Path

from schemas import Artifact


class ArtifactStore:
    """SHA-256 keyed file store. Two files per artifact: ``<digest>.bin``
    (raw bytes) and ``<digest>.json`` (metadata)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, blob: bytes, *, content_type: str, source: str,
            descriptor: str) -> str:
        digest = hashlib.sha256(blob).hexdigest()[:16]
        artifact_id = f"art:{digest}"
        (self.root / f"{digest}.bin").write_bytes(blob)
        meta = Artifact(
            id=artifact_id,
            content_type=content_type,
            size_bytes=len(blob),
            source=source,
            descriptor=descriptor,
        )
        (self.root / f"{digest}.json").write_text(meta.model_dump_json(indent=2))
        return artifact_id

    def get_bytes(self, artifact_id: str) -> bytes:
        digest = artifact_id.removeprefix("art:")
        return (self.root / f"{digest}.bin").read_bytes()

    def get_meta(self, artifact_id: str) -> Artifact:
        digest = artifact_id.removeprefix("art:")
        return Artifact.model_validate_json(
            (self.root / f"{digest}.json").read_text()
        )

    def exists(self, artifact_id: str) -> bool:
        digest = artifact_id.removeprefix("art:")
        return (self.root / f"{digest}.bin").exists()
