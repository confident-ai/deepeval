import subprocess
from pathlib import Path
from typing import List, Optional

from ..schema import CodeChunk
from .base import (
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_FILE_BYTES,
    _chunk_path,
    _passes_filters,
)


def _git_changed_files(root: Path, base: str, head: Optional[str]) -> List[str]:
    ref = f"{base}..{head}" if head else base
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", ref],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"git diff --name-only {ref} failed: {e}") from e
    return [line.strip() for line in out.splitlines() if line.strip()]


def collect_changed_files(
    path: str,
    base: str,
    head: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> List[CodeChunk]:
    """Collect chunks for only the files changed between two git refs.

    Scans whole changed files (not line-level hunks), coarse but reliable for
    PR/commit scanning. `path` should be the repository root. If `head` is
    omitted, compares `base` against the working tree.
    """
    root = Path(path)
    chunks: List[CodeChunk] = []
    for rel_path in _git_changed_files(root, base, head):
        abs_path = root / rel_path
        if not abs_path.is_file():
            continue  # deleted or renamed away
        if not _passes_filters(rel_path, include, exclude):
            continue
        chunks.extend(
            _chunk_path(abs_path, rel_path, max_file_bytes, max_chunk_chars)
        )
    return chunks
