import os
from pathlib import Path
from typing import List, Optional

from ..schema import CodeChunk
from .base import (
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_FILE_BYTES,
    SKIP_DIRS,
    _chunk_path,
    _passes_filters,
)


def _iter_files(root: Path):
    """
    Yield (absolute_path, posix_relative_path) for candidate files.
    """
    if root.is_file():
        yield root, root.name
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            directory for directory in dirnames if directory not in SKIP_DIRS
        ]
        for filename in filenames:
            abs_path = Path(dirpath) / filename
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            yield abs_path, rel_path


def collect_files(
    path: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> List[CodeChunk]:
    """Collect scannable code chunks from a file or directory tree.

    Args:
        path: A source file or a directory to walk.
        include: Optional glob allow-list (matched against the relative path). If given, a file must match at least one pattern.
        exclude: Optional glob deny-list (matched against the relative path).
        max_file_bytes: Skip files larger than this.
        max_chunk_chars: Split files into chunks no larger than this.
    """
    root = Path(path)
    chunks: List[CodeChunk] = []
    for abs_path, rel_path in _iter_files(root):
        if not _passes_filters(rel_path, include, exclude):
            continue
        chunks.extend(
            _chunk_path(abs_path, rel_path, max_file_bytes, max_chunk_chars)
        )
    return chunks
