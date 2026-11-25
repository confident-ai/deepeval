import fnmatch
from pathlib import Path
from typing import List, Optional

from ..schema import CodeChunk

DEFAULT_MAX_FILE_BYTES = 1_500_000
DEFAULT_MAX_CHUNK_CHARS = 12_000

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".scala": "scala",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".tf": "terraform",
    ".yaml": "yaml",
    ".yml": "yaml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "bower_components",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    "target",
    "vendor",
    ".terraform",
    "coverage",
    ".turbo",
    ".cache",
    ".idea",
    ".vscode",
}

SKIP_FILE_GLOBS = [
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "go.sum",
]


def _matches_any(value: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(value, pat) for pat in patterns)


def _language_for(path: Path) -> Optional[str]:
    return LANGUAGE_BY_EXT.get(path.suffix.lower())


def _read_text(path: Path, max_file_bytes: int) -> Optional[str]:
    """Return file text, or None if it's too large, unreadable, or binary."""
    try:
        if path.stat().st_size > max_file_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None  # binary
    return data.decode("utf-8", errors="replace")


def _chunk_text(
    rel_path: str, text: str, language: Optional[str], max_chunk_chars: int
) -> List[CodeChunk]:
    """Split a file into line-aligned chunks, tracking absolute start lines."""
    lines = text.splitlines(keepends=True)
    chunks: List[CodeChunk] = []
    buf: List[str] = []
    buf_len = 0
    start_line = 1

    for line_no, line in enumerate(lines, start=1):
        if buf and buf_len + len(line) > max_chunk_chars:
            chunks.append(
                CodeChunk(
                    file_path=rel_path,
                    content="".join(buf),
                    language=language,
                    start_line=start_line,
                )
            )
            buf = []
            buf_len = 0
            start_line = line_no
        buf.append(line)
        buf_len += len(line)

    if buf:
        chunks.append(
            CodeChunk(
                file_path=rel_path,
                content="".join(buf),
                language=language,
                start_line=start_line,
            )
        )
    return chunks


def _passes_filters(
    rel_path: str,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
) -> bool:
    name = rel_path.rsplit("/", 1)[-1]
    if _matches_any(name, SKIP_FILE_GLOBS):
        return False
    if include and not _matches_any(rel_path, include):
        return False
    if exclude and _matches_any(rel_path, exclude):
        return False
    return True


def _chunk_path(
    abs_path: Path,
    rel_path: str,
    max_file_bytes: int,
    max_chunk_chars: int,
) -> List[CodeChunk]:
    language = _language_for(abs_path)
    if language is None:
        return []
    text = _read_text(abs_path, max_file_bytes)
    if text is None:
        return []
    return _chunk_text(rel_path, text, language, max_chunk_chars)
