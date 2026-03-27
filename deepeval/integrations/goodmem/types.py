from dataclasses import dataclass
from typing import Optional


@dataclass
class GoodMemChunk:
    """A single retrieved chunk from GoodMem with metadata."""

    content: str
    score: Optional[float] = None
    chunk_id: str = ""
    memory_id: str = ""
    space_id: str = ""
