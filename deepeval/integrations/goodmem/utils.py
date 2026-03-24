import json
from typing import Any, Dict, List, Optional

import requests

from deepeval.integrations.goodmem.types import GoodMemChunk


def goodmem_retrieve(
    base_url: str,
    api_key: str,
    space_ids: List[str],
    query: str,
    top_k: int = 5,
    reranker: Optional[str] = None,
    relevance_threshold: Optional[float] = None,
    metadata_filter: Optional[str] = None,
) -> List[GoodMemChunk]:
    """Execute a semantic retrieval against GoodMem via raw HTTP.

    Returns a list of ``GoodMemChunk`` objects with content, scores, and IDs.
    """

    url = f"{base_url.rstrip('/')}/v1/memories:retrieve"

    space_keys: List[Dict[str, Any]] = []
    for sid in space_ids:
        key: Dict[str, Any] = {"spaceId": sid}
        if metadata_filter:
            key["filter"] = metadata_filter
        space_keys.append(key)

    body: Dict[str, Any] = {
        "message": query,
        "spaceKeys": space_keys,
        "requestedSize": top_k,
        "fetchMemory": True,
    }

    if reranker:
        body["postProcessor"] = {
            "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
            "config": {"reranker_id": reranker},
        }

    if relevance_threshold is not None:
        body.setdefault("postProcessor", {}).setdefault("config", {})[
            "relevance_threshold"
        ] = relevance_threshold

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/x-ndjson",
        "x-api-key": api_key,
    }

    response = requests.post(url, json=body, headers=headers, timeout=30)
    response.raise_for_status()

    return _parse_ndjson_response(response.text)


def _parse_ndjson_response(text: str) -> List[GoodMemChunk]:
    """Parse GoodMem's NDJSON streaming response into GoodMemChunk objects."""
    chunks: List[GoodMemChunk] = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "retrievedItem" in event:
            item = event["retrievedItem"]
            chunk_data = item.get("chunk", {})

            # Handle nested chunk structure
            inner_chunk = chunk_data.get("chunk", chunk_data)
            chunk_text = inner_chunk.get("chunkText", "")
            chunk_id = inner_chunk.get("chunkId", "")
            memory_id = inner_chunk.get("memoryId", "")
            space_id = inner_chunk.get("spaceId", "")

            chunks.append(
                GoodMemChunk(
                    content=chunk_text,
                    score=chunk_data.get(
                        "relevanceScore", item.get("relevanceScore")
                    ),
                    chunk_id=chunk_id,
                    memory_id=memory_id,
                    space_id=space_id,
                )
            )

    return chunks
