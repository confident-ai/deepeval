import json
from typing import Any, Dict, List, Optional

import requests


def goodmem_retrieve(
    base_url: str,
    api_key: str,
    space_id: str,
    query: str,
    top_k: int = 5,
    reranker: Optional[str] = None,
    relevance_threshold: Optional[float] = None,
    metadata_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a semantic retrieval against GoodMem via raw HTTP.

    Returns the parsed response dict with retrieved chunks.
    """

    url = f"{base_url.rstrip('/')}/v1/memories:retrieve"

    space_key: Dict[str, Any] = {"spaceId": space_id}
    if metadata_filter:
        space_key["filter"] = metadata_filter

    body: Dict[str, Any] = {
        "message": query,
        "spaceKeys": [space_key],
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


def _parse_ndjson_response(text: str) -> Dict[str, Any]:
    """Parse GoodMem's NDJSON streaming response into a structured dict."""
    chunks: List[Dict[str, Any]] = []

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

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "memory_id": memory_id,
                    "content": chunk_text,
                    "relevance_score": chunk_data.get(
                        "relevanceScore", item.get("relevanceScore")
                    ),
                }
            )

    return {"chunks": chunks}


def parse_chunks_to_texts(response: Dict[str, Any]) -> List[str]:
    """Extract plain text strings from a parsed retrieval response."""
    return [
        chunk["content"]
        for chunk in response.get("chunks", [])
        if chunk.get("content")
    ]
