"""Annotate retrieval context lists with node IDs for multimodal metric prompts."""

from __future__ import annotations

from typing import Any, List, Union

from deepeval.test_case import MLLMImage
from deepeval.test_case.llm_test_case import RetrievedContextData
from deepeval.utils import convert_to_multi_modal_array


def id_retrieval_context(
    retrieval_context: List[Any],
) -> List[Union[str, MLLMImage]]:
    """Prefix each text node with ``Node N:``; image nodes get a label line then the image."""
    annotated: List[Union[str, MLLMImage]] = []
    seq = convert_to_multi_modal_array(retrieval_context)
    for i, context in enumerate(seq):
        if isinstance(context, str):
            annotated.append(f"Node {i + 1}: {context}")
        elif isinstance(context, MLLMImage):
            annotated.append(f"Node {i + 1}:")
            annotated.append(context)
    return annotated


def retrieval_context_to_strings(
    retrieval_context: List[Union[str, RetrievedContextData]],
) -> List[str]:
    """Return ``retrieval_context`` with each ``RetrievedContextData`` item
    rendered through its own serializer (``"source: context"``), so metrics
    that join or split the context as text don't crash on structured items."""
    return [
        c.model_dump() if isinstance(c, RetrievedContextData) else c
        for c in retrieval_context
    ]
