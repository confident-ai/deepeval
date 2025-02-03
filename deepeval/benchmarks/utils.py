from typing import Optional

from deepeval.models import DeepEvalBaseLLM


def should_use_batch(model: DeepEvalBaseLLM, batch_size: Optional[int] = None):
    if batch_size is None:
        return False

    if not hasattr(model, "batch_generate"):
        return False

    return True
