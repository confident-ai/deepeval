from typing import Optional

from deepeval.models import DeepEvalBaseLLM


def should_use_batch(model: DeepEvalBaseLLM, batch_size: Optional[int] = None):
    if batch_size is None:
        return False

    # DeepEvalBaseLLM always defines batch_generate, but its default
    # implementation just raises NotImplementedError. Only use batching when
    # the model actually overrides it, otherwise fall back to per-sample
    # generation as documented.
    if (
        getattr(type(model), "batch_generate", None)
        is DeepEvalBaseLLM.batch_generate
    ):
        return False

    return hasattr(model, "batch_generate")
