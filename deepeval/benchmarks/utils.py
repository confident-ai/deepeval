from typing import Optional

from deepeval.models import DeepEvalBaseLLM


def validate_n_problems(n_problems: int, maximum: int, benchmark: str) -> int:
    """Validate a benchmark's ``n_problems`` against its full lower/upper range.

    ``n_problems`` reaches the evaluation loop twice: it slices the goldens
    (``goldens[: n_problems]``) and it is the denominator of the reported
    accuracy. A non-positive value is accepted by an upper-bound-only check but
    corrupts both, so reject it here rather than let it through:

    - ``0`` slices to an empty list and makes the accuracy a ZeroDivisionError.
    - a negative value slices *from the end* -- ``n_problems=-5`` runs all but
      the last 5 problems, so nearly the whole benchmark -- and then divides by
      the negative value, reporting a negative accuracy.
    """
    if not isinstance(n_problems, int) or isinstance(n_problems, bool):
        raise TypeError(
            f"{benchmark} n_problems must be an int, got "
            f"{type(n_problems).__name__}."
        )
    if n_problems < 1:
        raise ValueError(
            f"{benchmark} n_problems must be >= 1, got {n_problems}."
        )
    if n_problems > maximum:
        raise ValueError(
            f"{benchmark} only supports n_problems <= {maximum}, got "
            f"{n_problems}."
        )
    return n_problems


def validate_n_shots(n_shots: int, maximum: int, benchmark: str) -> int:
    """Validate a benchmark's ``n_shots`` against its full lower/upper range.

    ``n_shots`` selects how many few-shot examples are prepended to each
    prompt. A negative value is accepted by an upper-bound-only check and then
    silently slices the example pool from the end, so the prompt is built from
    a different number of examples than requested and the run is not the
    n-shot configuration it is reported as.
    """
    if not isinstance(n_shots, int) or isinstance(n_shots, bool):
        raise TypeError(
            f"{benchmark} n_shots must be an int, got "
            f"{type(n_shots).__name__}."
        )
    if n_shots < 0:
        raise ValueError(f"{benchmark} n_shots must be >= 0, got {n_shots}.")
    if n_shots > maximum:
        raise ValueError(
            f"{benchmark} only supports n_shots <= {maximum}, got {n_shots}."
        )
    return n_shots


def should_use_batch(model: DeepEvalBaseLLM, batch_size: Optional[int] = None):
    if batch_size is None:
        return False

    if not hasattr(model, "batch_generate"):
        return False

    return True
