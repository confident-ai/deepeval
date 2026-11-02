"""Shared helpers for the consistency community metrics.

The pair sampling, similarity primitives and rubric come from the ``repeatability-evals``
package (``pip install 'repeatability-evals>=0.3.0'``), which is imported lazily so that it is only
needed when one of these metrics is used.
"""

from typing import Any, List, Optional, Tuple

from deepeval.errors import MissingTestCaseParamsError

_INSTALL_HINT = "Install it with `pip install 'repeatability-evals>=0.3.0'`."


def require_repeatability_evals():
    """Import ``repeatability_evals`` lazily, with an actionable error if it is missing."""
    try:
        import repeatability_evals
    except ImportError as e:  # pragma: no cover - only without the package
        raise ImportError(
            f"This metric needs the `repeatability-evals` package. {_INSTALL_HINT}"
        ) from e
    if not hasattr(repeatability_evals, "render_prompt"):  # pragma: no cover
        raise ImportError(
            f"This metric needs a newer `repeatability-evals`. {_INSTALL_HINT}"
        )
    return repeatability_evals


def sample_pairs(
    n_runs: int, max_comparisons: Optional[int], seed: int = 0
) -> List[Tuple[int, int]]:
    """Which pairs of runs to compare: a random cycle by default (N pairs for N runs, every run
    compared at least twice), or ``max_comparisons`` pairs. ``None`` budget means the cycle.
    """
    return require_repeatability_evals().cycle_pairs(
        n_runs, budget=max_comparisons, seed=seed
    )


def get_runs(metadata: Optional[dict], key: str, minimum: int = 2) -> List[Any]:
    runs = (metadata or {}).get(key)
    if not isinstance(runs, (list, tuple)) or len(runs) < minimum:
        raise MissingTestCaseParamsError(
            f"Consistency needs at least {minimum} runs of the same input. Pass them in "
            f"`LLMTestCase(metadata={{'{key}': [...]}})`."
        )
    return list(runs)


def describe_pair(i: int, j: int) -> str:
    return f"runs {i + 1} and {j + 1}"
