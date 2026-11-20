"""Regression tests for https://github.com/confident-ai/deepeval/issues/382.

The legacy classifier backends (`detoxify`, `Dbias`) are dated, heavy,
optional dependencies. Importing their wrapper modules must never explode,
and instantiating them without the packages must raise a clear
`ImportError` (never `UnboundLocalError`/`NameError`/`ModuleNotFoundError`).
"""

import importlib.util

import pytest


def _missing(name: str) -> bool:
    return importlib.util.find_spec(name) is None


def test_detoxify_module_imports_without_package():
    from deepeval.models import detoxify_model  # noqa: F401


def test_unbias_module_imports_without_package():
    from deepeval.models import unbias_model  # noqa: F401


@pytest.mark.skipif(not _missing("detoxify"), reason="detoxify is installed")
def test_detoxify_model_raises_helpful_error():
    from deepeval.models.detoxify_model import DetoxifyModel

    with pytest.raises(ImportError, match=r"pip install deepeval\[toxicity\]"):
        DetoxifyModel()


@pytest.mark.skipif(not _missing("Dbias"), reason="Dbias is installed")
def test_unbiased_model_raises_helpful_error():
    from deepeval.models.unbias_model import UnBiasedModel

    with pytest.raises(ImportError, match=r"pip install deepeval\[bias\]"):
        UnBiasedModel()
