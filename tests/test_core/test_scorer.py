import importlib.util

import pytest

from deepeval.scorer import Scorer


def test_neural_bias_score_raises_clean_import_error_without_dbias():
    """
    Regression test for issue #382: before the fix, `Scorer.neural_bias_score`
    wrapped the `UnBiasedModel` import in a try/except that swallowed the
    real ImportError, so a missing Dbias install surfaced as an unrelated
    NameError instead of a clear install hint.
    """
    if importlib.util.find_spec("Dbias") is not None:
        pytest.skip("Dbias is installed in this environment")

    with pytest.raises(ImportError, match=r"pip install deepeval\[bias\]"):
        Scorer.neural_bias_score("some text")


def test_neural_toxic_score_raises_clean_import_error_without_detoxify():
    """
    Regression test for issue #382: same swallowed-ImportError bug as
    neural_bias_score, but for `Scorer.neural_toxic_score` / detoxify.
    """
    if importlib.util.find_spec("detoxify") is not None:
        pytest.skip("detoxify is installed in this environment")

    with pytest.raises(ImportError, match=r"pip install deepeval\[toxicity\]"):
        Scorer.neural_toxic_score("some text")
