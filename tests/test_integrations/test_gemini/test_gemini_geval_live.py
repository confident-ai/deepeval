import os
import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import assert_test
from deepeval.models import GeminiModel
from deepeval.key_handler import KEY_FILE_HANDLER, ModelKeyValues
from deepeval.metrics.g_eval.utils import Rubric


def _has_gemini_credentials():
    # Env-based
    env_ok = bool(
        os.getenv("GOOGLE_API_KEY")
        or (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            and os.getenv("GOOGLE_CLOUD_LOCATION")
        )
    )
    if env_ok:
        return True
    # Key-file based
    api_key = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.GOOGLE_API_KEY)
    project = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.GOOGLE_CLOUD_PROJECT)
    location = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.GOOGLE_CLOUD_LOCATION)
    return bool(api_key or (project and location))


@pytest.mark.skipif(
    not _has_gemini_credentials(),
    reason=(
        "Requires GOOGLE_API_KEY or (GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION) "
        "either in environment or configured via .deepeval/.deepeval"
    ),
)
def test_gemini_geval_live_strict_mode():
    """Test that the gemini model can evaluate a test case with strict mode, requiring a perfect score (1) to pass."""
    model = GeminiModel()

    metric = GEval(
        name="Validity",
        criteria="The response should directly answer the user question accurately.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=model,
        async_mode=False,
        strict_mode=True,
    )

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        expected_output="Paris",
    )

    assert_test(test_case=test_case, metrics=[metric], run_async=False)


@pytest.mark.skipif(
    not _has_gemini_credentials(),
    reason=(
        "Requires GOOGLE_API_KEY or (GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION) "
        "either in environment or configured via .deepeval/.deepeval"
    ),
)
def test_gemini_geval_live_custom_rubrics_0_to_5():
    """
    Test that the gemini model can evaluate a test case with custom rubrics from 0 to 5.

    Gemini-2.5-flash tokenizes the default rubric upper bound `10` to `0` and `1`,
    which will crash the current calculate_weighted_summed_score function.
    So we use a custom rubric from 0 to 5.
    """
    model = GeminiModel(model_name="gemini-2.5-flash")

    metric = GEval(
        name="Validity",
        criteria="The response should directly answer the user question accurately.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        rubric=[
            Rubric(
                score_range=(0, 1), expected_outcome="Irrelevant or incorrect."
            ),
            Rubric(
                score_range=(2, 3),
                expected_outcome="Partially correct or incomplete.",
            ),
            Rubric(
                score_range=(4, 5), expected_outcome="Mostly or fully correct."
            ),
        ],
        model=model,
        threshold=0.8,
        async_mode=False,
    )

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        expected_output="Paris",
    )

    assert_test(test_case=test_case, metrics=[metric], run_async=False)


##############################################
# Test Everything
##############################################

if __name__ == "__main__":
    test_gemini_geval_live_strict_mode()
    test_gemini_geval_live_custom_rubrics_0_to_5()
