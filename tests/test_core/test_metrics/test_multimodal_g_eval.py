import os
import pytest

from deepeval import assert_test
from deepeval.test_case import MLLMTestCase, MLLMTestCaseParams, MLLMImage
from deepeval.metrics.multimodal_metrics.multimodal_g_eval.multimodal_g_eval import (
    MultimodalGEval,
)
from deepeval.metrics.multimodal_metrics.multimodal_g_eval.template import (
    MultimodalGEvalTemplate,
)


BANANA = "./images/Banana.jpg"


class MyMMTemplate(MultimodalGEvalTemplate):
    """Custom template used to validate that MultimodalGEval honors evaluation_template."""

    called_steps = 0
    called_eval = 0
    called_strict = 0
    MARK = "[MyMMTemplate]"

    @staticmethod
    def generate_evaluation_steps(*, criteria, parameters):
        # return a string prompt
        MyMMTemplate.called_steps += 1
        base = MultimodalGEvalTemplate.generate_evaluation_steps(
            criteria=criteria, parameters=parameters
        )
        return f"{MyMMTemplate.MARK}\n{base}"

    @staticmethod
    def generate_evaluation_results(
        *,
        evaluation_steps,
        test_case_list,
        parameters,
        rubric,
        score_range,
        _additional_context,
    ):
        # return a list of messages
        MyMMTemplate.called_eval += 1
        base = MultimodalGEvalTemplate.generate_evaluation_results(
            evaluation_steps=evaluation_steps,
            test_case_list=test_case_list,
            parameters=parameters,
            rubric=rubric,
            score_range=score_range,
            _additional_context=_additional_context,
        )
        return [{"role": "system", "content": MyMMTemplate.MARK}] + list(base)

    @staticmethod
    def generate_strict_evaluation_results(
        *, evaluation_steps, test_case_list, parameters, _additional_context
    ):
        # return a list of messages
        MyMMTemplate.called_strict += 1
        base = MultimodalGEvalTemplate.generate_strict_evaluation_results(
            evaluation_steps=evaluation_steps,
            test_case_list=test_case_list,
            parameters=parameters,
            _additional_context=_additional_context,
        )
        return [{"role": "system", "content": MyMMTemplate.MARK}] + list(base)


# import deepeval.openai
@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="needs OPENAI_API_KEY",
)
@pytest.mark.parametrize("strict_mode", [False, True])
def test_multimodal_geval_uses_custom_evaluation_template(strict_mode):
    """
    Verifies that MultimodalGEval calls a provided custom evaluation_template.
    Passing means our subclass methods were invoked at least once.
    """
    metric = MultimodalGEval(
        name="MM Correctness (custom template)",
        criteria="Decide if the actual output correctly answers the question given the image.",
        evaluation_params=[
            MLLMTestCaseParams.ACTUAL_OUTPUT,
            MLLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        verbose_mode=True,
        async_mode=True,
        strict_mode=strict_mode,
        model="gpt-4o-mini",
        evaluation_template=MyMMTemplate,
    )

    tc = MLLMTestCase(
        input=["What fruit is shown in the image?", MLLMImage(url=BANANA)],
        actual_output=["A banana."],
        expected_output=["A banana."],
    )

    assert_test(tc, [metric])

    assert MyMMTemplate.called_steps >= 1
    # One of these two will be called depending on strict_mode (default False -> non-strict path)
    assert (MyMMTemplate.called_eval >= 1) or (MyMMTemplate.called_strict >= 1)
