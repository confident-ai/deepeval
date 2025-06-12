from deepeval.test_case import MLLMTestCaseParams, MLLMTestCase, ToolCall
from deepeval.test_case.mllm_test_case import MLLMImage
from deepeval.metrics.g_eval.utils import (
    unsupported_log_probs_gpt_models,
)
from deepeval.models import (
    DeepEvalBaseMLLM,
    MultimodalOpenAIModel,
)

from typing import List, Union


G_EVAL_PARAMS = {
    MLLMTestCaseParams.INPUT: "Input",
    MLLMTestCaseParams.ACTUAL_OUTPUT: "Actual Output",
    MLLMTestCaseParams.EXPECTED_OUTPUT: "Expected Output",
    MLLMTestCaseParams.CONTEXT: "Context",
    MLLMTestCaseParams.RETRIEVAL_CONTEXT: "Retrieval Context",
    MLLMTestCaseParams.EXPECTED_TOOLS: "Expected Tools",
    MLLMTestCaseParams.TOOLS_CALLED: "Tools Called",
}


def construct_g_eval_params_string(
    mllm_test_case_params: List[MLLMTestCaseParams],
):
    g_eval_params = [G_EVAL_PARAMS[param] for param in mllm_test_case_params]
    if len(g_eval_params) == 1:
        g_eval_params_str = g_eval_params[0]
    elif len(g_eval_params) == 2:
        g_eval_params_str = " and ".join(g_eval_params)
    else:
        g_eval_params_str = (
            ", ".join(g_eval_params[:-1]) + ", and " + g_eval_params[-1]
        )

    return g_eval_params_str


def construct_test_case_list(
    evaluation_params: List[MLLMTestCaseParams], test_case: MLLMTestCase
) -> List[Union[str, MLLMImage]]:
    test_case_list = []
    for param in evaluation_params:
        test_case_param_list = [f"\n\n\n{G_EVAL_PARAMS[param]}:\n"]
        value = getattr(test_case, param.value)
        for v in value:
            if isinstance(v, ToolCall):
                test_case_param_list.append(repr(v))
            else:
                test_case_param_list.append(v)
        test_case_list.extend(test_case_param_list)
    return test_case_list


def no_multimodal_log_prob_support(model: Union[str, DeepEvalBaseMLLM]):
    if isinstance(model, str) and model in unsupported_log_probs_gpt_models:
        return True
    elif (
        isinstance(model, MultimodalOpenAIModel)
        and model.model_name in unsupported_log_probs_gpt_models
    ):
        return True
    return False
