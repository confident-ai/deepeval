from typing import List, Optional, Union

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
    BaseArenaMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ConversationalTestCase,
    MLLMImage,
    ArenaTestCase,
    MultiTurnParams,
)
from deepeval.utils import convert_to_multi_modal_array

from .models import MULTIMODAL_SUPPORTED_MODELS


def _reject_multimodal_without_llm(metric, model: Optional[DeepEvalBaseLLM]):
    """A metric has no LLM only when a System One model (Jev) is judging it,
    and Jev reads text only."""
    if model is None:
        name = getattr(metric, "__name__", type(metric).__name__)
        raise ValueError(
            f"{name} is judged by System One (Jev) in this eval mode, and Jev "
            f"evaluates text only. Run multimodal test cases with "
            f'`eval_mode="llm"` or `eval_mode="hybrid"`.'
        )


def check_conversational_test_case_params(
    test_case: ConversationalTestCase,
    test_case_params: List[MultiTurnParams],
    metric: BaseConversationalMetric,
    require_chatbot_role: bool = False,
    model: Optional[DeepEvalBaseLLM] = None,
    multimodal: Optional[bool] = False,
):
    if multimodal:
        _reject_multimodal_without_llm(metric, model)
        if not model or not model.supports_multimodal():
            if model and type(model) in MULTIMODAL_SUPPORTED_MODELS.keys():
                valid_multimodal_models = []
                for model_name, model_data in MULTIMODAL_SUPPORTED_MODELS.get(
                    type(model)
                ).items():
                    if callable(model_data):
                        model_data = model_data()
                    if model_data.supports_multimodal:
                        valid_multimodal_models.append(model_name)
                raise ValueError(
                    f"The evaluation model {model.name} does not support multimodal evaluations at the moment. Available multi-modal models for the {model.__class__.__name__} provider includes {', '.join(valid_multimodal_models)}."
                )
            else:
                raise ValueError(
                    f"The evaluation model {model.name} does not support multimodal inputs, please use one of the following evaluation models: {', '.join([cls.__name__ for cls in MULTIMODAL_SUPPORTED_MODELS.keys()])}"
                )

    if isinstance(test_case, ConversationalTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'ConversationalTestCase' using the conversational '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

    if (
        MultiTurnParams.EXPECTED_OUTCOME in test_case_params
        and test_case.expected_outcome is None
    ):
        error_str = f"'expected_outcome' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if (
        MultiTurnParams.SCENARIO in test_case_params
        and test_case.scenario is None
    ):
        error_str = f"'scenario' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if (
        MultiTurnParams.METADATA in test_case_params
        and test_case.metadata is None
    ):
        error_str = f"'metadata' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if MultiTurnParams.TAGS in test_case_params and test_case.tags is None:
        error_str = f"'tags' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if require_chatbot_role and test_case.chatbot_role is None:
        error_str = f"'chatbot_role' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if len(test_case.turns) == 0:
        error_str = "'turns' in conversational test case cannot be empty."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)


def check_llm_test_case_params(
    test_case: LLMTestCase,
    test_case_params: List[SingleTurnParams],
    input_image_count: Optional[int],
    actual_output_image_count: Optional[int],
    metric: Union[BaseMetric, BaseArenaMetric],
    model: Optional[DeepEvalBaseLLM] = None,
    multimodal: Optional[bool] = False,
):
    if multimodal:
        _reject_multimodal_without_llm(metric, model)
        if not model or not model.supports_multimodal():
            if model and type(model) in MULTIMODAL_SUPPORTED_MODELS.keys():
                valid_multimodal_models = []
                for model_name, model_data in MULTIMODAL_SUPPORTED_MODELS.get(
                    type(model)
                ).items():
                    if callable(model_data):
                        model_data = model_data()
                    if model_data.supports_multimodal:
                        valid_multimodal_models.append(model_name)
                raise ValueError(
                    f"The evaluation model {model.name} does not support multimodal evaluations at the moment. Available multi-modal models for the {model.__class__.__name__} provider includes {', '.join(valid_multimodal_models)}."
                )
            else:
                raise ValueError(
                    f"The evaluation model {model.name} does not support multimodal inputs, please use one of the following evaluation models: {', '.join([cls.__name__ for cls in MULTIMODAL_SUPPORTED_MODELS.keys()])}"
                )

        if input_image_count:
            count = 0
            for ele in convert_to_multi_modal_array(test_case.input):
                if isinstance(ele, MLLMImage):
                    count += 1
            if count != input_image_count:
                error_str = f"Can only evaluate test cases with '{input_image_count}' input images using the '{metric.__name__}' metric. `{count}` found."
                raise ValueError(error_str)

        if actual_output_image_count:
            count = 0
            for ele in convert_to_multi_modal_array(test_case.actual_output):
                if isinstance(ele, MLLMImage):
                    count += 1
            if count != actual_output_image_count:
                error_str = f"Can only evaluate test cases with '{actual_output_image_count}' output images using the '{metric.__name__}' metric. `{count}` found."
                raise ValueError(error_str)

    if isinstance(test_case, LLMTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'LLMTestCase' using the non-conversational '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

    # Centralized: if a metric requires actual_output, reject empty/whitespace
    # (including empty multimodal outputs) as "missing params".
    if SingleTurnParams.ACTUAL_OUTPUT in test_case_params:
        actual_output = getattr(test_case, SingleTurnParams.ACTUAL_OUTPUT.value)
        if isinstance(actual_output, str) and actual_output == "":
            error_str = f"'actual_output' cannot be empty for the '{metric.__name__}' metric"
            metric.error = error_str
            raise MissingTestCaseParamsError(error_str)

    missing_params = []
    for param in test_case_params:
        if getattr(test_case, param.value) is None:
            missing_params.append(f"'{param.value}'")

    if missing_params:
        if len(missing_params) == 1:
            missing_params_str = missing_params[0]
        elif len(missing_params) == 2:
            missing_params_str = " and ".join(missing_params)
        else:
            missing_params_str = (
                ", ".join(missing_params[:-1]) + ", and " + missing_params[-1]
            )

        error_str = f"{missing_params_str} cannot be None for the '{metric.__name__}' metric"
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)


def check_arena_test_case_params(
    arena_test_case: ArenaTestCase,
    test_case_params: List[SingleTurnParams],
    metric: BaseArenaMetric,
    model: Optional[DeepEvalBaseLLM] = None,
    multimodal: Optional[bool] = False,
):
    if not isinstance(arena_test_case, ArenaTestCase):
        raise ValueError(
            f"Expected ArenaTestCase, got {type(arena_test_case).__name__}"
        )

    cases = [contestant.test_case for contestant in arena_test_case.contestants]
    ref_input = cases[0].input
    for case in cases[1:]:
        if case.input != ref_input:
            raise ValueError("All contestants must have the same 'input'.")

    ref_expected = cases[0].expected_output
    for case in cases[1:]:
        if case.expected_output != ref_expected:
            raise ValueError(
                "All contestants must have the same 'expected_output'."
            )

    for test_case in cases:
        check_llm_test_case_params(
            test_case, test_case_params, None, None, metric, model, multimodal
        )
