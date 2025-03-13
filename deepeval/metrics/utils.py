import inspect
import json
import re
from typing import Any, Dict, Optional, List, Union, Tuple
from deepeval.errors import MissingTestCaseParamsError
from deepeval.models import (
    GPTModel,
    DeepEvalBaseLLM,
    MultimodalGPTModel,
    DeepEvalBaseMLLM,
)
from deepeval.models.gpt_model_schematic import SchematicGPTModel

from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    MLLMTestCase,
    MLLMTestCaseParams,
    ConversationalTestCase,
    MLLMImage,
)


def copy_metrics(
    metrics: List[
        Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric]
    ],
) -> List[Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric]]:
    copied_metrics = []
    for metric in metrics:
        metric_class = type(metric)
        args = vars(metric)

        superclasses = metric_class.__mro__

        valid_params = []

        for superclass in superclasses:
            signature = inspect.signature(superclass.__init__)
            superclass_params = signature.parameters.keys()
            valid_params.extend(superclass_params)
        valid_params = set(valid_params)
        valid_args = {key: args[key] for key in valid_params if key in args}

        copied_metrics.append(metric_class(**valid_args))
    return copied_metrics


def format_turns(
    llm_test_cases: List[LLMTestCase], test_case_params: List[LLMTestCaseParams]
) -> List[Dict[str, Union[str, List[str]]]]:
    res = []
    for llm_test_case in llm_test_cases:
        dict = {}
        for param in test_case_params:
            value = getattr(llm_test_case, param.value)
            if value:
                dict[param.value] = value
        res.append(dict)
    return res


def process_llm_test_cases_windows(
    llm_test_cases_windows: List[List[LLMTestCase]],
    test_case_params: List[LLMTestCaseParams],
) -> List[List[Dict[str, str]]]:
    res = []
    for llm_test_cases_window in llm_test_cases_windows:
        window = []
        for llm_test_case in llm_test_cases_window:
            dict = {}
            for param in test_case_params:
                if getattr(llm_test_case, param.value):
                    value = getattr(llm_test_case, param.value)
                    dict[param.value] = value
            window.append(dict)
        res.append(window)
    return res


def get_turns_in_sliding_window(turns: List[LLMTestCase], window_size: int):
    for i in range(len(turns)):
        yield turns[max(0, i - window_size + 1) : i + 1]


def print_verbose_logs(metric: str, logs: str):
    print("*" * 50)
    print(f"{metric} Verbose Logs")
    print("*" * 50)
    print("")
    print(logs)
    print("")
    print("=" * 70)


def construct_verbose_logs(metric: BaseMetric, steps: List[str]) -> str:
    verbose_logs = ""
    for i in range(len(steps) - 1):
        verbose_logs += steps[i]

        # don't add new line for penultimate step
        if i < len(steps) - 2:
            verbose_logs += " \n \n"

    if metric.verbose_mode:
        # only print reason and score for deepeval
        print_verbose_logs(metric.__name__, verbose_logs + f"\n \n{steps[-1]}")

    return verbose_logs


def check_conversational_test_case_params(
    test_case: ConversationalTestCase,
    test_case_params: List[LLMTestCaseParams],
    metric: BaseConversationalMetric,
    require_chatbot_role: bool = False,
):
    if isinstance(test_case, ConversationalTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'ConversationalTestCase' using the conversational '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

    if require_chatbot_role and test_case.chatbot_role is None:
        error_str = f"'chatbot_role' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if len(test_case.turns) == 0:
        error_str = "'turns' in conversational test case cannot be empty."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    for turn in test_case.turns:
        test_case = turn
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
                    ", ".join(missing_params[:-1])
                    + ", and "
                    + missing_params[-1]
                )

            error_str = f"{missing_params_str} for `llm_test_case` turns cannot be None for the '{metric.__name__}' metric"
            metric.error = error_str
            raise MissingTestCaseParamsError(error_str)


def check_llm_test_case_params(
    test_case: LLMTestCase,
    test_case_params: List[LLMTestCaseParams],
    metric: BaseMetric,
):
    if isinstance(test_case, LLMTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'LLMTestCase' using the non-conversational '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

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


def check_mllm_test_case_params(
    test_case: MLLMTestCase,
    test_case_params: List[MLLMTestCaseParams],
    input_image_count: Optional[int],
    actual_output_image_count: Optional[int],
    metric: BaseMetric,
):
    if input_image_count:
        count = 0
        for ele in test_case.input:
            if isinstance(ele, MLLMImage):
                count += 1
        if count != input_image_count:
            error_str = f"Can only evaluate test cases with '{input_image_count}' input images using the '{metric.__name__}' metric. `{count}` found."
            raise ValueError(error_str)

    if actual_output_image_count:
        count = 0
        for ele in test_case.actual_output:
            if isinstance(ele, MLLMImage):
                count += 1
        if count != actual_output_image_count:
            error_str = f"Unable to evaluate test cases with '{actual_output_image_count}' output images using the '{metric.__name__}' metric. `{count}` found."
            raise ValueError(error_str)

    if isinstance(test_case, MLLMTestCase) is False:
        error_str = f"Unable to evaluate test cases that are not of type 'MLLMTestCase' using the '{metric.__name__}' metric."
        metric.error = error_str
        raise ValueError(error_str)

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


def trimAndLoadJson(
    input_string: str,
    metric: Optional[BaseMetric] = None,
) -> Any:
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    # Remove trailing comma if one is present
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        if metric is not None:
            metric.error = error_str
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def initialize_model(
    model: Optional[Union[str, DeepEvalBaseLLM, GPTModel]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
    """
    Returns a tuple of (initialized DeepEvalBaseLLM, using_native_model boolean)
    """
    # If model is a GPTModel, it should be deemed as using native model
    if isinstance(model, GPTModel):
        return model, True
    # If model is a DeepEvalBaseLLM but not a GPTModel, we can not assume it is a native model
    if isinstance(model, DeepEvalBaseLLM):
        return model, False

    # If the model is a string, we initialize a GPTModel and use as a native model
    if isinstance(model, str) or model is None:
        return GPTModel(model=model), True

    # Otherwise (the model is a wrong type), we raise an error
    raise TypeError(
        f"Unsupported type for model: {type(model)}. Expected None, str, DeepEvalBaseLLM, or GPTModel."
    )


def initialize_multimodal_model(
    model: Optional[Union[str, DeepEvalBaseMLLM, MultimodalGPTModel]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
    """
    Returns a tuple of (initialized DeepEvalBaseMLLM, using_native_model boolean)
    """
    # If model is a MultimodalGPTModel, it should be deemed as using native model
    if isinstance(model, MultimodalGPTModel):
        return model, True
    # If model is a DeepEvalBaseMLLM but not a MultimodalGPTModel, we can not assume it is a native model
    if isinstance(model, DeepEvalBaseMLLM):
        return model, False
    # Otherwise (the model is a string or None), we initialize a GPTModel and use as a native model
    return MultimodalGPTModel(model=model), True


def initialize_schematic_model(
    model: Optional[Union[str, DeepEvalBaseLLM, SchematicGPTModel]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
    """
    Returns a tuple of (initialized DeepEvalBaseLLM, using_native_model boolean)
    """
    # If model is a GPTModel, it should be deemed as using native model
    if isinstance(model, SchematicGPTModel):
        return model, True
    # If model is a DeepEvalBaseLLM but not a GPTModel, we can not assume it is a native model
    if isinstance(model, DeepEvalBaseLLM):
        return model, False
    # Otherwise (the model is a string or None), we initialize a GPTModel and use as a native model
    return SchematicGPTModel(model=model), True
