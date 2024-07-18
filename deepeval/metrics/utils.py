import json
from typing import Any, Optional, List, Union, Tuple
from deepeval.models import GPTModel, DeepEvalBaseLLM

from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)


def construct_verbose_logs(metric: BaseMetric, steps: List[str]) -> str:
    verbose_logs = ""
    for i in range(len(steps) - 1):
        verbose_logs += steps[i]

        # don't add new line for penultimate step
        if i < len(steps) - 2:
            verbose_logs += "\n\n"

    if metric.verbose_mode:
        # only print reason and score for deepeval
        print_verbose_logs(metric.__name__, verbose_logs + f"\n\n{steps[-1]}")

    return verbose_logs


def validate_conversational_test_case(
    test_case: ConversationalTestCase,
    metric: BaseMetric,
) -> LLMTestCase:
    if len(test_case.messages) == 0:
        error_str = "'messages' in conversational test case cannot be empty."
        metric.error = error_str
        raise ValueError(error_str)

    return test_case.messages[len(test_case.messages) - 1]


def check_llm_test_case_params(
    test_case: LLMTestCase,
    test_case_params: List[LLMTestCaseParams],
    metric: BaseMetric,
):
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
        raise ValueError(error_str)


def trimAndLoadJson(
    input_string: str, metric: Optional[BaseMetric] = None
) -> Any:
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

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
    # Otherwise (the model is a string or None), we initialize a GPTModel and use as a native model
    return GPTModel(model=model), True


def print_verbose_logs(metric: str, logs: str):
    print("*" * 50)
    print(f"{metric} Verbose Logs")
    print("*" * 50)
    print("")
    print(logs)
    print("")
    print("=" * 70)
