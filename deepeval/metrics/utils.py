import inspect
import json
import re
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from deepeval.errors import (
    MissingTestCaseParamsError,
)
from deepeval.utils import convert_to_multi_modal_array
from deepeval.models import (
    DeepEvalBaseLLM,
    GPTModel,
    AnthropicModel,
    AzureOpenAIModel,
    OllamaModel,
    LocalModel,
    OpenAIEmbeddingModel,
    AzureOpenAIEmbeddingModel,
    OllamaEmbeddingModel,
    LocalEmbeddingModel,
    GeminiModel,
    AmazonBedrockModel,
    LiteLLMModel,
    PortkeyModel,
    KimiModel,
    GrokModel,
    DeepSeekModel,
)
from deepeval.models.llms.constants import (
    OPENAI_MODELS_DATA,
    GEMINI_MODELS_DATA,
    OLLAMA_MODELS_DATA,
    ANTHROPIC_MODELS_DATA,
    GROK_MODELS_DATA,
    KIMI_MODELS_DATA,
)
from deepeval.key_handler import (
    ModelKeyValues,
    EmbeddingKeyValues,
    KEY_FILE_HANDLER,
)
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    BaseArenaMetric,
)
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
    MLLMImage,
    Turn,
    ArenaTestCase,
    ToolCall,
    TurnParams,
)

MULTIMODAL_SUPPORTED_MODELS = {
    GPTModel: OPENAI_MODELS_DATA,
    GeminiModel: GEMINI_MODELS_DATA,
    OllamaModel: OLLAMA_MODELS_DATA,
    AzureOpenAIModel: OPENAI_MODELS_DATA,
    KimiModel: KIMI_MODELS_DATA,
    AnthropicModel: ANTHROPIC_MODELS_DATA,
    GrokModel: GROK_MODELS_DATA,
}


def copy_metrics(
    metrics: List[Union[BaseMetric, BaseConversationalMetric]],
) -> List[Union[BaseMetric, BaseConversationalMetric]]:
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


def convert_turn_to_dict(
    turn: Turn,
    turn_params: List[TurnParams] = [TurnParams.CONTENT, TurnParams.ROLE],
) -> Dict:
    result = {
        param.value: getattr(turn, param.value)
        for param in turn_params
        if (
            param != TurnParams.SCENARIO
            and param != TurnParams.EXPECTED_OUTCOME
            and getattr(turn, param.value) is not None
        )
    }
    return result


def get_turns_in_sliding_window(turns: List[Turn], window_size: int):
    for i in range(len(turns)):
        yield turns[max(0, i - window_size + 1) : i + 1]


def get_unit_interactions(turns: List[Turn]) -> List[List[Turn]]:
    units: List[List[Turn]] = []
    current: List[Turn] = []
    has_user = False

    for turn in turns:
        # Boundary: user after assistant, but only if we've already seen a user in current
        if (
            current
            and current[-1].role == "assistant"
            and turn.role == "user"
            and has_user
        ):
            units.append(current)  # finalize previous unit
            current = [turn]  # start new unit with this user
            has_user = True
            continue

        # Otherwise just accumulate
        current.append(turn)
        if turn.role == "user":
            has_user = True

    # Finalize last unit only if it ends with assistant and includes a user
    if (
        current
        and len(current) > 1
        and current[-1].role == "assistant"
        and has_user
    ):
        units.append(current)

    return units


def print_tools_called(tools_called_list: List[ToolCall]):
    string = "[\n"
    for index, tools_called in enumerate(tools_called_list):
        json_string = json.dumps(tools_called.model_dump(), indent=4)
        indented_json_string = "\n".join(
            "  " + line for line in json_string.splitlines()
        )
        string += indented_json_string
        if index < len(tools_called_list) - 1:
            string += ",\n"
        else:
            string += "\n"
    string += "]"
    return string


def print_verbose_logs(metric: str, logs: str):
    sys.stdout.write("*" * 50 + "\n")
    sys.stdout.write(f"{metric} Verbose Logs\n")
    sys.stdout.write("*" * 50 + "\n")
    sys.stdout.write("\n")
    sys.stdout.write(logs + "\n")
    sys.stdout.write("\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()


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
    test_case_params: List[TurnParams],
    metric: BaseConversationalMetric,
    require_chatbot_role: bool = False,
    model: Optional[DeepEvalBaseLLM] = None,
    multimodal: Optional[bool] = False,
):
    if multimodal:
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
        TurnParams.EXPECTED_OUTCOME in test_case_params
        and test_case.expected_outcome is None
    ):
        error_str = f"'expected_outcome' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
        metric.error = error_str
        raise MissingTestCaseParamsError(error_str)

    if TurnParams.SCENARIO in test_case_params and test_case.scenario is None:
        error_str = f"'scenario' in a conversational test case cannot be empty for the '{metric.__name__}' metric."
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
    test_case_params: List[LLMTestCaseParams],
    input_image_count: Optional[int],
    actual_output_image_count: Optional[int],
    metric: Union[BaseMetric, BaseArenaMetric],
    model: Optional[DeepEvalBaseLLM] = None,
    multimodal: Optional[bool] = False,
):
    if multimodal:
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
    if LLMTestCaseParams.ACTUAL_OUTPUT in test_case_params:
        actual_output = getattr(
            test_case, LLMTestCaseParams.ACTUAL_OUTPUT.value
        )
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
    test_case_params: List[LLMTestCaseParams],
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


SchemaType = TypeVar("SchemaType")
ReturnType = TypeVar("ReturnType")


def generate_with_schema_and_extract(
    metric: Union[BaseMetric, BaseArenaMetric, BaseConversationalMetric],
    prompt: Any,
    schema_cls: Type[SchemaType],
    *,
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
) -> ReturnType:
    """
    Synchronous wrapper:
    - calls model.generate_with_schema(...)
    - accrues cost if applicable
    - if schema instance -> extract_schema
      else parse JSON -> extract_json
    """
    if metric.using_native_model:
        result, cost = metric.model.generate_with_schema(
            prompt, schema=schema_cls
        )
        metric._accrue_cost(cost)
    else:
        result = metric.model.generate_with_schema(prompt, schema=schema_cls)
    if isinstance(result, schema_cls):
        return extract_schema(result)
    data = trimAndLoadJson(result, metric)
    return extract_json(data)


async def a_generate_with_schema_and_extract(
    metric: Union[BaseMetric, BaseArenaMetric, BaseConversationalMetric],
    prompt: Any,
    schema_cls: Type[SchemaType],
    *,
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
) -> ReturnType:
    if metric.using_native_model:
        result, cost = await metric.model.a_generate_with_schema(
            prompt, schema=schema_cls
        )
        metric._accrue_cost(cost)
    else:
        result = await metric.model.a_generate_with_schema(
            prompt, schema=schema_cls
        )

    if isinstance(result, schema_cls):
        return extract_schema(result)

    data = trimAndLoadJson(result, metric)
    return extract_json(data)


###############################################
# Default Model Providers
###############################################


def should_use_anthropic_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_ANTHROPIC_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_azure_openai():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_AZURE_OPENAI)
    return value.lower() == "yes" if value is not None else False


def should_use_local_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_LOCAL_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_ollama_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.LOCAL_MODEL_API_KEY)
    return value == "ollama"


def should_use_gemini_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_GEMINI_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_openai_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_OPENAI_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_litellm():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_LITELLM)
    return value.lower() == "yes" if value is not None else False


def should_use_portkey():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_PORTKEY_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_deepseek_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_DEEPSEEK_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_moonshot_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_MOONSHOT_MODEL)
    return value.lower() == "yes" if value is not None else False


def should_use_grok_model():
    value = KEY_FILE_HANDLER.fetch_data(ModelKeyValues.USE_GROK_MODEL)
    return value.lower() == "yes" if value is not None else False


###############################################
# LLM
###############################################


def initialize_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> Tuple[DeepEvalBaseLLM, bool]:
    """
    Returns a tuple of (initialized DeepEvalBaseLLM, using_native_model boolean)
    """
    # If model is natively supported, it should be deemed as using native model
    if is_native_model(model):
        return model, True
    # If model is a DeepEvalBaseLLM but not a native model, we can not assume it is a native model
    if isinstance(model, DeepEvalBaseLLM):
        return model, False
    if should_use_openai_model():
        return GPTModel(), True
    if should_use_gemini_model():
        return GeminiModel(), True
    if should_use_litellm():
        return LiteLLMModel(), True
    if should_use_portkey():
        return PortkeyModel(), True
    if should_use_ollama_model():
        return OllamaModel(), True
    elif should_use_local_model():
        return LocalModel(), True
    elif should_use_azure_openai():
        return AzureOpenAIModel(model=model), True
    elif should_use_moonshot_model():
        return KimiModel(model=model), True
    elif should_use_grok_model():
        return GrokModel(), True
    elif should_use_deepseek_model():
        return DeepSeekModel(model=model), True
    elif should_use_anthropic_model():
        return AnthropicModel(), True
    elif isinstance(model, str) or model is None:
        return GPTModel(model=model), True

    # Otherwise (the model is a wrong type), we raise an error
    raise TypeError(
        f"Unsupported type for model: {type(model)}. Expected None, str, DeepEvalBaseLLM, GPTModel, AzureOpenAIModel, LiteLLMModel, OllamaModel, LocalModel."
    )


def is_native_model(
    model: Optional[Union[str, DeepEvalBaseLLM]] = None,
) -> bool:
    if (
        isinstance(model, GPTModel)
        or isinstance(model, AnthropicModel)
        or isinstance(model, AzureOpenAIModel)
        or isinstance(model, OllamaModel)
        or isinstance(model, LocalModel)
        or isinstance(model, GeminiModel)
        or isinstance(model, AmazonBedrockModel)
        or isinstance(model, LiteLLMModel)
        or isinstance(model, KimiModel)
        or isinstance(model, GrokModel)
        or isinstance(model, DeepSeekModel)
    ):
        return True
    else:
        return False


###############################################
# Multimodal Model
###############################################


###############################################
# Embedding Model
###############################################


def should_use_azure_openai_embedding():
    value = KEY_FILE_HANDLER.fetch_data(
        EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING
    )
    return value.lower() == "yes" if value is not None else False


def should_use_local_embedding():
    value = KEY_FILE_HANDLER.fetch_data(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
    return value.lower() == "yes" if value is not None else False


def should_use_ollama_embedding():
    api_key = KEY_FILE_HANDLER.fetch_data(
        EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY
    )
    return api_key == "ollama"


def initialize_embedding_model(
    model: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
) -> DeepEvalBaseEmbeddingModel:
    if isinstance(model, DeepEvalBaseEmbeddingModel):
        return model
    if should_use_ollama_embedding():
        return OllamaEmbeddingModel()
    elif should_use_local_embedding():
        return LocalEmbeddingModel()
    elif should_use_azure_openai_embedding():
        return AzureOpenAIEmbeddingModel()
    elif isinstance(model, str) or model is None:
        return OpenAIEmbeddingModel(model=model)

    # Otherwise (the model is a wrong type), we raise an error
    raise TypeError(
        f"Unsupported type for embedding model: {type(model)}. Expected None, str, DeepEvalBaseEmbeddingModel, OpenAIEmbeddingModel, AzureOpenAIEmbeddingModel, OllamaEmbeddingModel, LocalEmbeddingModel."
    )
