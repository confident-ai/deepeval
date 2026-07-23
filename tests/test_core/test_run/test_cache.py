import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import pytest
from deepeval.metrics import BaseMetric
from deepeval.metrics.answer_relevancy.answer_relevancy import (
    AnswerRelevancyMetric,
)
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.metrics.dag.nodes import TaskNode, VerdictNode
from deepeval.metrics.exact_match.exact_match import ExactMatchMetric
from deepeval.metrics.pattern_match.pattern_match import PatternMatchMetric
from deepeval.metrics.tool_correctness.tool_correctness import (
    ToolCorrectnessMetric,
)
from deepeval.models.llms.openai_model import GPTModel
from pydantic import (
    BaseModel as PydanticBaseModel,
    Field,
    PrivateAttr,
    create_model,
)

try:
    from pydantic import AliasChoices, field_validator
except ImportError:
    AliasChoices = None
    from pydantic import validator as _pydantic_v1_validator

    def field_validator(*fields, **kwargs):
        def decorator(fn):
            validator_fn = fn.__func__ if isinstance(fn, classmethod) else fn
            return _pydantic_v1_validator(*fields, allow_reuse=True)(
                validator_fn
            )

        return decorator


from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.test_run import cache_identity as cache_identity_module
from deepeval.test_run.api import MetricData
from deepeval.test_run.cache import (
    Cache,
    CachedMetricData,
    CachedTestCase,
    CachedTestRun,
    MetricConfiguration,
)


class _ConfigurableMetric(BaseMetric):
    def __init__(
        self,
        rubric: str,
        weights: dict[str, float],
        steps=None,
        threshold: float = 0.5,
    ):
        self.rubric = rubric
        self.weights = weights
        self.steps = steps or []
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = False
        self.error = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score = 1.0
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Configurable Custom"


class _MetricWithSecretParameter(_ConfigurableMetric):
    def __init__(self, rubric: str, api_key: str, client=None):
        super().__init__(rubric=rubric, weights={"harm": 1.0})
        self.api_key = api_key
        self.client = client


class _MetricWithUnderscoreSecretParameter(_ConfigurableMetric):
    def __init__(self, rubric: str, _access_token: str):
        super().__init__(rubric=rubric, weights={"harm": 1.0})
        self._access_token = _access_token


class _MetricWithNestedConfigParameter(_ConfigurableMetric):
    def __init__(self, config: dict):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.config = config


class _EndpointMetric(_ConfigurableMetric):
    def __init__(self, endpoint: str):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.endpoint = endpoint


class _TokenizerMetric(_ConfigurableMetric):
    def __init__(self, tokenizer_name: str):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.tokenizer_name = tokenizer_name


class _TypeSensitiveCollectionMetric(_ConfigurableMetric):
    def __init__(self, value):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.value = value


class _CacheMode(Enum):
    STRICT = "strict"


class _IntCacheMode(IntEnum):
    STRICT = 1


class _StrCacheMode(str, Enum):
    STRICT = "strict"


class _EmbeddingsConfigMetric(_ConfigurableMetric):
    def __init__(self, embeddings):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.embeddings = embeddings


class _FakeEmbeddings:
    pass


class _AliasedConstructorMetric(_ConfigurableMetric):
    def __init__(self, prompt_template: str):
        super().__init__(rubric="internal", weights={"harm": 1.0})
        self.prompt = prompt_template


class _KwargsMetric(_ConfigurableMetric):
    def __init__(self, **kwargs):
        super().__init__(rubric="kwargs", weights={"harm": 1.0})
        self.mode = kwargs["mode"]


class _ArgsMetric(_ConfigurableMetric):
    def __init__(self, *args):
        super().__init__(rubric="args", weights={"harm": 1.0})
        self.mode = args[0]


class _TelemetryMetric(_ConfigurableMetric):
    def __init__(self, rubric: str, _track: bool):
        super().__init__(rubric=rubric, weights={"harm": 1.0})
        self._track = _track


class _PropertyBackedMetric(_ConfigurableMetric):
    def __init__(self, config: dict):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self._config = config

    @property
    def config(self):
        raise AssertionError("cache extraction should not access properties")


class _NamedModel:
    def __init__(self, name: str):
        self.name = name

    def get_model_name(self):
        return self.name


class _SensitiveNamedModel(_NamedModel):
    pass


class _TunableNamedModel(_NamedModel):
    def __init__(self, name: str, temperature: float):
        super().__init__(name)
        self.temperature = temperature


class _ExplodingModelName:
    def __init__(self, name: str = "model-a"):
        self.name = name

    def get_model_name(self):
        raise AssertionError(
            "cache fingerprinting should not call get_model_name"
        )


class _UnnamedModel:
    pass


class _MetricWithModelWithoutEvaluationModel(_ConfigurableMetric):
    def __init__(self, model):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.model = model


class _MetricWithDerivedEvaluationModel(_MetricWithModelWithoutEvaluationModel):
    def __init__(self, model):
        super().__init__(model)
        self.evaluation_model = self.model.get_model_name()


class _MetricWithPrivateDerivedEvaluationModel(_ConfigurableMetric):
    def __init__(self, model):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self._model = model
        self.evaluation_model = self._model.get_model_name()


class _MetricWithInitializedNativeModel(_ConfigurableMetric):
    def __init__(self, model: str):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.model = GPTModel(model=model, api_key="sk-test")
        self.using_native_model = True
        self.evaluation_model = self.model.get_model_name()


class _MetricWithInjectedNativeModel(_ConfigurableMetric):
    def __init__(self, model):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.model = model
        self.using_native_model = True
        self.evaluation_model = model.model_name


class _NativeModelWithMaxTokens:
    __module__ = "deepeval.models.llms.anthropic_model"

    def __init__(self, model_name: str, max_tokens: int):
        self.model_name = model_name
        self._max_tokens = max_tokens


class _NativeModelWithVertexBackend:
    __module__ = "deepeval.models.llms.gemini_model"

    def __init__(self, model_name: str, use_vertexai: bool):
        self.model_name = model_name
        self.project = "deepeval-test-project"
        self.location = "us-central1"
        self.use_vertexai = use_vertexai


# Mirrors Pydantic alias helper versions that expose slot-backed attributes
# rather than a plain __dict__.
_SlotBackedAliasChoices = type(
    "AliasChoices",
    (),
    {
        "__module__": "pydantic.aliases",
        "__slots__": ("choices",),
        "__init__": lambda self, choices: setattr(self, "choices", choices),
    },
)

_SlotBackedAliasPath = type(
    "AliasPath",
    (),
    {
        "__module__": "pydantic.aliases",
        "__slots__": ("path",),
        "__init__": lambda self, path: setattr(self, "path", path),
    },
)


class _MetricWithEvaluationModelString(_ConfigurableMetric):
    def __init__(self, evaluation_model: str):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.evaluation_model = evaluation_model


class _MetricWithLanguage(_ConfigurableMetric):
    def __init__(self, language: str):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.language = language


class _FreeFormConfigMetric(_ConfigurableMetric):
    def __init__(
        self,
        criteria=None,
        evaluation_steps=None,
        assessment_questions=None,
    ):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.assessment_questions = assessment_questions


class _HostileCriteriaMetric(_ConfigurableMetric):
    criteria_accesses = 0

    def __init__(self):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.evaluation_steps = None

    @property
    def criteria(self):
        type(self).criteria_accesses += 1
        raise AssertionError("cache fingerprinting should not access criteria")


class _ScalarConfigMetric(_ConfigurableMetric):
    def __init__(self, field: str, value):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        setattr(self, field, value)


class _ExplodingStringValue:
    def __str__(self):
        raise AssertionError(
            "cache fingerprinting should not stringify arbitrary objects"
        )


class _ExplodingClassValue:
    def __getattribute__(self, name):
        if name == "__class__":
            raise AssertionError(
                "cache fingerprinting should use type(value), not __class__"
            )
        return object.__getattribute__(self, name)


class _HostileDictMetric(_ConfigurableMetric):
    def __getattribute__(self, name):
        if name == "__dict__":
            raise RuntimeError(
                "cache fingerprinting should miss closed on hostile __dict__"
            )
        return object.__getattribute__(self, name)


class _HostileNativeModel:
    __module__ = "deepeval.models.llms.hostile"

    def __getattribute__(self, name):
        if name == "__dict__":
            raise RuntimeError(
                "cache fingerprinting should miss closed on hostile model fields"
            )
        return object.__getattribute__(self, name)


class _MetricWithHostileNativeModel(_ConfigurableMetric):
    def __init__(self, model):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.model = model
        self.using_native_model = True
        self.evaluation_model = "hostile-native-model"


class _ExplodingDict(dict):
    def __len__(self):
        raise AssertionError(
            "cache fingerprinting should not call custom mapping methods"
        )

    def __iter__(self):
        raise AssertionError(
            "cache fingerprinting should not iterate custom mappings"
        )

    def items(self):
        raise AssertionError(
            "cache fingerprinting should not call custom mapping items"
        )


class _ExplodingList(list):
    def __len__(self):
        raise AssertionError(
            "cache fingerprinting should not call custom sequence methods"
        )

    def __iter__(self):
        raise AssertionError(
            "cache fingerprinting should not iterate custom sequences"
        )


class _MetricWithEvaluationModelObject(_ConfigurableMetric):
    def __init__(self, evaluation_model):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.evaluation_model = evaluation_model


class _DAGChildMetric(_ConfigurableMetric):
    def __init__(self, setting):
        super().__init__(rubric="child", weights={"harm": 1.0})
        self.setting = setting


class _FirstSchema(PydanticBaseModel):
    value: str


class _SecondSchema(PydanticBaseModel):
    value: str


class _LiteralIntSchema(PydanticBaseModel):
    value: Literal[1]


class _LiteralStringSchema(PydanticBaseModel):
    value: Literal["1"]


class _ValidatedSchema(PydanticBaseModel):
    value: int

    @field_validator("value")
    @classmethod
    def validate_value(cls, value):
        return value


class _SchemaWithPrivateState(PydanticBaseModel):
    value: str

    _mode: str = PrivateAttr(default="strict")


class _SchemaWithHostileGetattribute(PydanticBaseModel):
    value: str

    def __getattribute__(self, name):
        if name == "__pydantic_private__":
            raise RuntimeError(
                "cache fingerprinting should not call model __getattribute__"
            )
        return super().__getattribute__(name)


class _SchemaWithHostileConfig(PydanticBaseModel):
    value: str


class _SchemaWithHostileField(PydanticBaseModel):
    value: str


class _HostileConfigMapping(dict):
    def __iter__(self):
        raise AssertionError(
            "cache fingerprinting should not iterate custom model_config"
        )

    def items(self):
        raise AssertionError(
            "cache fingerprinting should not call custom model_config items"
        )


class _HostilePydanticField:
    def __getattribute__(self, name):
        raise AssertionError(
            "cache fingerprinting should not inspect custom field objects"
        )


class _HostileFieldKey:
    def __str__(self):
        raise AssertionError(
            "cache fingerprinting should not stringify custom field keys"
        )


class _HostileIterable:
    def __iter__(self):
        raise AssertionError(
            "cache fingerprinting should not iterate custom DAG containers"
        )


class _SchemaMetric(_ConfigurableMetric):
    def __init__(self, expected_schema: PydanticBaseModel):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.expected_schema = expected_schema


class _SchemaClassMetric(_ConfigurableMetric):
    def __init__(self, expected_schema: type[PydanticBaseModel]):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.expected_schema = expected_schema


class _NoDictMetric:
    __slots__ = ("threshold", "strict_mode", "rubric")

    def __init__(self, rubric: str = "safety"):
        self.threshold = 0.5
        self.strict_mode = False
        self.rubric = rubric

    @property
    def __name__(self):
        return "No Dict Metric"


class _NoInitMetric(BaseMetric):
    threshold = 0.5
    strict_mode = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return 1.0

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return 1.0

    def is_successful(self) -> bool:
        return True

    @property
    def __name__(self):
        return "No Init Metric"


class _SlotsModelMetric:
    __slots__ = ("threshold", "strict_mode", "model")

    def __init__(self, model: str):
        self.threshold = 0.5
        self.strict_mode = False
        self.model = model

    @property
    def __name__(self):
        return "Slots Model Metric"


class _SignaturelessMetric(_ConfigurableMetric):
    def __init__(self, rubric: str):
        super().__init__(rubric=rubric, weights={"harm": 1.0})


class _DAGBackedMetric(_ConfigurableMetric):
    def __init__(self, dag: DeepAcyclicGraph):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.dag = dag


class _CustomTaskNode(TaskNode):
    pass


@dataclass
class _CyclicDataclassConfig:
    mode: str
    next_config: object = None


class _DataclassConfigMetric(_ConfigurableMetric):
    def __init__(self, config: _CyclicDataclassConfig):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.config = config


class _ModelWithExplodingDump(PydanticBaseModel):
    value: str

    def model_dump(self, *args, **kwargs):
        raise AssertionError("cache fingerprinting should not call model_dump")

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        raise AssertionError(
            "cache fingerprinting should not generate live schemas"
        )


class _ExplodingModelMetric(_ConfigurableMetric):
    def __init__(self, config: _ModelWithExplodingDump):
        super().__init__(rubric="safety", weights={"harm": 1.0})
        self.config = config


def _named_schema_default():
    return "first"


def _single_node_dag(instructions: str) -> DeepAcyclicGraph:
    return DeepAcyclicGraph(
        root_nodes=[
            TaskNode(
                instructions=instructions,
                output_label="result",
                children=[],
                evaluation_params=[SingleTurnParams.INPUT],
            )
        ]
    )


def _shared_child_dag(shared: bool) -> DeepAcyclicGraph:
    first_shared_child = TaskNode(
        instructions="Shared child",
        output_label="shared",
        children=[],
        evaluation_params=[SingleTurnParams.INPUT],
    )
    second_shared_child = (
        first_shared_child
        if shared
        else TaskNode(
            instructions="Shared child",
            output_label="shared",
            children=[],
            evaluation_params=[SingleTurnParams.INPUT],
        )
    )
    return DeepAcyclicGraph(
        root_nodes=[
            TaskNode(
                instructions="Root A",
                output_label="a",
                children=[first_shared_child],
                evaluation_params=[SingleTurnParams.INPUT],
            ),
            TaskNode(
                instructions="Root B",
                output_label="b",
                children=[second_shared_child],
                evaluation_params=[SingleTurnParams.INPUT],
            ),
        ]
    )


def _metric_child_dag(setting) -> DeepAcyclicGraph:
    return DeepAcyclicGraph(
        root_nodes=[
            VerdictNode(
                verdict=True,
                child=_DAGChildMetric(setting=setting),
            )
        ]
    )


def _metric_child_with_evaluation_model_dag(
    evaluation_model: str,
) -> DeepAcyclicGraph:
    return DeepAcyclicGraph(
        root_nodes=[
            VerdictNode(
                verdict=True,
                child=_MetricWithEvaluationModelString(evaluation_model),
            )
        ]
    )


def _cached_metric_data(metric: BaseMetric) -> CachedMetricData:
    return CachedMetricData(
        metric_data=MetricData(
            name=metric.__name__,
            threshold=metric.threshold,
            success=True,
            score=1.0,
        ),
        metric_configuration=Cache.create_metric_configuration(metric),
    )


def _cached_case_for(metric: BaseMetric) -> CachedTestCase:
    return CachedTestCase(cached_metrics_data=[_cached_metric_data(metric)])


def _encode_repeatedly(value: str, passes: int = 5) -> str:
    for _ in range(passes):
        value = quote(value, safe="")
    return value


def test_pattern_match_cache_hits_when_constructor_parameters_match():
    expected_entry = _cached_metric_data(
        PatternMatchMetric(pattern=r"expected", ignore_case=True)
    )
    cached_case = CachedTestCase(
        cached_metrics_data=[
            _cached_metric_data(ExactMatchMetric()),
            expected_entry,
        ]
    )

    assert (
        Cache.get_metric_data(
            PatternMatchMetric(pattern=r"expected", ignore_case=True),
            cached_case,
        )
        is expected_entry
    )


def test_pattern_match_cache_misses_when_pattern_changes():
    cached_case = _cached_case_for(
        PatternMatchMetric(pattern=r"expected", ignore_case=False)
    )

    assert (
        Cache.get_metric_data(
            PatternMatchMetric(pattern=r"actual", ignore_case=False),
            cached_case,
        )
        is None
    )


def test_pattern_match_cache_misses_when_ignore_case_changes():
    cached_case = _cached_case_for(
        PatternMatchMetric(pattern=r"expected", ignore_case=False)
    )

    assert (
        Cache.get_metric_data(
            PatternMatchMetric(pattern=r"expected", ignore_case=True),
            cached_case,
        )
        is None
    )


def test_custom_metric_cache_misses_when_constructor_parameter_changes():
    cached_case = _cached_case_for(
        _ConfigurableMetric(
            rubric="safety",
            weights={"harm": 1.0, "accuracy": 0.5},
        )
    )

    assert (
        Cache.get_metric_data(
            _ConfigurableMetric(
                rubric="quality",
                weights={"harm": 1.0, "accuracy": 0.5},
            ),
            cached_case,
        )
        is None
    )


def test_custom_metric_cache_misses_when_nested_constructor_parameter_changes():
    cached_case = _cached_case_for(
        _ConfigurableMetric(
            rubric="safety",
            weights={"harm": 1.0, "accuracy": 0.5},
        )
    )

    assert (
        Cache.get_metric_data(
            _ConfigurableMetric(
                rubric="safety",
                weights={"harm": 0.5, "accuracy": 0.5},
            ),
            cached_case,
        )
        is None
    )


def test_custom_metric_cache_ignores_dict_insertion_order():
    cached_case = _cached_case_for(
        _ConfigurableMetric(
            rubric="safety",
            weights={"harm": 1.0, "accuracy": 0.5},
        )
    )

    assert (
        Cache.get_metric_data(
            _ConfigurableMetric(
                rubric="safety",
                weights={"accuracy": 0.5, "harm": 1.0},
            ),
            cached_case,
        )
        is not None
    )


def test_mapping_constructor_parameters_with_non_string_keys_miss_closed():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithNestedConfigParameter({1: "x"})
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["config"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _MetricWithNestedConfigParameter({1: "x"}),
            _cached_case_for(_MetricWithNestedConfigParameter({1: "x"})),
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _MetricWithNestedConfigParameter({"1": "x"}),
            _cached_case_for(_MetricWithNestedConfigParameter({1: "x"})),
        )
        is None
    )


def test_custom_metric_cache_preserves_ordered_constructor_parameters():
    cached_case = _cached_case_for(
        _ConfigurableMetric(
            rubric="safety",
            weights={"harm": 1.0},
            steps=["first", "second"],
        )
    )

    assert (
        Cache.get_metric_data(
            _ConfigurableMetric(
                rubric="safety",
                weights={"harm": 1.0},
                steps=["second", "first"],
            ),
            cached_case,
        )
        is None
    )


def test_collection_constructor_parameter_types_do_not_collide():
    cached_tuple = _cached_case_for(_TypeSensitiveCollectionMetric(("a", "b")))
    cached_frozenset = _cached_case_for(
        _TypeSensitiveCollectionMetric(frozenset({"a", "b"}))
    )

    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric(["a", "b"]),
            cached_tuple,
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric({"a", "b"}),
            cached_frozenset,
        )
        is None
    )


def test_enum_and_path_constructor_parameter_types_do_not_collide_with_strings():
    cached_enum = _cached_case_for(
        _TypeSensitiveCollectionMetric(_CacheMode.STRICT)
    )
    cached_path = _cached_case_for(
        _TypeSensitiveCollectionMetric(Path("strict"))
    )

    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric(_CacheMode.STRICT),
            cached_enum,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric("strict"),
            cached_enum,
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric(Path("strict")),
            cached_path,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric("strict"),
            cached_path,
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric(1),
            _cached_case_for(
                _TypeSensitiveCollectionMetric(_IntCacheMode.STRICT)
            ),
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _TypeSensitiveCollectionMetric("strict"),
            _cached_case_for(
                _TypeSensitiveCollectionMetric(_StrCacheMode.STRICT)
            ),
        )
        is None
    )


def test_secret_bearing_path_constructor_values_miss_cache_closed():
    for path_value in (
        Path("/download/token/abc123"),
        Path("object/X-Amz-Signature=abc123"),
        Path("/download%2Ftoken%2Fabc123"),
        Path("/download%2Fsig%2Fabc123"),
        Path("/download%252Ftoken%252Fabc123"),
        Path("object/X-Amz-Signature%253Dabc123"),
        Path(_encode_repeatedly("/download/token/abc123")),
        Path(_encode_repeatedly("object/X-Amz-Signature=abc123")),
        Path(r"C:\downloads\token\abc123"),
        Path(r"\server\share\sig\abc123"),
        Path(r"\\server\share\sig\abc123"),
        Path(_encode_repeatedly(r"C:\downloads\token\abc123")),
        Path(_encode_repeatedly(r"\\server\share\sig\abc123")),
    ):
        normalized = Cache._normalize_cache_parameter(
            "endpoint", path_value, set()
        )

        assert normalized == {"__deepeval_uncacheable__": "sensitive"}
        metric_configuration = Cache.create_metric_configuration(
            _TypeSensitiveCollectionMetric(path_value)
        )
        assert metric_configuration.custom_parameters is not None
        assert (
            metric_configuration.custom_parameters["value"]
            == Cache._UNCACHEABLE_CACHE_VALUE
        )
        assert str(path_value) not in json.dumps(
            metric_configuration.model_dump()
        )
        assert (
            Cache.get_metric_data(
                _TypeSensitiveCollectionMetric(path_value),
                _cached_case_for(_TypeSensitiveCollectionMetric(path_value)),
            )
            is None
        )


def test_non_finite_float_constructor_values_miss_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _ConfigurableMetric(
            rubric="safety",
            weights={"harm": float("inf")},
            steps=["first"],
        )
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["weights"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _ConfigurableMetric(
                rubric="safety",
                weights={"harm": float("inf")},
                steps=["first"],
            ),
            _cached_case_for(
                _ConfigurableMetric(
                    rubric="safety",
                    weights={"harm": float("inf")},
                    steps=["first"],
                )
            ),
        )
        is None
    )


def test_non_finite_metric_config_fields_miss_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        ExactMatchMetric(threshold=float("inf"))
    )

    assert metric_configuration.threshold == Cache._UNCACHEABLE_CACHE_VALUE
    assert "Infinity" not in json.dumps(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            ExactMatchMetric(threshold=float("inf")),
            _cached_case_for(ExactMatchMetric(threshold=float("inf"))),
        )
        is None
    )


def test_custom_metric_cache_misses_closed_when_constructor_parameter_is_unrecoverable():
    metric_configuration = Cache.create_metric_configuration(
        _AliasedConstructorMetric(prompt_template="safety prompt")
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["prompt_template"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _AliasedConstructorMetric(prompt_template="safety prompt"),
            _cached_case_for(
                _AliasedConstructorMetric(prompt_template="safety prompt")
            ),
        )
        is None
    )


def test_custom_metric_with_var_kwargs_misses_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _KwargsMetric(mode="strict")
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["kwargs"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _KwargsMetric(mode="strict"),
            _cached_case_for(_KwargsMetric(mode="strict")),
        )
        is None
    )


def test_custom_metric_with_var_args_misses_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _ArgsMetric("strict")
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["args"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _ArgsMetric("strict"),
            _cached_case_for(_ArgsMetric("strict")),
        )
        is None
    )


def test_metric_runtime_fields_do_not_invalidate_cache():
    cached_case = _cached_case_for(
        _ConfigurableMetric(
            rubric="safety",
            weights={"harm": 1.0, "accuracy": 0.5},
        )
    )
    metric = _ConfigurableMetric(
        rubric="safety",
        weights={"harm": 1.0, "accuracy": 0.5},
    )

    metric.score = 0.0
    metric.score_breakdown = {"other": 0.25}
    metric.reason = "fresh runtime details should not be cache config"
    metric.success = False
    metric.error = "runtime failure from a later run"
    metric.evaluation_cost = 123.45
    metric.input_tokens = 99
    metric.output_tokens = 101
    metric.verbose_logs = "runtime-only logs"
    metric.skipped = True

    cached_metric_data = Cache.get_metric_data(metric, cached_case)
    assert cached_metric_data is not None
    assert cached_metric_data.metric_data.name == "Configurable Custom"
    assert cached_metric_data.metric_data.score == 1.0


def test_opaque_embedding_config_field_misses_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _EmbeddingsConfigMetric(_FakeEmbeddings())
    )

    assert metric_configuration.embeddings == Cache._UNCACHEABLE_CACHE_VALUE

    assert (
        Cache.get_metric_data(
            _EmbeddingsConfigMetric(_FakeEmbeddings()),
            _cached_case_for(_EmbeddingsConfigMetric(_FakeEmbeddings())),
        )
        is None
    )


def test_string_model_name_is_fingerprinted_when_evaluation_model_is_not_stored():
    cached_case = _cached_case_for(
        _MetricWithModelWithoutEvaluationModel("model-a")
    )
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.evaluation_model is None
    assert metric_configuration.custom_parameters is not None
    assert metric_configuration.custom_parameters["model"].startswith("sha256:")
    assert "model-a" not in json.dumps(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            _MetricWithModelWithoutEvaluationModel("model-a"),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _MetricWithModelWithoutEvaluationModel("model-b"),
            cached_case,
        )
        is None
    )


def test_safe_direct_evaluation_model_string_hits_and_misses_by_value():
    cached_case = _cached_case_for(_MetricWithEvaluationModelString("model-a"))
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.evaluation_model == "model-a"
    assert (
        Cache.get_metric_data(
            _MetricWithEvaluationModelString("model-a"),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _MetricWithEvaluationModelString("model-b"),
            cached_case,
        )
        is None
    )


def test_initialized_native_model_objects_reuse_stable_evaluation_model_string():
    cached_case = _cached_case_for(_MetricWithInitializedNativeModel("model-a"))
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.evaluation_model == "model-a"
    assert metric_configuration.custom_parameters is not None
    assert metric_configuration.custom_parameters["model"].startswith("sha256:")
    assert "sk-test" not in json.dumps(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            _MetricWithInitializedNativeModel("model-a"),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _MetricWithInitializedNativeModel("model-b"),
            cached_case,
        )
        is None
    )


def test_native_model_cache_identity_includes_generation_settings():
    cached_case = _cached_case_for(
        AnswerRelevancyMetric(
            model=GPTModel(
                model="gpt-4o-mini",
                temperature=0,
                generation_kwargs={"top_p": 0.1},
                api_key="sk-test",
            ),
            async_mode=False,
        )
    )
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.custom_parameters is not None
    assert metric_configuration.custom_parameters["model"].startswith("sha256:")
    assert "sk-test" not in json.dumps(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            AnswerRelevancyMetric(
                model=GPTModel(
                    model="gpt-4o-mini",
                    temperature=0,
                    generation_kwargs={"top_p": 0.1},
                    api_key="sk-test",
                ),
                async_mode=False,
            ),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            AnswerRelevancyMetric(
                model=GPTModel(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    generation_kwargs={"top_p": 0.1},
                    api_key="sk-test",
                ),
                async_mode=False,
            ),
            cached_case,
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            AnswerRelevancyMetric(
                model=GPTModel(
                    model="gpt-4o-mini",
                    temperature=0,
                    generation_kwargs={"top_p": 0.9},
                    api_key="sk-test",
                ),
                async_mode=False,
            ),
            cached_case,
        )
        is None
    )


def test_native_model_cache_identity_includes_anthropic_max_tokens():
    cached_case = _cached_case_for(
        _MetricWithInjectedNativeModel(
            _NativeModelWithMaxTokens(
                model_name="claude-test",
                max_tokens=256,
            )
        )
    )
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.custom_parameters is not None
    assert metric_configuration.custom_parameters["model"].startswith("sha256:")
    assert (
        Cache.get_metric_data(
            _MetricWithInjectedNativeModel(
                _NativeModelWithMaxTokens(
                    model_name="claude-test",
                    max_tokens=256,
                )
            ),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _MetricWithInjectedNativeModel(
                _NativeModelWithMaxTokens(
                    model_name="claude-test",
                    max_tokens=1024,
                )
            ),
            cached_case,
        )
        is None
    )


def test_native_model_cache_identity_includes_gemini_vertex_backend_flag():
    cached_case = _cached_case_for(
        _MetricWithInjectedNativeModel(
            _NativeModelWithVertexBackend(
                model_name="gemini-test",
                use_vertexai=True,
            )
        )
    )
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.custom_parameters is not None
    assert metric_configuration.custom_parameters["model"].startswith("sha256:")
    assert (
        Cache.get_metric_data(
            _MetricWithInjectedNativeModel(
                _NativeModelWithVertexBackend(
                    model_name="gemini-test",
                    use_vertexai=True,
                )
            ),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _MetricWithInjectedNativeModel(
                _NativeModelWithVertexBackend(
                    model_name="gemini-test",
                    use_vertexai=False,
                )
            ),
            cached_case,
        )
        is None
    )


def test_real_initialize_model_metric_keeps_cache_hit_for_stable_string_model():
    cached_case = _cached_case_for(
        AnswerRelevancyMetric(
            model=GPTModel(model="gpt-4o-mini", api_key="sk-test"),
            async_mode=False,
        )
    )

    assert (
        Cache.get_metric_data(
            AnswerRelevancyMetric(
                model=GPTModel(model="gpt-4o-mini", api_key="sk-test"),
                async_mode=False,
            ),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            AnswerRelevancyMetric(
                model=GPTModel(model="gpt-4o", api_key="sk-test"),
                async_mode=False,
            ),
            cached_case,
        )
        is None
    )


def test_tool_correctness_metric_with_native_model_keeps_cache_hit_without_evaluation_model():
    cached_case = _cached_case_for(
        ToolCorrectnessMetric(
            model=GPTModel(model="gpt-4o-mini", api_key="sk-test"),
            async_mode=False,
        )
    )
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.evaluation_model is None
    assert metric_configuration.custom_parameters is not None
    assert metric_configuration.custom_parameters["model"].startswith("sha256:")
    assert "sk-test" not in json.dumps(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            ToolCorrectnessMetric(
                model=GPTModel(model="gpt-4o-mini", api_key="sk-test"),
                async_mode=False,
            ),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            ToolCorrectnessMetric(
                model=GPTModel(model="gpt-4o", api_key="sk-test"),
                async_mode=False,
            ),
            cached_case,
        )
        is None
    )


def test_hostile_native_model_dict_access_misses_cache_closed():
    metric = _MetricWithHostileNativeModel(_HostileNativeModel())
    metric_configuration = Cache.create_metric_configuration(metric)

    assert metric_configuration.evaluation_model == "hostile-native-model"
    assert metric_configuration.custom_parameters == {
        "model": Cache._UNCACHEABLE_CACHE_VALUE
    }
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_model_objects_miss_closed_even_when_names_match():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithModelWithoutEvaluationModel(
            _TunableNamedModel("model-a", temperature=0.0)
        )
    )

    assert metric_configuration.evaluation_model is None
    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["model"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _MetricWithModelWithoutEvaluationModel(
                _TunableNamedModel("model-a", temperature=1.0)
            ),
            _cached_case_for(
                _MetricWithModelWithoutEvaluationModel(
                    _TunableNamedModel("model-a", temperature=0.0)
                )
            ),
        )
        is None
    )


def test_derived_metric_model_objects_miss_closed_without_persisting_raw_value():
    metric = _MetricWithDerivedEvaluationModel(_NamedModel("model-a"))
    metric_configuration = Cache.create_metric_configuration(metric)

    assert (
        metric_configuration.evaluation_model == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["model"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "model-a" not in json.dumps(metric_configuration.model_dump())
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_model_name_fingerprinting_uses_passive_fields_without_method_calls():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithModelWithoutEvaluationModel(_ExplodingModelName("model-a"))
    )

    assert metric_configuration.evaluation_model is None
    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["model"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _MetricWithModelWithoutEvaluationModel(
                _ExplodingModelName("model-a")
            ),
            _cached_case_for(
                _MetricWithModelWithoutEvaluationModel(
                    _ExplodingModelName("model-a")
                )
            ),
        )
        is None
    )


def test_unidentified_model_objects_miss_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithModelWithoutEvaluationModel(_UnnamedModel())
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["model"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _MetricWithModelWithoutEvaluationModel(_UnnamedModel()),
            _cached_case_for(
                _MetricWithModelWithoutEvaluationModel(_UnnamedModel())
            ),
        )
        is None
    )


def test_sensitive_model_names_miss_cache_closed_without_persisting_raw_value():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithModelWithoutEvaluationModel(
            _SensitiveNamedModel("https://user:pass@example.com/model")
        )
    )

    assert metric_configuration.evaluation_model is None
    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["model"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "https://user:pass@example.com/model" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert (
        Cache.get_metric_data(
            _MetricWithModelWithoutEvaluationModel(
                _SensitiveNamedModel("https://user:pass@example.com/model")
            ),
            _cached_case_for(
                _MetricWithModelWithoutEvaluationModel(
                    _SensitiveNamedModel("https://user:pass@example.com/model")
                )
            ),
        )
        is None
    )


def test_sensitive_direct_evaluation_model_string_misses_closed():
    metric = _MetricWithEvaluationModelString(
        "https://user:pass@example.com/model"
    )
    metric_configuration = Cache.create_metric_configuration(metric)

    assert (
        metric_configuration.evaluation_model == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "https://user:pass@example.com/model" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_encoded_query_and_fragment_evaluation_model_secrets_miss_cache_closed():
    for evaluation_model in (
        "https://example.com/model?state=Bearer%20secret-token",
        "https://example.com/model#sig=secret-signature",
        "https://example.com/model#access_token=secret-token",
    ):
        metric = _MetricWithEvaluationModelString(evaluation_model)
        metric_configuration = Cache.create_metric_configuration(metric)

        assert (
            metric_configuration.evaluation_model
            == Cache._UNCACHEABLE_CACHE_VALUE
        )
        assert evaluation_model not in json.dumps(
            metric_configuration.model_dump()
        )
        assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_non_string_evaluation_model_misses_closed_without_str_call():
    metric = _MetricWithEvaluationModelObject(_ExplodingStringValue())
    metric_configuration = Cache.create_metric_configuration(metric)

    assert (
        metric_configuration.evaluation_model == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_free_form_metric_config_fields_are_fingerprinted_not_cached_raw():
    cached_case = _cached_case_for(
        _FreeFormConfigMetric(
            criteria="Judge helpfulness",
            evaluation_steps=["Check whether the answer is helpful"],
            assessment_questions=["Was the answer useful?"],
        )
    )
    metric_configuration = cached_case.cached_metrics_data[
        0
    ].metric_configuration

    assert metric_configuration.criteria.startswith("sha256:")
    assert metric_configuration.evaluation_steps.startswith("sha256:")
    assert metric_configuration.assessment_questions.startswith("sha256:")
    assert "Judge helpfulness" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert (
        Cache.get_metric_data(
            _FreeFormConfigMetric(
                criteria="Judge helpfulness",
                evaluation_steps=["Check whether the answer is helpful"],
                assessment_questions=["Was the answer useful?"],
            ),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _FreeFormConfigMetric(
                criteria="Judge harmfulness",
                evaluation_steps=["Check whether the answer is helpful"],
                assessment_questions=["Was the answer useful?"],
            ),
            cached_case,
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _FreeFormConfigMetric(
                criteria="Judge helpfulness",
                evaluation_steps=["Check whether the answer is concise"],
                assessment_questions=["Was the answer useful?"],
            ),
            cached_case,
        )
        is None
    )


def test_sensitive_free_form_metric_config_fields_miss_cache_closed():
    metric = _FreeFormConfigMetric(
        criteria="Bearer secret-token should not be written",
        evaluation_steps=["Check sig=secret-signature"],
        assessment_questions=["Does api_key=sk-secret appear?"],
    )
    metric_configuration = Cache.create_metric_configuration(metric)

    assert metric_configuration.criteria == Cache._UNCACHEABLE_CACHE_VALUE
    assert (
        metric_configuration.evaluation_steps == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        metric_configuration.assessment_questions
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    serialized = json.dumps(metric_configuration.model_dump())
    assert "secret-token" not in serialized
    assert "secret-signature" not in serialized
    assert "sk-secret" not in serialized
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_uncacheable_evaluation_steps_never_hit_cache_even_when_values_match():
    metric = _FreeFormConfigMetric(
        criteria="Judge helpfulness",
        evaluation_steps=["Check sig=secret-signature"],
        assessment_questions=["Was the answer useful?"],
    )
    cached_case = _cached_case_for(metric)

    assert (
        cached_case.cached_metrics_data[0].metric_configuration.evaluation_steps
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert Cache.get_metric_data(metric, cached_case) is None


def test_hostile_free_form_metric_config_property_misses_cache_closed():
    _HostileCriteriaMetric.criteria_accesses = 0
    metric = _HostileCriteriaMetric()
    metric_configuration = Cache.create_metric_configuration(metric)

    assert metric_configuration.criteria == Cache._UNCACHEABLE_CACHE_VALUE
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None
    assert _HostileCriteriaMetric.criteria_accesses == 0


def test_sensitive_scalar_metric_config_fields_miss_cache_closed():
    for field, value, secret in (
        ("threshold", "sk-threshold-secret", "sk-threshold-secret"),
        ("strict_mode", "Bearer strict-secret", "strict-secret"),
        ("include_reason", "token=reason-secret", "reason-secret"),
        ("n", "sig=n-secret", "n-secret"),
    ):
        metric = _ScalarConfigMetric(field, value)
        metric_configuration = Cache.create_metric_configuration(metric)

        assert (
            getattr(metric_configuration, field)
            == Cache._UNCACHEABLE_CACHE_VALUE
        )
        assert secret not in json.dumps(metric_configuration.model_dump())
        cached_case = CachedTestCase(
            cached_metrics_data=[
                CachedMetricData(
                    metric_data=MetricData(
                        name=metric.__name__,
                        threshold=0.5,
                        success=True,
                        score=1.0,
                    ),
                    metric_configuration=metric_configuration,
                )
            ]
        )
        assert Cache.get_metric_data(metric, cached_case) is None


def test_sensitive_derived_model_names_miss_cache_closed_without_persisting_raw_value():
    metric = _MetricWithDerivedEvaluationModel(
        _SensitiveNamedModel("https://user:pass@example.com/model")
    )
    metric_configuration = Cache.create_metric_configuration(metric)

    assert (
        metric_configuration.evaluation_model == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["model"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "https://user:pass@example.com/model" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_sensitive_private_model_names_miss_cache_closed_without_persisting_raw_value():
    metric = _MetricWithPrivateDerivedEvaluationModel(
        _SensitiveNamedModel("https://user:pass@example.com/model")
    )
    metric_configuration = Cache.create_metric_configuration(metric)

    assert (
        metric_configuration.evaluation_model == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["model"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "https://user:pass@example.com/model" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_metric_constructor_parameters_are_stored_in_cache_configuration():
    metric_configuration = Cache.create_metric_configuration(
        _ConfigurableMetric(
            rubric="safety",
            weights={"harm": 1.0, "accuracy": 0.5},
        )
    )

    custom_parameters = metric_configuration.custom_parameters
    assert custom_parameters is not None
    assert set(custom_parameters) == {
        "rubric",
        "steps",
        "weights",
    }
    assert all(
        value.startswith("sha256:") for value in custom_parameters.values()
    )
    assert "safety" not in set(custom_parameters.values())


def test_no_dict_metric_constructor_parameters_miss_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _NoDictMetric("safety")
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["rubric"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _NoDictMetric("safety"),
            _cached_case_for(_NoDictMetric("safety")),
        )
        is None
    )


def test_hostile_metric_dict_access_misses_cache_closed():
    metric = _HostileDictMetric(rubric="safety", weights={"harm": 1.0})
    metric_configuration = Cache.create_metric_configuration(metric)

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["rubric"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        metric_configuration.custom_parameters["weights"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert Cache.get_metric_data(metric, _cached_case_for(metric)) is None


def test_metric_with_no_explicit_init_has_no_constructor_state_to_compare():
    cached_case = _cached_case_for(_NoInitMetric())

    assert (
        Cache.create_metric_configuration(_NoInitMetric()).custom_parameters
        is None
    )
    assert Cache.get_metric_data(_NoInitMetric(), cached_case) is not None


def test_slots_metric_with_unrecoverable_model_misses_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _SlotsModelMetric("model-a")
    )

    assert metric_configuration.custom_parameters == {
        "model": Cache._UNCACHEABLE_CACHE_VALUE
    }
    assert (
        Cache.get_metric_data(
            _SlotsModelMetric("model-a"),
            _cached_case_for(_SlotsModelMetric("model-a")),
        )
        is None
    )


def test_signature_introspection_failure_misses_cache_closed(monkeypatch):
    original_signature = cache_identity_module.inspect.signature

    def raise_for_signatureless_metric(value, *args, **kwargs):
        if value is _SignaturelessMetric.__init__:
            raise ValueError("signature unavailable")
        return original_signature(value, *args, **kwargs)

    monkeypatch.setattr(
        cache_identity_module.inspect,
        "signature",
        raise_for_signatureless_metric,
    )
    metric_configuration = Cache.create_metric_configuration(
        _SignaturelessMetric("safety")
    )

    assert metric_configuration.custom_parameters == {
        "__signature__": Cache._UNCACHEABLE_CACHE_VALUE
    }
    assert (
        Cache.get_metric_data(
            _SignaturelessMetric("safety"),
            _cached_case_for(_SignaturelessMetric("safety")),
        )
        is None
    )


def test_pydantic_schema_class_constructor_parameter_includes_field_shape():
    first_schema = create_model(
        "ReusableSchema",
        value=(str, ...),
        __module__=__name__,
    )
    second_schema = create_model(
        "ReusableSchema",
        value=(int, ...),
        __module__=__name__,
    )
    cached_case = _cached_case_for(_SchemaClassMetric(first_schema))

    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(first_schema),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(second_schema),
            cached_case,
        )
        is None
    )


def test_pydantic_v1_config_affects_schema_identity():
    pydantic_v1 = pytest.importorskip("pydantic.v1")

    class AllowConfig:
        extra = "allow"

    class ForbidConfig:
        extra = "forbid"

    first_schema = pydantic_v1.create_model(
        "ReusableV1ConfigSchema",
        value=(str, ...),
        __module__=__name__,
        __config__=AllowConfig,
    )
    second_schema = pydantic_v1.create_model(
        "ReusableV1ConfigSchema",
        value=(str, ...),
        __module__=__name__,
        __config__=ForbidConfig,
    )
    cached_case = _cached_case_for(_SchemaClassMetric(first_schema))

    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(first_schema),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(second_schema),
            cached_case,
        )
        is None
    )


def test_pydantic_schema_free_form_metadata_misses_closed_without_persisting_raw_value():
    schema = create_model(
        "DescSchema",
        value=(
            str,
            Field(
                default="same",
                title="Answer",
                description="sk-should-not-be-written",
                examples=["bearer-should-not-be-written"],
                json_schema_extra={"format": "secret-should-not-be-written"},
            ),
        ),
        __module__=__name__,
    )
    metric_configuration = Cache.create_metric_configuration(
        _SchemaClassMetric(schema)
    )
    payload = json.dumps(metric_configuration.model_dump())

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "sk-should-not-be-written" not in payload
    assert "bearer-should-not-be-written" not in payload
    assert "secret-should-not-be-written" not in payload
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(schema),
            _cached_case_for(_SchemaClassMetric(schema)),
        )
        is None
    )


def test_pydantic_schema_callable_json_schema_extra_misses_cache_closed():
    def mutate_schema(schema):
        schema["x-mode"] = "strict"

    schema = create_model(
        "CallableSchema",
        value=(str, Field(default="same", json_schema_extra=mutate_schema)),
        __module__=__name__,
    )
    metric_configuration = Cache.create_metric_configuration(
        _SchemaClassMetric(schema)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(schema),
            _cached_case_for(_SchemaClassMetric(schema)),
        )
        is None
    )


def test_pydantic_schema_alias_choices_affect_cache_identity():
    if AliasChoices is None:
        pytest.skip("AliasChoices requires Pydantic v2")

    first_schema = create_model(
        "AliasSchema",
        value=(
            str,
            Field(
                default="same",
                validation_alias=AliasChoices("answer", "result"),
            ),
        ),
        __module__=__name__,
    )
    second_schema = create_model(
        "AliasSchema",
        value=(
            str,
            Field(
                default="same",
                validation_alias=AliasChoices("output", "result"),
            ),
        ),
        __module__=__name__,
    )
    cached_case = _cached_case_for(_SchemaClassMetric(first_schema))

    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(first_schema),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(second_schema),
            cached_case,
        )
        is None
    )


def test_slot_backed_pydantic_alias_helpers_are_cacheable():
    alias_choices = _SlotBackedAliasChoices(["answer", "result"])
    alias_path = _SlotBackedAliasPath(["payload", "answer"])

    assert Cache._pydantic_alias_candidates(alias_choices) == [
        "answer",
        "result",
    ]
    assert Cache._pydantic_alias_identity(alias_choices) == {
        "choices": ["answer", "result"]
    }
    assert Cache._pydantic_alias_candidates(alias_path) == [
        "payload",
        "answer",
    ]
    assert Cache._pydantic_alias_identity(alias_path) == {
        "path": ["payload", "answer"]
    }


def test_pydantic_instance_sensitive_alias_values_miss_cache_closed():
    schema = create_model(
        "AliasedInstanceSchema",
        value=(str, Field(alias="Authorization")),
        __module__=__name__,
    )
    instance = schema(Authorization="first-secret")
    metric_configuration = Cache.create_metric_configuration(
        _SchemaMetric(instance)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "first-secret" not in json.dumps(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            _SchemaMetric(instance),
            _cached_case_for(_SchemaMetric(instance)),
        )
        is None
    )


def test_pydantic_schema_secret_defaults_miss_cache_closed():
    direct_secret_schema = create_model(
        "DirectSecretSchema",
        api_key=(str, "sk-should-not-be-written"),
        __module__=__name__,
    )
    aliased_secret_schema = create_model(
        "AliasedSecretSchema",
        value=(
            str,
            Field(
                "bearer-should-not-be-written",
                alias="Authorization",
            ),
        ),
        __module__=__name__,
    )
    schemas = [
        (direct_secret_schema, "sk-should-not-be-written"),
        (aliased_secret_schema, "bearer-should-not-be-written"),
    ]
    if AliasChoices is not None:
        alias_choices_secret_schema = create_model(
            "AliasChoicesSecretSchema",
            value=(
                str,
                Field(
                    "choice-should-not-be-written",
                    validation_alias=AliasChoices("Authorization", "trace_id"),
                ),
            ),
            __module__=__name__,
        )
        schemas.append(
            (alias_choices_secret_schema, "choice-should-not-be-written")
        )

    for schema, secret in schemas:
        metric_configuration = Cache.create_metric_configuration(
            _SchemaClassMetric(schema)
        )

        assert metric_configuration.custom_parameters is not None
        assert (
            metric_configuration.custom_parameters["expected_schema"]
            == Cache._UNCACHEABLE_CACHE_VALUE
        )
        assert secret not in json.dumps(metric_configuration.model_dump())
        assert (
            Cache.get_metric_data(
                _SchemaClassMetric(schema),
                _cached_case_for(_SchemaClassMetric(schema)),
            )
            is None
        )


def test_pydantic_private_attributes_miss_cache_closed():
    schema = _SchemaWithPrivateState(value="same")
    schema._mode = "strict"
    metric_configuration = Cache.create_metric_configuration(
        _SchemaMetric(schema)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaMetric(schema),
            _cached_case_for(_SchemaMetric(schema)),
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_SchemaWithPrivateState),
            _cached_case_for(_SchemaClassMetric(_SchemaWithPrivateState)),
        )
        is None
    )


def test_pydantic_custom_getattribute_misses_cache_closed():
    schema = _SchemaWithHostileGetattribute(value="same")
    metric_configuration = Cache.create_metric_configuration(
        _SchemaMetric(schema)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaMetric(schema),
            _cached_case_for(_SchemaMetric(schema)),
        )
        is None
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_SchemaWithHostileGetattribute),
            _cached_case_for(
                _SchemaClassMetric(_SchemaWithHostileGetattribute)
            ),
        )
        is None
    )


def test_pydantic_custom_model_config_mapping_misses_cache_closed(monkeypatch):
    monkeypatch.setattr(
        _SchemaWithHostileConfig,
        "model_config",
        _HostileConfigMapping(extra="allow"),
    )
    metric_configuration = Cache.create_metric_configuration(
        _SchemaClassMetric(_SchemaWithHostileConfig)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_SchemaWithHostileConfig),
            _cached_case_for(_SchemaClassMetric(_SchemaWithHostileConfig)),
        )
        is None
    )


def test_pydantic_custom_field_object_misses_cache_closed(monkeypatch):
    monkeypatch.setattr(
        _SchemaWithHostileField,
        "model_fields",
        {"value": _HostilePydanticField()},
    )
    metric_configuration = Cache.create_metric_configuration(
        _SchemaClassMetric(_SchemaWithHostileField)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_SchemaWithHostileField),
            _cached_case_for(_SchemaClassMetric(_SchemaWithHostileField)),
        )
        is None
    )


def test_pydantic_field_map_with_custom_key_misses_cache_without_str_call(
    monkeypatch,
):
    monkeypatch.setattr(
        _SchemaWithHostileField,
        "model_fields",
        {_HostileFieldKey(): _SchemaWithHostileField.model_fields["value"]},
    )
    metric_configuration = Cache.create_metric_configuration(
        _SchemaClassMetric(_SchemaWithHostileField)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_SchemaWithHostileField),
            _cached_case_for(_SchemaClassMetric(_SchemaWithHostileField)),
        )
        is None
    )


def test_pydantic_schema_with_validator_misses_cache_closed():
    cached_case = _cached_case_for(_SchemaClassMetric(_ValidatedSchema))

    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_ValidatedSchema),
            cached_case,
        )
        is None
    )


def test_pydantic_v1_root_validator_misses_cache_closed():
    pydantic_v1 = pytest.importorskip("pydantic.v1")

    class RootValidatedSchema(pydantic_v1.BaseModel):
        value: int

        @pydantic_v1.root_validator(pre=True)
        def validate_root(cls, values):
            return values

    metric_configuration = Cache.create_metric_configuration(
        _SchemaClassMetric(RootValidatedSchema)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(RootValidatedSchema),
            _cached_case_for(_SchemaClassMetric(RootValidatedSchema)),
        )
        is None
    )


def test_pydantic_schema_with_lambda_default_factory_misses_cache_closed():
    schema = create_model(
        "FactorySchema",
        value=(str, Field(default_factory=lambda: "first")),
        __module__=__name__,
    )
    cached_case = _cached_case_for(_SchemaClassMetric(schema))

    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(schema),
            cached_case,
        )
        is None
    )


def test_pydantic_schema_with_named_default_factory_misses_cache_closed():
    schema = create_model(
        "NamedFactorySchema",
        value=(str, Field(default_factory=_named_schema_default)),
        __module__=__name__,
    )
    cached_case = _cached_case_for(_SchemaClassMetric(schema))

    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(schema),
            cached_case,
        )
        is None
    )


def test_dag_constructor_parameter_uses_structural_identity():
    cached_case = _cached_case_for(
        _DAGBackedMetric(_single_node_dag("Extract safety issues"))
    )

    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(_single_node_dag("Extract safety issues")),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(_single_node_dag("Extract factual issues")),
            cached_case,
        )
        is None
    )


def test_dag_constructor_parameter_preserves_shared_node_topology():
    cached_case = _cached_case_for(_DAGBackedMetric(_shared_child_dag(True)))

    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(_shared_child_dag(True)),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(_shared_child_dag(False)),
            cached_case,
        )
        is None
    )


def test_dag_constructor_parameter_misses_closed_on_cycles():
    root = TaskNode(
        instructions="Root",
        output_label="root",
        children=[],
        evaluation_params=[SingleTurnParams.INPUT],
    )
    root.children = [root]
    dag = DeepAcyclicGraph(root_nodes=[root])
    metric_configuration = Cache.create_metric_configuration(
        _DAGBackedMetric(dag)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["dag"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(dag),
            _cached_case_for(_DAGBackedMetric(dag)),
        )
        is None
    )


def test_dag_root_nodes_custom_iterable_misses_cache_without_iteration():
    root = TaskNode(
        instructions="Root",
        output_label="root",
        children=[],
        evaluation_params=[SingleTurnParams.INPUT],
    )
    dag = DeepAcyclicGraph(root_nodes=[root])
    dag.root_nodes = _HostileIterable()
    metric_configuration = Cache.create_metric_configuration(
        _DAGBackedMetric(dag)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["dag"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(dag),
            _cached_case_for(_DAGBackedMetric(dag)),
        )
        is None
    )


def test_dag_children_custom_iterable_misses_cache_without_iteration():
    root = TaskNode(
        instructions="Root",
        output_label="root",
        children=[],
        evaluation_params=[SingleTurnParams.INPUT],
    )
    root.children = _HostileIterable()
    dag = DeepAcyclicGraph(root_nodes=[root])
    metric_configuration = Cache.create_metric_configuration(
        _DAGBackedMetric(dag)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["dag"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(dag),
            _cached_case_for(_DAGBackedMetric(dag)),
        )
        is None
    )


def test_dag_metric_child_constructor_parameter_invalidates_cache():
    cached_case = _cached_case_for(
        _DAGBackedMetric(_metric_child_dag("strict"))
    )

    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(_metric_child_dag("strict")),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(_metric_child_dag("relaxed")),
            cached_case,
        )
        is None
    )


def test_dag_metric_child_opaque_constructor_parameter_misses_cache_closed():
    cached_case = _cached_case_for(
        _DAGBackedMetric(_metric_child_dag(object()))
    )

    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(_metric_child_dag(object())),
            cached_case,
        )
        is None
    )


def test_dag_metric_child_uncacheable_config_field_misses_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _DAGBackedMetric(
            _metric_child_with_evaluation_model_dag(
                "https://user:pass@example.com/model"
            )
        )
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["dag"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "https://user:pass@example.com/model" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(
                _metric_child_with_evaluation_model_dag(
                    "https://user:pass@example.com/model"
                )
            ),
            _cached_case_for(
                _DAGBackedMetric(
                    _metric_child_with_evaluation_model_dag(
                        "https://user:pass@example.com/model"
                    )
                )
            ),
        )
        is None
    )


def test_dag_with_unknown_node_type_misses_cache_closed():
    dag = DeepAcyclicGraph(
        root_nodes=[
            _CustomTaskNode(
                instructions="Root",
                output_label="root",
                children=[],
                evaluation_params=[SingleTurnParams.INPUT],
            )
        ]
    )
    metric_configuration = Cache.create_metric_configuration(
        _DAGBackedMetric(dag)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["dag"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(dag),
            _cached_case_for(_DAGBackedMetric(dag)),
        )
        is None
    )


def test_dag_with_unknown_child_node_misses_cache_closed():
    dag = DeepAcyclicGraph(
        root_nodes=[
            TaskNode(
                instructions="Root",
                output_label="root",
                children=[
                    _CustomTaskNode(
                        instructions="Unsupported child",
                        output_label="child",
                        children=[],
                        evaluation_params=[SingleTurnParams.INPUT],
                    )
                ],
                evaluation_params=[SingleTurnParams.INPUT],
            )
        ]
    )
    metric_configuration = Cache.create_metric_configuration(
        _DAGBackedMetric(dag)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["dag"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _DAGBackedMetric(dag),
            _cached_case_for(_DAGBackedMetric(dag)),
        )
        is None
    )


def test_pydantic_schema_constructor_parameter_includes_schema_class_identity():
    cached_case = _cached_case_for(
        _SchemaMetric(_FirstSchema(value="same-value"))
    )

    assert (
        Cache.get_metric_data(
            _SchemaMetric(_FirstSchema(value="same-value")),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _SchemaMetric(_SecondSchema(value="same-value")),
            cached_case,
        )
        is None
    )


def test_pydantic_literal_annotation_values_preserve_type_identity():
    cached_case = _cached_case_for(_SchemaClassMetric(_LiteralIntSchema))

    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_LiteralIntSchema),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(_LiteralStringSchema),
            cached_case,
        )
        is None
    )


def test_pydantic_literal_annotation_opaque_values_miss_closed_without_str_call():
    schema = create_model(
        "OpaqueLiteralSchema",
        value=(Literal[_ExplodingStringValue()], ...),
        __module__=__name__,
    )
    metric_configuration = Cache.create_metric_configuration(
        _SchemaClassMetric(schema)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["expected_schema"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _SchemaClassMetric(schema),
            _cached_case_for(_SchemaClassMetric(schema)),
        )
        is None
    )


def test_sensitive_constructor_parameters_are_marked_uncacheable_not_cached_raw():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithSecretParameter(
            rubric="safety",
            api_key="sk-should-not-be-written",
            client=object(),
        )
    )

    custom_parameters = metric_configuration.custom_parameters
    assert custom_parameters is not None
    assert custom_parameters == {
        "api_key": Cache._UNCACHEABLE_CACHE_VALUE,
        "client": Cache._UNCACHEABLE_CACHE_VALUE,
        "rubric": custom_parameters["rubric"],
    }
    assert custom_parameters["rubric"].startswith("sha256:")
    assert "sk-should-not-be-written" not in str(
        metric_configuration.model_dump()
    )


def test_nested_sensitive_constructor_values_are_marked_uncacheable():
    first = Cache.create_metric_configuration(
        _MetricWithNestedConfigParameter(
            config={
                "headers": {
                    "Authorization": "Bearer first-secret",
                    "X-Trace": "stable",
                },
                "mode": "strict",
            }
        )
    )
    second = Cache.create_metric_configuration(
        _MetricWithNestedConfigParameter(
            config={
                "headers": {
                    "Authorization": "Bearer second-secret",
                    "X-Trace": "stable",
                },
                "mode": "strict",
            }
        )
    )

    assert "first-secret" not in str(first.model_dump())
    assert "second-secret" not in str(second.model_dump())
    assert first.custom_parameters is not None
    assert second.custom_parameters is not None
    assert first.custom_parameters["config"] == Cache._UNCACHEABLE_CACHE_VALUE
    assert second.custom_parameters["config"] == Cache._UNCACHEABLE_CACHE_VALUE
    assert (
        Cache.get_metric_data(
            _MetricWithNestedConfigParameter(
                config={
                    "headers": {
                        "Authorization": "Bearer first-secret",
                        "X-Trace": "stable",
                    },
                    "mode": "strict",
                }
            ),
            _cached_case_for(
                _MetricWithNestedConfigParameter(
                    config={
                        "headers": {
                            "Authorization": "Bearer first-secret",
                            "X-Trace": "stable",
                        },
                        "mode": "strict",
                    }
                )
            ),
        )
        is None
    )


def test_secret_shaped_values_under_generic_names_miss_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        PatternMatchMetric(pattern="sk-live-secret", ignore_case=False)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["pattern"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "sk-live-secret" not in json.dumps(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            PatternMatchMetric(pattern="sk-live-secret", ignore_case=False),
            _cached_case_for(
                PatternMatchMetric(pattern="sk-live-secret", ignore_case=False)
            ),
        )
        is None
    )


def test_custom_header_secret_values_under_generic_keys_miss_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithNestedConfigParameter(
            config={
                "headers": {
                    "X-Session": "Bearer secret-session-value",
                    "X-Trace": "stable",
                },
                "mode": "strict",
            }
        )
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["config"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "secret-session-value" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert (
        Cache.get_metric_data(
            _MetricWithNestedConfigParameter(
                config={
                    "headers": {
                        "X-Session": "Bearer secret-session-value",
                        "X-Trace": "stable",
                    },
                    "mode": "strict",
                }
            ),
            _cached_case_for(
                _MetricWithNestedConfigParameter(
                    config={
                        "headers": {
                            "X-Session": "Bearer secret-session-value",
                            "X-Trace": "stable",
                        },
                        "mode": "strict",
                    }
                )
            ),
        )
        is None
    )


def test_opaque_constructor_value_with_hostile_class_access_misses_closed():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithNestedConfigParameter(config=_ExplodingClassValue())
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["config"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _MetricWithNestedConfigParameter(config=_ExplodingClassValue()),
            _cached_case_for(
                _MetricWithNestedConfigParameter(config=_ExplodingClassValue())
            ),
        )
        is None
    )


def test_custom_container_constructor_values_miss_closed_without_protocol_calls():
    for value in (
        _ExplodingDict({"mode": "strict"}),
        _ExplodingList(["strict"]),
    ):
        metric_configuration = Cache.create_metric_configuration(
            _MetricWithNestedConfigParameter(config=value)
        )

        assert metric_configuration.custom_parameters is not None
        assert (
            metric_configuration.custom_parameters["config"]
            == Cache._UNCACHEABLE_CACHE_VALUE
        )
        assert (
            Cache.get_metric_data(
                _MetricWithNestedConfigParameter(config=value),
                _cached_case_for(
                    _MetricWithNestedConfigParameter(config=value)
                ),
            )
            is None
        )


def test_common_secret_name_values_are_marked_uncacheable():
    for name in (
        "openaiApiKey",
        "auth_token",
        "github_token",
        "api_token",
        "access_token",
        "sessionToken",
        "bearer_token",
        "jwt_token",
        "aws_access_key_id",
        "Proxy-Authorization",
        "Set-Cookie",
        "clientSecret",
        "key",
    ):
        first = Cache._normalize_cache_parameter(name, "first-secret", set())
        second = Cache._normalize_cache_parameter(name, "second-secret", set())

        assert first == {"__deepeval_uncacheable__": "sensitive"}
        assert second == {"__deepeval_uncacheable__": "sensitive"}
        assert "first-secret" not in str(first)
        assert "second-secret" not in str(second)


def test_malformed_url_like_constructor_value_misses_cache_closed():
    for endpoint in (
        "http://[bad",
        "https://example.com:abc/path",
        "https://exa mple.com/path",
    ):
        normalized = Cache._normalize_cache_parameter(
            "endpoint", endpoint, set()
        )

        assert normalized == {"__deepeval_uncacheable__": "sensitive"}
        metric_configuration = Cache.create_metric_configuration(
            _EndpointMetric(endpoint)
        )
        assert metric_configuration.custom_parameters is not None
        assert (
            metric_configuration.custom_parameters["endpoint"]
            == Cache._UNCACHEABLE_CACHE_VALUE
        )
        assert endpoint not in json.dumps(metric_configuration.model_dump())
        assert (
            Cache.get_metric_data(
                _EndpointMetric(endpoint),
                _cached_case_for(_EndpointMetric(endpoint)),
            )
            is None
        )


def test_connection_secret_names_and_signed_urls_are_marked_uncacheable():
    for name in (
        "connection_string",
        "conn_str",
        "dsn",
        "database_url",
        "broker_url",
        "jdbc_url",
    ):
        normalized = Cache._normalize_cache_parameter(
            name, "postgres://user:pass@example.com/db", set()
        )

        assert normalized == {"__deepeval_uncacheable__": "sensitive"}

    signed_url = Cache._normalize_cache_parameter(
        "endpoint",
        "https://example.com/object?X-Amz-Signature=abc123",
        set(),
    )

    assert signed_url == {"__deepeval_uncacheable__": "sensitive"}

    metric_configuration = Cache.create_metric_configuration(
        _EndpointMetric("https://example.com/object?X-Amz-Signature=abc123")
    )
    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["endpoint"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "X-Amz-Signature" not in json.dumps(
        metric_configuration.model_dump()
    )
    assert (
        Cache.get_metric_data(
            _EndpointMetric(
                "https://example.com/object?X-Amz-Signature=abc123"
            ),
            _cached_case_for(
                _EndpointMetric(
                    "https://example.com/object?X-Amz-Signature=abc123"
                )
            ),
        )
        is None
    )


def test_dsn_and_token_bearing_url_variants_miss_cache_closed():
    for endpoint in (
        "postgres:pass@example.com/db",
        "https://example.com/object#access_token=abc123",
        "https://example.com/download/token/abc123",
        "https://example.com/download/token%2Fabc123",
        "https://example.com/download/X-Amz-Signature%2Fabc123",
        "https://example.com/object?state=Bearer%2520secret-token",
        f"https://example.com/object?state={_encode_repeatedly('Bearer secret-token')}",
        "//user:pass@example.com/path",
        quote("//user:pass@example.com/path", safe=""),
        _encode_repeatedly("//user:pass@example.com/path", passes=2),
        r"C:\downloads\token\abc123",
        r"\server\share\sig\abc123",
        r"\\server\share\sig\abc123",
        _encode_repeatedly(r"C:\downloads\token\abc123"),
        _encode_repeatedly(r"\\server\share\sig\abc123"),
        "/download?sig=abc123",
        "/download#access_token=abc123",
        "/download%2Ftoken%2Fabc123",
        "/download%2Fsig%2Fabc123",
        "/download%252Ftoken%252Fabc123",
        "/download%252Fsig%252Fabc123",
        _encode_repeatedly("/download/token/abc123"),
        _encode_repeatedly("/download/sig/abc123"),
        _encode_repeatedly("/download/X-Amz-Signature=abc123"),
    ):
        normalized = Cache._normalize_cache_parameter(
            "endpoint", endpoint, set()
        )

        assert normalized == {"__deepeval_uncacheable__": "sensitive"}
        metric_configuration = Cache.create_metric_configuration(
            _EndpointMetric(endpoint)
        )
        assert metric_configuration.custom_parameters is not None
        assert (
            metric_configuration.custom_parameters["endpoint"]
            == Cache._UNCACHEABLE_CACHE_VALUE
        )
        assert endpoint not in json.dumps(metric_configuration.model_dump())
        assert (
            Cache.get_metric_data(
                _EndpointMetric(endpoint),
                _cached_case_for(_EndpointMetric(endpoint)),
            )
            is None
        )


def test_top_level_sensitive_constructor_value_misses_cache_closed():
    cached_case = _cached_case_for(
        _MetricWithSecretParameter(
            rubric="safety",
            api_key="same-secret",
        )
    )

    assert (
        Cache.get_metric_data(
            _MetricWithSecretParameter(
                rubric="safety",
                api_key="same-secret",
            ),
            cached_case,
        )
        is None
    )


def test_underscore_prefixed_sensitive_constructor_value_misses_cache_closed():
    metric_configuration = Cache.create_metric_configuration(
        _MetricWithUnderscoreSecretParameter(
            rubric="safety",
            _access_token="secret-token",
        )
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["_access_token"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert "secret-token" not in str(metric_configuration.model_dump())
    assert (
        Cache.get_metric_data(
            _MetricWithUnderscoreSecretParameter(
                rubric="safety",
                _access_token="secret-token",
            ),
            _cached_case_for(
                _MetricWithUnderscoreSecretParameter(
                    rubric="safety",
                    _access_token="secret-token",
                )
            ),
        )
        is None
    )


def test_tokenizer_name_is_not_treated_as_a_secret_parameter():
    assert (
        Cache._normalize_cache_parameter("tokenizer_name", "cl100k_base", set())
        == "cl100k_base"
    )


def test_internal_constructor_parameters_do_not_invalidate_cache():
    first = Cache.create_metric_configuration(
        _TelemetryMetric(rubric="safety", _track=True)
    )
    second = Cache.create_metric_configuration(
        _TelemetryMetric(rubric="safety", _track=False)
    )

    assert first.custom_parameters is not None
    assert first.custom_parameters == second.custom_parameters
    assert "_track" not in first.custom_parameters


def test_large_constructor_parameter_exceeds_budget_without_traversal():
    value = list(range(Cache._MAX_CACHE_PARAMETER_COLLECTION_ITEMS + 1))

    normalized = Cache._normalize_cache_parameter("config", value, set())

    assert normalized["__deepeval_uncacheable__"] == "budget-exceeded"
    assert (
        Cache._cache_parameter_fingerprint("config", value)
        == Cache._UNCACHEABLE_CACHE_VALUE
    )


def test_semantic_tokenizer_constructor_parameter_invalidates_cache():
    cached_case = _cached_case_for(_TokenizerMetric(tokenizer_name="cl100k"))

    assert (
        Cache.get_metric_data(
            _TokenizerMetric(tokenizer_name="o200k"),
            cached_case,
        )
        is None
    )


def test_private_backed_constructor_parameter_invalidates_cache_without_property_access():
    cached_case = _cached_case_for(
        _PropertyBackedMetric(config={"mode": "strict"})
    )

    assert (
        Cache.get_metric_data(
            _PropertyBackedMetric(config={"mode": "strict"}),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _PropertyBackedMetric(config={"mode": "relaxed"}),
            cached_case,
        )
        is None
    )


def test_dataclass_constructor_parameter_with_cycle_misses_cache_closed_without_asdict():
    strict_config = _CyclicDataclassConfig(mode="strict")
    strict_config.next_config = strict_config
    metric_configuration = Cache.create_metric_configuration(
        _DataclassConfigMetric(strict_config)
    )

    assert metric_configuration.custom_parameters is not None
    assert (
        metric_configuration.custom_parameters["config"]
        == Cache._UNCACHEABLE_CACHE_VALUE
    )
    assert (
        Cache.get_metric_data(
            _DataclassConfigMetric(strict_config),
            _cached_case_for(_DataclassConfigMetric(strict_config)),
        )
        is None
    )


def test_pydantic_constructor_parameter_uses_raw_fields_without_dump_hooks():
    cached_case = _cached_case_for(
        _ExplodingModelMetric(_ModelWithExplodingDump(value="stable"))
    )

    assert (
        Cache.get_metric_data(
            _ExplodingModelMetric(_ModelWithExplodingDump(value="stable")),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _ExplodingModelMetric(_ModelWithExplodingDump(value="changed")),
            cached_case,
        )
        is None
    )


def test_legacy_cache_configuration_without_custom_parameters_loads_safely():
    legacy_configuration = MetricConfiguration(
        threshold=1.0,
        strict_mode=False,
        include_reason=False,
    )

    assert legacy_configuration.custom_parameters is None
    assert Cache.same_metric_configs(ExactMatchMetric(), legacy_configuration)
    assert not Cache.same_metric_configs(
        PatternMatchMetric(pattern=r"expected", ignore_case=False),
        legacy_configuration,
    )


def test_current_cache_payload_round_trips_custom_parameters(tmp_path):
    cache_file = tmp_path / "cache.json"
    cached_run = CachedTestRun(
        test_cases_lookup_map={
            "case": _cached_case_for(
                _ConfigurableMetric(
                    rubric="safety",
                    weights={"harm": 1.0, "accuracy": 0.5},
                )
            )
        }
    )

    with cache_file.open("w") as file:
        cached_run.save(file)

    loaded_run = CachedTestRun.load(json.loads(cache_file.read_text()))
    cached_case = loaded_run.test_cases_lookup_map["case"]

    assert (
        Cache.get_metric_data(
            _ConfigurableMetric(
                rubric="safety",
                weights={"harm": 1.0, "accuracy": 0.5},
            ),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _ConfigurableMetric(
                rubric="quality",
                weights={"harm": 1.0, "accuracy": 0.5},
            ),
            cached_case,
        )
        is None
    )


def test_current_cache_payload_round_trips_language_config(tmp_path):
    cache_file = tmp_path / "cache.json"
    cached_run = CachedTestRun(
        test_cases_lookup_map={
            "case": _cached_case_for(_MetricWithLanguage(language="en"))
        }
    )

    with cache_file.open("w") as file:
        cached_run.save(file)

    loaded_payload = json.loads(cache_file.read_text())
    cached_configuration = loaded_payload["test_cases_lookup_map"]["case"][
        "cached_metrics_data"
    ][0]["metric_configuration"]

    assert cached_configuration["language"] == "en"

    loaded_run = CachedTestRun.load(loaded_payload)
    cached_case = loaded_run.test_cases_lookup_map["case"]

    assert (
        Cache.get_metric_data(
            _MetricWithLanguage(language="en"),
            cached_case,
        )
        is not None
    )
    assert (
        Cache.get_metric_data(
            _MetricWithLanguage(language="es"),
            cached_case,
        )
        is None
    )


def test_legacy_cache_payload_without_custom_parameters_loads_safely():
    cached_run = CachedTestRun.load(
        {
            "test_cases_lookup_map": {
                "case": {
                    "cached_metrics_data": [
                        {
                            "metric_data": {
                                "name": "Exact Match",
                                "threshold": 1.0,
                                "success": True,
                                "score": 1.0,
                            },
                            "metric_configuration": {
                                "threshold": 1.0,
                                "strict_mode": False,
                                "include_reason": False,
                            },
                        },
                        {
                            "metric_data": {
                                "name": "Pattern Match",
                                "threshold": 1.0,
                                "success": True,
                                "score": 1.0,
                            },
                            "metric_configuration": {
                                "threshold": 1.0,
                                "strict_mode": False,
                                "include_reason": False,
                            },
                        },
                    ]
                }
            }
        }
    )
    cached_case = cached_run.test_cases_lookup_map["case"]

    cached_exact = Cache.get_metric_data(ExactMatchMetric(), cached_case)
    assert cached_exact is not None
    assert cached_exact.metric_data.name == "Exact Match"

    assert (
        Cache.get_metric_data(
            PatternMatchMetric(pattern=r"expected", ignore_case=False),
            cached_case,
        )
        is None
    )
