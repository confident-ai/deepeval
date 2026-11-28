import json
import pytest
from pydantic import BaseModel, ConfigDict

from deepeval.metrics import GEval, JsonCorrectnessMetric
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams
from deepeval.test_run.api import MetricData
from deepeval.test_run.cache import (
    Cache,
    CachedMetricData,
    CachedTestCase,
    CachedTestRun,
    MetricConfiguration,
)


class _StaticJudge(DeepEvalBaseLLM):
    def __init__(self):
        pass

    def load_model(self):
        return self

    def get_model_name(self):
        return "static-judge"

    def generate(self, prompt, schema=None):
        raise AssertionError("the cache comparison must not call the judge")

    async def a_generate(self, prompt, schema=None):
        raise AssertionError("the cache comparison must not call the judge")


LENIENT = [Rubric(score_range=(0, 10), expected_outcome="Any helpful reply.")]
STRICT = [
    Rubric(score_range=(0, 2), expected_outcome="No policy link."),
    Rubric(score_range=(3, 10), expected_outcome="Links the policy."),
]


def _geval(rubric=None):
    return GEval(
        name="Refund handling",
        criteria="Does the reply handle the refund request well?",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        rubric=rubric,
        model=_StaticJudge(),
    )


def _as_read_from_cache_file(configuration: MetricConfiguration):
    return MetricConfiguration.model_validate_json(
        configuration.model_dump_json()
    )


def test_changed_geval_rubric_does_not_reuse_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval(LENIENT))
    )

    assert not Cache.same_metric_configs(_geval(STRICT), cached)
    assert not Cache.same_metric_configs(_geval(), cached)


def test_same_geval_rubric_reuses_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval(STRICT))
    )

    assert Cache.same_metric_configs(_geval(STRICT), cached)


def test_geval_without_rubric_reuses_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval())
    )

    assert Cache.same_metric_configs(_geval(), cached)


def test_rubric_added_after_caching_does_not_reuse_cached_score():
    # Cache files written before the rubric was recorded look the same: the
    # entry has no rubric.
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval())
    )
    assert getattr(cached, "rubric", None) is None

    assert not Cache.same_metric_configs(_geval(STRICT), cached)


class _LenientSchema(BaseModel):
    name: str


class _StrictSchema(BaseModel):
    name: str
    order_id: int


def _json_correctness(expected_schema, async_mode=True):
    return JsonCorrectnessMetric(
        expected_schema=expected_schema,
        model=_StaticJudge(),
        async_mode=async_mode,
    )


def test_changed_json_correctness_expected_schema_does_not_reuse_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_json_correctness(_LenientSchema))
    )

    assert not Cache.same_metric_configs(
        _json_correctness(_StrictSchema), cached
    )
    assert not Cache.same_metric_configs(
        _json_correctness(_StrictSchema, async_mode=False), cached
    )


def test_redefined_schema_with_same_name_does_not_reuse_cached_score():
    class Output(BaseModel):
        name: str

    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_json_correctness(Output))
    )

    class Output(BaseModel):
        name: str
        order_id: int

    assert not Cache.same_metric_configs(_json_correctness(Output), cached)


@pytest.mark.parametrize("async_mode", [True, False])
def test_same_json_correctness_expected_schema_reuses_cached_score(async_mode):
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(
            _json_correctness(_StrictSchema, async_mode=async_mode)
        )
    )

    assert Cache.same_metric_configs(
        _json_correctness(_StrictSchema, async_mode=async_mode), cached
    )


def test_json_schema_extra_with_tuple_still_reuses_cached_score():
    class SchemaWithTupleExtra(BaseModel):
        model_config = ConfigDict(json_schema_extra={"tags": ("a", "b")})
        name: str

    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(
            _json_correctness(SchemaWithTupleExtra)
        )
    )

    assert Cache.same_metric_configs(
        _json_correctness(SchemaWithTupleExtra), cached
    )


def test_unschemable_expected_schema_distinguishes_different_fields():
    class CustomA:
        pass

    class CustomB:
        pass

    class Output(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        value: CustomA

    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_json_correctness(Output))
    )
    assert Cache.same_metric_configs(_json_correctness(Output), cached)

    class Output(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        value: CustomB

    assert not Cache.same_metric_configs(_json_correctness(Output), cached)


def test_cached_entry_without_expected_schema_backwards_compat():
    # Pre-fix cache entries have no expected_schema (defaults to None).
    legacy_geval_cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval())
    )
    assert getattr(legacy_geval_cached, "expected_schema", None) is None

    assert Cache.same_metric_configs(_geval(), legacy_geval_cached)
    assert not Cache.same_metric_configs(
        _json_correctness(_StrictSchema), legacy_geval_cached
    )


def test_cached_test_run_save_load_roundtrip_expected_schema(tmp_path):
    cache_file = tmp_path / "cache.json"
    cached_run = CachedTestRun(
        test_cases_lookup_map={
            "case-1": CachedTestCase(
                cached_metrics_data=[
                    CachedMetricData(
                        metric_data=MetricData(
                            name="Json Correctness",
                            threshold=1.0,
                            success=True,
                            score=1.0,
                            strictMode=True,
                        ),
                        metric_configuration=Cache.create_metric_configuration(
                            _json_correctness(_LenientSchema)
                        ),
                    )
                ]
            )
        }
    )
    with open(cache_file, "w", encoding="utf-8") as f:
        cached_run.save(f)

    with open(cache_file, "r", encoding="utf-8") as f:
        loaded_run = CachedTestRun.load(json.load(f))

    loaded_config = (
        loaded_run.test_cases_lookup_map["case-1"]
        .cached_metrics_data[0]
        .metric_configuration
    )
    assert Cache.same_metric_configs(
        _json_correctness(_LenientSchema), loaded_config
    )
    assert not Cache.same_metric_configs(
        _json_correctness(_StrictSchema), loaded_config
    )
