from pydantic import BaseModel
from deepeval.metrics import JsonCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_run.cache import Cache, MetricConfiguration


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


class Lenient(BaseModel):
    name: str


class Strict(BaseModel):
    name: str
    order_id: int


def _metric(schema=None):
    return JsonCorrectnessMetric(
        expected_schema=schema,
        model=_StaticJudge(),
    )


def _as_read_from_cache_file(configuration: MetricConfiguration):
    return MetricConfiguration.model_validate_json(
        configuration.model_dump_json()
    )


def test_changed_json_correctness_schema_does_not_reuse_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_metric(Lenient))
    )

    assert not Cache.same_metric_configs(_metric(Strict), cached)
    assert not Cache.same_metric_configs(_metric(None), cached)


def test_same_json_correctness_schema_reuses_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_metric(Strict))
    )

    assert Cache.same_metric_configs(_metric(Strict), cached)


def test_json_correctness_without_schema_reuses_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_metric(None))
    )

    assert Cache.same_metric_configs(_metric(None), cached)


def test_schema_added_after_caching_does_not_reuse_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_metric(None))
    )
    assert getattr(cached, "expected_schema", None) is None

    assert not Cache.same_metric_configs(_metric(Strict), cached)
