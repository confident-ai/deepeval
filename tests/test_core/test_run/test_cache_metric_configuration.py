from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams
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
