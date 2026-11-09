from deepeval.metrics import PromptAlignmentMetric
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


LENIENT_INSTRUCTIONS = ["Reply in English"]
STRICT_INSTRUCTIONS = [
    "Reply in English",
    "Include a direct link to the refund policy",
]


def _prompt_alignment(instructions):
    return PromptAlignmentMetric(
        prompt_instructions=instructions,
        model=_StaticJudge(),
    )


def _as_read_from_cache_file(configuration: MetricConfiguration):
    return MetricConfiguration.model_validate_json(
        configuration.model_dump_json()
    )


def test_changed_prompt_alignment_instructions_does_not_reuse_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(
            _prompt_alignment(LENIENT_INSTRUCTIONS)
        )
    )

    assert not Cache.same_metric_configs(
        _prompt_alignment(STRICT_INSTRUCTIONS), cached
    )


def test_same_prompt_alignment_instructions_reuses_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(
            _prompt_alignment(STRICT_INSTRUCTIONS)
        )
    )

    assert Cache.same_metric_configs(
        _prompt_alignment(STRICT_INSTRUCTIONS), cached
    )


def test_legacy_cache_without_prompt_instructions_does_not_reuse_cached_score():
    cached = MetricConfiguration(
        threshold=0.5,
        evaluation_model="static-judge",
        strict_mode=False,
        include_reason=True,
    )
    assert getattr(cached, "prompt_instructions", None) is None

    assert not Cache.same_metric_configs(
        _prompt_alignment(STRICT_INSTRUCTIONS), cached
    )
