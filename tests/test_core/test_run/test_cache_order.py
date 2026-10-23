import json
from enum import Enum

import pytest
from pydantic import BaseModel

from deepeval.dataset import Expectations
from deepeval.evaluate.utils import create_metric_data
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, RetrievedContextData
from deepeval.test_run.cache import (
    Cache,
    CachedMetricData,
    CachedTestCase,
    CachedTestRun,
    TestRunCacheManager as CacheManager,
)
from deepeval.utils import serialize


RELEVANT = "The warranty lasts two years."
DISTRACTOR = "The package is blue."


class RankJudge(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def get_model_name(self):
        return "fixed-rank-judge"

    def generate(self, prompt, schema=None):
        assert schema.__name__ == "Verdicts"
        ordered = sorted(
            (prompt.index(text), text) for text in (RELEVANT, DISTRACTOR)
        )
        return schema(
            verdicts=[
                {
                    "verdict": "yes" if text == RELEVANT else "no",
                    "reason": "Fixed relevance for this record.",
                }
                for _, text in ordered
            ]
        )

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def precision_metric():
    return ContextualPrecisionMetric(
        model=RankJudge(),
        threshold=0.75,
        include_reason=False,
        async_mode=False,
    )


def _case(**changes):
    fields = {
        "input": "How long is the warranty?",
        "actual_output": "Two years.",
        "expected_output": "Two years.",
        "retrieval_context": [RELEVANT, DISTRACTOR],
    }
    fields.update(changes)
    return LLMTestCase(**fields)


@pytest.fixture
def cache_manager(tmp_path):
    def create():
        manager = CacheManager()
        manager.cache_file_name = str(tmp_path / "cache.json")
        manager.temp_cache_file_name = str(tmp_path / "temp-cache.json")
        return manager

    return create


def store(manager, case, payload, hyperparameters=None, to_temp=False):
    manager.cache_test_case(case, payload, hyperparameters, to_temp=to_temp)
    if to_temp:
        manager.wrap_up_cached_test_run()


@pytest.mark.parametrize("to_temp", [False, True])
def test_reordered_retrieval_does_not_replay_precision(cache_manager, to_temp):
    first = _case()
    reversed_case = _case(retrieval_context=[DISTRACTOR, RELEVANT])
    metric = precision_metric()
    assert metric.measure(first, _show_indicator=False) == 1.0
    payload = CachedTestCase(
        cached_metrics_data=[
            CachedMetricData(
                metric_data=create_metric_data(metric),
                metric_configuration=Cache.create_metric_configuration(metric),
            )
        ]
    )
    store(cache_manager(), first, payload, to_temp=to_temp)

    reader = cache_manager()
    same = Cache.get_metric_data(
        precision_metric(), reader.get_cached_test_case(first, None)
    )
    assert same.metric_data.score == 1.0
    assert same.metric_data.success is True
    assert reader.get_cached_test_case(reversed_case, None) is None

    # With one relevant record at rank two, average precision is 1 / 2.
    fresh = precision_metric()
    assert fresh.measure(reversed_case, _show_indicator=False) == 0.5
    assert fresh.is_successful() is False
    fresh_payload = CachedTestCase(
        cached_metrics_data=[
            CachedMetricData(
                metric_data=create_metric_data(fresh),
                metric_configuration=Cache.create_metric_configuration(fresh),
            )
        ]
    )
    store(reader, reversed_case, fresh_payload)
    reloaded = cache_manager()
    for case, expected_score in ((first, 1.0), (reversed_case, 0.5)):
        cached = Cache.get_metric_data(
            precision_metric(), reloaded.get_cached_test_case(case, None)
        )
        assert cached.metric_data.score == expected_score
        assert cached.metric_data.success is (expected_score >= 0.75)


@pytest.mark.parametrize("field", ["context", "retrieval_context"])
@pytest.mark.parametrize(
    "left,right",
    [
        (["first", "second"], ["second", "first"]),
        (["first", "first", "second"], ["first", "second", "first"]),
        (["first"], ["first", "first"]),
        (None, []),
    ],
)
def test_context_identity_survives_disk_reload(
    cache_manager, field, left, right
):
    first = _case(**{field: left})
    payload = CachedTestCase(hyperparameters="first")
    store(cache_manager(), first, payload)
    reader = cache_manager()
    assert reader.get_cached_test_case(first, None) == payload
    assert reader.get_cached_test_case(_case(**{field: right}), None) is None


@pytest.mark.parametrize(
    "left,right,hit",
    [
        ({"model": "local", "seed": 1}, {"seed": 1, "model": "local"}, True),
        ({"config": {"a": 1, "b": 2}}, {"config": {"b": 2, "a": 1}}, True),
        ({"steps": ["read", "answer"]}, {"steps": ["answer", "read"]}, False),
        ({"steps": [["a", "b"]]}, {"steps": [["b", "a"]]}, False),
        ({"steps": ["a"]}, {"steps": ["a", "a"]}, False),
        ({"value": 1}, {"value": "1"}, False),
        ({"value": True}, {"value": 1}, False),
        ({"value": None}, {"value": []}, False),
    ],
)
def test_hyperparameter_identity_after_reload(cache_manager, left, right, hit):
    case = _case()
    payload = CachedTestCase(hyperparameters="first")
    store(cache_manager(), case, payload, left)
    result = cache_manager().get_cached_test_case(case, right)
    assert (result == payload) is hit


class Phase(Enum):
    READ = "read"
    ANSWER = "answer"


class Settings(BaseModel):
    phases: list[Phase]


@pytest.mark.parametrize(
    "left,right,hit",
    [
        (Phase.READ, Phase.READ, True),
        (Phase.READ, Phase.ANSWER, False),
        (Settings(phases=[Phase.READ]), Settings(phases=[Phase.READ]), True),
        (
            Settings(phases=[Phase.READ, Phase.ANSWER]),
            Settings(phases=[Phase.ANSWER, Phase.READ]),
            False,
        ),
    ],
)
def test_cache_encoder_handles_complex_hyperparameters(
    cache_manager, left, right, hit
):
    case = _case()
    payload = CachedTestCase(hyperparameters="first")
    store(cache_manager(), case, payload, {"config": left})
    result = cache_manager().get_cached_test_case(case, {"config": right})
    assert (result == payload) is hit


def test_structured_retrieval_preserves_normalization_and_order(cache_manager):
    first = _case(
        retrieval_context=[
            RetrievedContextData(source="manual", context=RELEVANT),
            RetrievedContextData(source="box", context=DISTRACTOR),
        ]
    )
    payload = CachedTestCase(hyperparameters="first")
    store(cache_manager(), first, payload)
    reader = cache_manager()
    assert reader.get_cached_test_case(_case(), None) == payload
    reversed_case = first.model_copy(
        update={"retrieval_context": list(reversed(first.retrieval_context))}
    )
    assert reader.get_cached_test_case(reversed_case, None) is None


def test_expectations_and_evidence_still_participate_in_key(cache_manager):
    first = _case(
        expectations=Expectations(must=["Answer the question."]),
        metadata={"phase": "read"},
    )
    payload = CachedTestCase(hyperparameters="first")
    store(cache_manager(), first, payload)
    reader = cache_manager()
    assert reader.get_cached_test_case(first, None) == payload
    for updates in (
        {"expectations": Expectations(must=["Cite the manual."])},
        {"metadata": {"phase": "answer"}},
    ):
        assert (
            reader.get_cached_test_case(first.model_copy(update=updates), None)
            is None
        )


def test_ambiguous_legacy_key_is_not_reused(cache_manager):
    case = _case(retrieval_context=[DISTRACTOR, RELEVANT])
    # Old list sorting let a rank-one score be written under this rank-two key.
    legacy_key = serialize(
        {
            "input": case.input,
            "actual_output": case.actual_output,
            "expected_output": case.expected_output,
            "context": case.context,
            "retrieval_context": [RELEVANT, DISTRACTOR],
            "hyperparameters": None,
        }
    )
    stale = CachedTestCase(hyperparameters="ambiguous legacy result")
    manager = cache_manager()
    with open(manager.cache_file_name, "w") as stream:
        CachedTestRun(test_cases_lookup_map={legacy_key: stale}).save(stream)

    reader = cache_manager()
    assert reader.get_cached_test_case(case, None) is None
    replacement = CachedTestCase(hyperparameters="new ordered result")
    store(reader, case, replacement)
    assert cache_manager().get_cached_test_case(case, None) == replacement
    with open(manager.cache_file_name) as stream:
        keys = json.load(stream)["test_cases_lookup_map"]
    assert legacy_key in keys
    assert len(keys) == 2
