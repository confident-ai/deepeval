"""Batched System One in `evaluate()`: every `system_one` metric on a test
case shares one Jev request."""

import pytest

from deepeval.evaluate import execute as execute_module
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    RoleAdherenceMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)
from deepeval.metrics.utils import system_one_batch
from deepeval.metrics.utils.system_one_batch import (
    a_measure_system_one_batch,
    measure_system_one_batch,
)
from deepeval.models.system_one.limits import SystemOneContextLimitError
from deepeval.models.utils import EvaluationCost
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from tests.test_metrics.system_one_fakes import (
    ExplodingLLM,
    ExplodingSystemOneModel,
    FakeSystemOneModel,
    answer_everything,
)


def jev(name: str = "fake-jev", **kwargs) -> FakeSystemOneModel:
    return FakeSystemOneModel(
        answer_fn=answer_everything(), name=name, **kwargs
    )


def relevancy(model, **kwargs) -> AnswerRelevancyMetric:
    return AnswerRelevancyMetric(
        model=ExplodingLLM(),
        system_one_model=model,
        eval_mode="system_one",
        **kwargs,
    )


def faithfulness(model, **kwargs) -> FaithfulnessMetric:
    return FaithfulnessMetric(
        model=ExplodingLLM(),
        system_one_model=model,
        eval_mode="system_one",
        **kwargs,
    )


def make_case(**overrides) -> LLMTestCase:
    fields = dict(
        input="Where is the Eiffel Tower?",
        actual_output="The Eiffel Tower is in Paris.",
        retrieval_context=["The Eiffel Tower stands in Paris, France."],
    )
    fields.update(overrides)
    return LLMTestCase(**fields)


async def run(metrics, tc, is_async, **kwargs):
    kwargs.setdefault("ignore_errors", False)
    kwargs.setdefault("skip_on_missing_params", False)
    if is_async:
        return await a_measure_system_one_batch(metrics, tc, **kwargs)
    return measure_system_one_batch(metrics, tc, **kwargs)


sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)


@sync_and_async
@pytest.mark.asyncio
async def test_metrics_share_one_request(is_async):
    model = jev()
    metrics = [relevancy(model), faithfulness(model)]

    handled = await run(metrics, make_case(), is_async)

    assert handled == metrics
    assert len(model.calls) == 1
    state, questions = model.calls[0]
    assert set(state["test_case"]) == {
        "input",
        "actual_output",
        "retrieval_context",
    }
    assert {key.split(".")[0] for key in questions} == {"m0", "m1"}


@sync_and_async
@pytest.mark.asyncio
async def test_batched_results_match_separate_measures(is_async):
    model = jev()
    batched = [relevancy(model, threshold=0.3), faithfulness(model)]
    await run(batched, make_case(), is_async)

    for metric, alone in zip(
        batched, [relevancy(jev(), threshold=0.3), faithfulness(jev())]
    ):
        alone.measure(make_case())
        assert metric.score == alone.score
        assert metric.reason == alone.reason
        assert metric.success == alone.success
        assert metric.confidence == alone.confidence


@sync_and_async
@pytest.mark.asyncio
async def test_single_jev_metric_is_not_batched(is_async):
    model = jev()
    assert await run([relevancy(model)], make_case(), is_async) == []
    assert model.calls == []


@sync_and_async
@pytest.mark.asyncio
async def test_eval_mode_override_is_honoured(is_async):
    model = jev()
    on_llm = AnswerRelevancyMetric(
        model=ExplodingLLM(), system_one_model=model, eval_mode="llm"
    )
    metrics = [relevancy(model), faithfulness(model), on_llm]

    handled = await run(metrics, make_case(), is_async)

    assert handled == metrics[:2]
    assert len(model.calls) == 1


@sync_and_async
@pytest.mark.asyncio
async def test_each_jev_model_gets_its_own_request(is_async):
    first, second = jev("jev-a"), jev("jev-b")

    await run([relevancy(first), faithfulness(second)], make_case(), is_async)

    assert len(first.calls) == 1
    assert len(second.calls) == 1


@sync_and_async
@pytest.mark.asyncio
async def test_request_over_budget_is_split(is_async, monkeypatch):
    def at_most_three_questions(state, questions):
        if len(questions) > 3:
            raise SystemOneContextLimitError(
                "too big", estimated_tokens=2, limit_tokens=1
            )

    monkeypatch.setattr(
        system_one_batch, "check_context_budget", at_most_three_questions
    )
    model = jev()
    metrics = [relevancy(model), faithfulness(model)]

    await run(metrics, make_case(), is_async)

    assert len(model.calls) == 2
    assert all(metric.score is not None for metric in metrics)


@sync_and_async
@pytest.mark.asyncio
async def test_failed_request_errors_only_its_metrics(is_async):
    down = ExplodingSystemOneModel(RuntimeError("jev is down"))
    healthy = jev("healthy-jev")
    failed = [relevancy(down), faithfulness(down)]
    judged = relevancy(healthy)

    await run(failed + [judged], make_case(), is_async, ignore_errors=True)

    assert len(down.calls) == 1
    for metric in failed:
        assert "jev is down" in metric.error
        assert metric.success is False
    assert judged.error is None
    assert judged.score is not None


@sync_and_async
@pytest.mark.asyncio
async def test_failed_request_raises_without_ignore_errors(is_async):
    down = ExplodingSystemOneModel(RuntimeError("jev is down"))
    with pytest.raises(RuntimeError, match="jev is down"):
        await run([relevancy(down), faithfulness(down)], make_case(), is_async)


@sync_and_async
@pytest.mark.asyncio
async def test_cost_is_split_by_question_share(is_async):
    model = jev(cost=EvaluationCost(0.6, input_tokens=11, output_tokens=0))
    metrics = [relevancy(model), faithfulness(model)]

    await run(metrics, make_case(), is_async)

    assert sum(metric.evaluation_cost for metric in metrics) == pytest.approx(
        0.6
    )
    assert sum(metric.input_tokens for metric in metrics) == 11


def test_split_cost_keeps_totals():
    shares = system_one_batch._split_cost(
        EvaluationCost(1.0, input_tokens=10, output_tokens=1), [1, 2]
    )
    assert [s.value for s in shares] == pytest.approx([1 / 3, 2 / 3])
    assert [s.input_tokens for s in shares] == [3, 7]
    assert sum(s.output_tokens for s in shares) == 1
    assert system_one_batch._split_cost(None, [1, 2]) == [None, None]


@sync_and_async
@pytest.mark.asyncio
async def test_missing_param_is_skipped(is_async):
    model = jev()
    missing_context = faithfulness(model)
    metrics = [relevancy(model), missing_context, relevancy(model)]

    handled = await run(
        metrics,
        make_case(retrieval_context=None),
        is_async,
        skip_on_missing_params=True,
    )

    assert handled == metrics
    assert missing_context.skipped is True
    assert missing_context.error is None
    assert len(model.calls) == 1


@sync_and_async
@pytest.mark.asyncio
async def test_metric_without_a_whole_jev_request_runs_its_own_measure(
    is_async,
):
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import ToolCall

    model = jev()
    tool_correctness = ToolCorrectnessMetric(
        system_one_model=model, eval_mode="system_one"
    )
    metrics = [relevancy(model), faithfulness(model), tool_correctness]
    tc = make_case(
        tools_called=[ToolCall(name="search")],
        expected_tools=[ToolCall(name="search")],
    )

    handled = await run(metrics, tc, is_async)

    assert handled == metrics[:2]
    assert tool_correctness.error is None
    assert len(model.calls) == 1


@sync_and_async
@pytest.mark.asyncio
async def test_code_scored_json_correctness_runs_its_own_measure(is_async):
    from pydantic import BaseModel

    from deepeval.metrics import JsonCorrectnessMetric

    class Answer(BaseModel):
        city: str

    model = jev()
    json_correctness = JsonCorrectnessMetric(
        expected_schema=Answer, eval_mode="system_one"
    )
    metrics = [relevancy(model), faithfulness(model), json_correctness]
    tc = make_case(actual_output='{"city": "Paris"}')

    handled = await run(metrics, tc, is_async)

    assert handled == metrics[:2]
    assert json_correctness.error is None
    assert len(model.calls) == 1
    json_correctness.measure(tc)
    assert json_correctness.score == 1.0


@sync_and_async
@pytest.mark.asyncio
async def test_multimodal_test_case_is_not_batched(is_async):
    from deepeval.test_case import MLLMImage

    model = jev()
    tc = make_case(
        actual_output=f"Here: {MLLMImage(url='https://example.com/a.png')}"
    )
    assert tc.multimodal
    assert (
        await run([relevancy(model), faithfulness(model)], tc, is_async) == []
    )
    assert model.calls == []


###############################################
# Conversational
###############################################


def jev_judged(cls, model, **kwargs):
    return cls(
        model=ExplodingLLM(),
        system_one_model=model,
        eval_mode="system_one",
        **kwargs,
    )


def make_conversation(**overrides) -> ConversationalTestCase:
    fields = dict(
        chatbot_role="A concise travel guide.",
        turns=[
            Turn(role="user", content="Where is the Eiffel Tower?"),
            Turn(
                role="assistant",
                content="It is in Paris.",
                retrieval_context=["The Eiffel Tower stands in Paris."],
            ),
        ],
    )
    fields.update(overrides)
    return ConversationalTestCase(**fields)


@sync_and_async
@pytest.mark.asyncio
async def test_conversational_metrics_share_one_request(is_async):
    model = jev()
    metrics = [
        jev_judged(TurnRelevancyMetric, model),
        jev_judged(TurnFaithfulnessMetric, model),
    ]

    handled = await run(metrics, make_conversation(), is_async)

    assert handled == metrics
    assert len(model.calls) == 1
    turns = model.calls[0][0]["turns"]
    assert len(turns) == 2
    assert turns[1]["retrieval_context"]
    assert turns[1]["content"] == "It is in Paris."

    for metric, cls in zip(
        metrics, [TurnRelevancyMetric, TurnFaithfulnessMetric]
    ):
        alone = jev_judged(cls, jev())
        alone.measure(make_conversation())
        assert metric.score == alone.score
        assert metric.reason == alone.reason


@sync_and_async
@pytest.mark.asyncio
async def test_conversational_missing_required_fields_are_skipped(is_async):
    from deepeval.metrics import MultiTurnMCPUseMetric

    model = jev()
    role_adherence = jev_judged(RoleAdherenceMetric, model)
    mcp_use = jev_judged(MultiTurnMCPUseMetric, model)
    judged = [
        jev_judged(TurnRelevancyMetric, model),
        jev_judged(TurnFaithfulnessMetric, model),
    ]

    await run(
        [role_adherence, mcp_use, *judged],
        make_conversation(chatbot_role=None),
        is_async,
        skip_on_missing_params=True,
    )

    assert role_adherence.skipped is True
    assert mcp_use.skipped is True
    assert len(model.calls) == 1
    assert all(metric.score is not None for metric in judged)


def test_every_whole_jev_metric_uses_prepare_measure():
    import pathlib
    import re

    import deepeval.metrics

    root = pathlib.Path(deepeval.metrics.__file__).parent
    stale = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "base_metric.py"
        and "def _system_one_eval_spec(" in (text := path.read_text())
        and re.search(r"check_(llm|conversational)_test_case_params\(", text)
    ]
    assert stale == []


###############################################
# Executors
###############################################


def _scores(results):
    return [
        [(m.name, m.score, m.success, m.reason) for m in r.metrics_data]
        for r in results
    ]


SINGLE_TURN = (
    [make_case(), make_case(input="What is in Paris?")],
    [AnswerRelevancyMetric, FaithfulnessMetric],
)
CONVERSATIONAL = (
    [make_conversation(), make_conversation(chatbot_role="A curt guide.")],
    [TurnRelevancyMetric, TurnFaithfulnessMetric],
)


@pytest.mark.parametrize(
    "cases, metric_classes",
    [SINGLE_TURN, CONVERSATIONAL],
    ids=["single_turn", "conversational"],
)
@pytest.mark.asyncio
async def test_sync_and_async_executors_agree(cases, metric_classes):
    configs = dict(
        error_config=ErrorConfig(
            ignore_errors=False, skip_on_missing_params=False
        ),
        display_config=DisplayConfig(show_indicator=False, verbose_mode=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )

    sync_model = jev()
    sync_results = execute_module.execute_test_cases(
        test_cases=cases,
        metrics=[jev_judged(cls, sync_model) for cls in metric_classes],
        **configs,
    )
    async_model = jev()
    async_results = await execute_module.a_execute_test_cases(
        test_cases=cases,
        metrics=[jev_judged(cls, async_model) for cls in metric_classes],
        async_config=AsyncConfig(max_concurrent=2, throttle_value=0),
        **configs,
    )

    assert len(sync_model.calls) == len(cases)
    assert len(async_model.calls) == len(cases)
    assert sorted(_scores(sync_results)) == sorted(_scores(async_results))
    assert all(
        m.score is not None for r in sync_results for m in r.metrics_data
    )
