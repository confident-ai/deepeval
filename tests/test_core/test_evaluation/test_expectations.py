import json
import importlib
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from deepeval import assert_test, evaluate
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.dataset.utils import (
    convert_goldens_to_test_cases,
    convert_test_cases_to_goldens,
    convert_convo_goldens_to_convo_test_cases,
    convert_convo_test_cases_to_convo_goldens,
)
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    Expectations,
    LLMTestCase,
    ConversationalTestCase,
    ToolCall,
    Turn,
)
from deepeval.test_run import global_test_run_manager
from deepeval.test_run.cache import (
    CachedTestCase,
    CachedTestRun,
    TestRunCacheManager,
)


class Judge:
    def __init__(self):
        self.prompts = []
        self.status = "pass"
        self.invalid_ids = False

    def get_model_name(self):
        return "stub-expectation-judge"

    def generate_with_schema(self, prompt, schema):
        self.prompts.append(prompt)
        requirements = json.loads(
            prompt.split("Requirements: ", 1)[1].split("\nObserved case: ")[0]
        )
        observations = json.loads(prompt.split("\nObserved case: ", 1)[1])
        status = (
            "fail"
            if observations.get("actual_output") == "bad"
            else self.status
        )
        return schema.model_validate(
            {
                "verdicts": [
                    {
                        "id": "unknown" if self.invalid_ids else item["id"],
                        "status": status,
                        "reason": "checked condition",
                        "evidence": "observed response",
                    }
                    for item in requirements
                ]
            }
        )

    async def a_generate_with_schema(self, prompt, schema):
        return self.generate_with_schema(prompt, schema)


class PassingMetric(BaseMetric):
    threshold = 1

    @property
    def __name__(self):
        return "Existing metric"

    def measure(self, test_case, **kwargs):
        self.score = 1
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case)


@pytest.fixture(autouse=True)
def expectation_mode(monkeypatch):
    from deepeval.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "DEEPEVAL_EVAL_MODE", None)


@pytest.fixture
def judge(monkeypatch):
    module = importlib.import_module("deepeval.evaluate.expectations")
    judge = Judge()
    monkeypatch.setattr(
        module, "initialize_model", lambda *args, **kwargs: (judge, False)
    )
    global_test_run_manager.reset()
    monkeypatch.setattr(
        global_test_run_manager,
        "wrap_up_test_run",
        lambda *a, **k: (None, None),
    )
    yield judge
    global_test_run_manager.reset()


def run(cases, run_async=False, show_indicator=False, **kwargs):
    return evaluate(
        cases,
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(
            show_indicator=show_indicator, print_results=False
        ),
        cache_config=CacheConfig(use_cache=False, write_cache=False),
        **kwargs,
    ).test_results


@pytest.mark.parametrize(
    "model,fields",
    [
        (Golden, {"input": "hello"}),
        (LLMTestCase, {"input": "hello"}),
        (ConversationalGolden, {"scenario": "hello"}),
        (
            ConversationalTestCase,
            {"turns": [Turn(role="user", content="hello")]},
        ),
    ],
)
def test_model_serialization(model, fields):
    case = model(
        **fields,
        expectations={
            "must": [" Greet the user "],
            "mustNot": ["Disclose a password"],
        },
    )
    assert case.expectations.must == ["Greet the user"]
    restored = model.model_validate(
        case.model_dump(by_alias=True, exclude_none=True)
    )
    assert restored.expectations == case.expectations


@pytest.mark.parametrize(
    "value",
    [
        {"must": "one paragraph"},
        {"must": [""]},
        {"must_not": ["  "]},
        {"must": [2]},
        {"must_nt": ["typo"]},
    ],
)
def test_invalid_expectations(value):
    with pytest.raises(ValidationError):
        Expectations.model_validate(value)


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("show_indicator", [False, True])
def test_expectations_only_and_failure(judge, run_async, show_indicator):
    cases = [
        LLMTestCase(
            input="hello",
            actual_output=output,
            expectations={
                "must": ["Greet the user"],
                "must_not": ["Disclose a password"],
            },
        )
        for output in ["hello", "bad"]
    ]
    results = run(cases, run_async, show_indicator)
    assert [r.success for r in results] == [True, False]
    assert len(judge.prompts) == 2
    assert all(len(r.metrics_data) == 1 for r in results)
    assert results[0].expectations == cases[0].expectations
    assert "must_not[0]" in results[0].metrics_data[0].reason
    assert (
        results[0].metrics_data[0].evaluation_model == "stub-expectation-judge"
    )


@pytest.mark.parametrize("run_async", [False, True])
def test_conversation_checks_all_turns(judge, run_async):
    case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Cancel"),
            Turn(role="assistant", content="Please confirm"),
            Turn(role="user", content="No"),
        ],
        expected_outcome="Cancellation handled",
        expectations={"must_not": ["Cancel without confirmation"]},
    )
    results = run([case], run_async)
    assert results[0].success
    evidence = json.loads(judge.prompts[0].split("\nObserved case: ", 1)[1])
    assert len(evidence["turns"]) == 3
    assert "expected_outcome" not in evidence


@pytest.mark.parametrize("run_async", [False, True])
def test_mixed_cases_and_metrics(judge, run_async):
    cases = [
        LLMTestCase(
            input="hello", actual_output="hello", expectations=expectations
        )
        for expectations in [
            {"must": ["Greet"]},
            None,
            {},
            {"must_not": ["Insult"]},
        ]
    ]
    metrics = [PassingMetric()]
    results = run(cases, run_async, metrics=metrics)
    assert len(metrics) == 1
    assert [len(r.metrics_data) for r in results] == [2, 1, 1, 2]
    assert len(judge.prompts) == 2


@pytest.mark.parametrize("run_async", [False, True])
def test_assert_test_needs_no_metric(judge, run_async):
    case = LLMTestCase(
        input="hello", actual_output="hello", expectations={"must": ["Greet"]}
    )
    assert_test(case, run_async=run_async)
    case.actual_output = "bad"
    with pytest.raises(AssertionError, match="Expectations"):
        assert_test(case, run_async=run_async)


@pytest.mark.parametrize("run_async", [False, True])
def test_unknown_is_error_not_pass(judge, run_async):
    judge.status = "unable_to_evaluate"
    case = LLMTestCase(
        input="cancel",
        actual_output="Done",
        expectations={"must": ["Cancel subscription"]},
    )
    results = run(
        [case], run_async, error_config=ErrorConfig(ignore_errors=True)
    )
    assert results[0].success is False
    assert results[0].metrics_data[0].score is None
    assert "Unable to evaluate expectations" in results[0].metrics_data[0].error
    with pytest.raises(ValueError, match="Unable to evaluate expectations"):
        run([case], run_async)


def test_invalid_judge_response_is_error(judge):
    judge.invalid_ids = True
    with pytest.raises(ValueError, match="verdict IDs"):
        run(
            [
                LLMTestCase(
                    input="hello",
                    actual_output="hello",
                    expectations={"must": ["Greet"]},
                )
            ]
        )


def test_empty_expectations_do_not_initialize_judge(monkeypatch):
    module = importlib.import_module("deepeval.evaluate.expectations")
    initialize = Mock(side_effect=AssertionError("must not initialize judge"))
    monkeypatch.setattr(module, "initialize_model", initialize)
    with pytest.raises(ValueError, match="expectations"):
        run([LLMTestCase(input="hello", expectations={})])
    initialize.assert_not_called()


@pytest.mark.parametrize("conversational", [False, True])
@pytest.mark.parametrize("file_type", ["json", "jsonl", "csv"])
def test_dataset_roundtrip(tmp_path, conversational, file_type):
    expectations = Expectations(
        must=["Ask for confirmation"], must_not=["Act before confirmation"]
    )
    if conversational:
        golden = ConversationalGolden(
            scenario="Cancellation",
            turns=[Turn(role="user", content="Cancel")],
            expectations=expectations,
        )
        cases = convert_convo_goldens_to_convo_test_cases([golden])
        restored = convert_convo_test_cases_to_convo_goldens(cases)[0]
    else:
        golden = Golden(
            input="Cancel", actual_output="Confirm?", expectations=expectations
        )
        cases = convert_goldens_to_test_cases([golden])
        restored = convert_test_cases_to_goldens(cases)[0]
    assert cases[0].expectations == restored.expectations == expectations
    dataset = EvaluationDataset(goldens=[golden])
    path = dataset.save_as(file_type, str(tmp_path), "expectations")
    loaded = EvaluationDataset()
    getattr(loaded, f"add_goldens_from_{file_type}_file")(path)
    assert loaded.goldens[0].expectations == expectations


def test_cache_invalidated_by_requirements_and_actions(monkeypatch):
    import deepeval.test_run.cache as module

    monkeypatch.setattr(module, "portalocker", object())
    manager = TestRunCacheManager()
    manager.disable_write_cache = False
    cached_run = CachedTestRun()
    monkeypatch.setattr(
        manager, "get_cached_test_run", lambda **kwargs: cached_run
    )
    monkeypatch.setattr(manager, "save_cached_test_run", lambda **kwargs: None)
    case = LLMTestCase(
        input="cancel", actual_output="Done", expectations={"must": ["Cancel"]}
    )
    manager.cache_test_case(case, CachedTestCase(), None)
    assert manager.get_cached_test_case(case, None) is not None
    case.expectations.must[0] = "Ask for confirmation"
    assert manager.get_cached_test_case(case, None) is None
    case.expectations.must[0] = "Cancel"
    case.tools_called = [ToolCall(name="cancel")]
    assert manager.get_cached_test_case(case, None) is None


def test_trace_scoped_assert_uses_golden_expectations(judge, monkeypatch):
    from time import perf_counter
    from deepeval.evaluate.execute.trace_scope import (
        _assert_test_from_current_trace,
    )
    from deepeval.tracing.context import current_trace_context
    from deepeval.tracing.types import BaseSpan, TraceSpanStatus
    from tests.test_core.test_evaluation.test_trace_scope_assert_test import (
        _make_pytest_wrapped_trace,
    )

    span = BaseSpan(
        uuid="app",
        trace_uuid="trace",
        parent_uuid="wrapper",
        status=TraceSpanStatus.SUCCESS,
        children=[],
        start_time=perf_counter(),
        end_time=perf_counter(),
        name="app",
        input="hello",
        output="hello",
    )
    trace = _make_pytest_wrapped_trace(span)
    token = current_trace_context.set(trace)
    try:
        result = _assert_test_from_current_trace(
            Golden(input="hello", expectations={"must": ["Greet"]}),
            display_config=DisplayConfig(
                show_indicator=False, print_results=False
            ),
        )
    finally:
        current_trace_context.reset(token)
    assert result.success
    assert result.metrics_data[0].name == "Expectations"
    assert '"trace"' in judge.prompts[0]


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("explicit_metrics", [False, True])
def test_golden_iterator_evaluates_expectations(
    judge, run_async, explicit_metrics
):
    import asyncio
    from deepeval.tracing import observe, update_current_trace

    @observe()
    def app(value):
        update_current_trace(input=value, output="hello")
        return "hello"

    @observe()
    async def async_app(value):
        update_current_trace(input=value, output="hello")
        return "hello"

    dataset = EvaluationDataset(
        goldens=[Golden(input="hello", expectations={"must": ["Greet"]})]
    )
    for golden in dataset.evals_iterator(
        metrics=[PassingMetric()] if explicit_metrics else None,
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    ):
        if run_async:
            asyncio.create_task(async_app(golden.input))
        else:
            app(golden.input)
    assert len(judge.prompts) == 1


@pytest.mark.parametrize("run_async", [False, True])
def test_expectations_and_classifiers_both_affect_success(judge, run_async):
    from deepeval.classifiers.base_classifier import BaseClassifier, Label

    class Classifier(BaseClassifier):
        name = "Greeting"
        labels = [Label(name="greeting"), Label(name="other")]

        def classify(self, test_case, **kwargs):
            self.label = "greeting"
            return self.label

        async def a_classify(self, test_case, **kwargs):
            return self.classify(test_case)

    cases = [
        LLMTestCase(
            input="hello",
            actual_output=output,
            expected_labels={"Greeting": label},
            expectations={"must": ["Greet"]},
        )
        for output, label in [
            ("hello", "greeting"),
            ("hello", "other"),
            ("bad", "greeting"),
        ]
    ]
    results = run(cases, run_async, classifiers=[Classifier()])
    assert [result.success for result in results] == [True, False, False]
    assert [result.classifications[0].success for result in results] == [
        True,
        False,
        True,
    ]


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("conversational", [False, True])
def test_no_verdict_for_cases_without_expectations(
    judge, run_async, conversational
):
    if conversational:
        plain = ConversationalTestCase(
            turns=[Turn(role="user", content="hello")]
        )
    else:
        plain = LLMTestCase(input="hello", actual_output="hello")
    checked = plain.model_copy(deep=True)
    checked.expectations = Expectations(must_not=["Disclose a password"])
    results = run([plain, checked], run_async)
    assert len(results) == 1
    assert results[0].expectations == checked.expectations
    assert len(judge.prompts) == 1


@pytest.mark.parametrize("run_async", [False, True])
def test_each_case_uses_its_own_model(judge, monkeypatch, run_async):
    from deepeval.metrics.utils import initialize_model
    from tests.test_metrics.system_one_fakes import CannedLLM

    module = importlib.import_module("deepeval.evaluate.expectations")
    monkeypatch.setattr(module, "initialize_model", initialize_model)
    models = [
        CannedLLM(
            json.dumps(
                {
                    "verdicts": [
                        {
                            "id": "must[0]",
                            "status": status,
                            "reason": "custom judge",
                            "evidence": "response",
                        }
                    ]
                }
            ),
            name=name,
        )
        for name, status in [("first-judge", "pass"), ("second-judge", "fail")]
    ]
    cases = [
        LLMTestCase(
            input="hello",
            actual_output="hello",
            expectations=Expectations(
                must=["Greet"], model=model, eval_mode="llm"
            ),
        )
        for model in models
    ]
    results = run(cases, run_async)
    assert [result.success for result in results] == [True, False]
    assert [result.metrics_data[0].evaluation_model for result in results] == [
        "first-judge",
        "second-judge",
    ]
    assert all(len(model.prompts) == 1 for model in models)
    assert not judge.prompts


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("conversational", [False, True])
def test_system_one_mode(judge, monkeypatch, run_async, conversational):
    from deepeval.metrics.utils import initialize_model
    from tests.test_metrics.system_one_fakes import (
        FakeSystemOneModel,
        answer_everything,
        ExplodingLLM,
    )

    module = importlib.import_module("deepeval.evaluate.expectations")
    system_one = FakeSystemOneModel(answer_fn=answer_everything())
    monkeypatch.setattr(module, "initialize_model", initialize_model)
    monkeypatch.setattr(
        module, "initialize_system_one_model", lambda *args: system_one
    )
    expectations = Expectations(
        must=["Greet"],
        must_not=["Insult"],
        model=ExplodingLLM(),
        eval_mode="system_one",
    )
    case = (
        ConversationalTestCase(
            turns=[Turn(role="assistant", content="hello")],
            expectations=expectations,
        )
        if conversational
        else LLMTestCase(
            input="hello", actual_output="hello", expectations=expectations
        )
    )
    results = run([case], run_async)
    assert results[0].success
    assert len(system_one.calls) == 1
    assert not judge.prompts


@pytest.mark.parametrize(
    "global_mode,local_mode",
    [
        ("llm", "system_one"),
        ("system_one", "llm"),
        ("hybrid", "llm"),
        ("hybrid", "system_one"),
    ],
)
def test_global_mode_overrides_object(
    judge, monkeypatch, global_mode, local_mode
):
    from deepeval.config.settings import get_settings
    from deepeval.config.eval_mode import EvalMode
    from tests.test_metrics.system_one_fakes import (
        FakeSystemOneModel,
        answer_everything,
    )
    from deepeval.evaluate.expectations import _SingleTurnExpectations

    monkeypatch.setattr(get_settings(), "DEEPEVAL_EVAL_MODE", global_mode)
    module = importlib.import_module("deepeval.evaluate.expectations")
    fake = FakeSystemOneModel(answer_fn=answer_everything())
    monkeypatch.setattr(
        module, "initialize_system_one_model", lambda *args: fake
    )
    evaluator = _SingleTurnExpectations(
        Expectations(
            must=["Greet"], model="configured-name", eval_mode=local_mode
        )
    )
    assert evaluator.eval_mode is (
        EvalMode.LLM if global_mode == "hybrid" else EvalMode(global_mode)
    )


@pytest.mark.parametrize("run_async", [False, True])
def test_system_one_missing_evidence_is_an_error(judge, monkeypatch, run_async):
    from deepeval.metrics.utils import initialize_model
    from deepeval.models.system_one.schema import ChoiceAnswer, SystemOneAnswers
    from tests.test_metrics.system_one_fakes import (
        FakeSystemOneModel,
        ExplodingLLM,
    )

    def answer(questions):
        return SystemOneAnswers(
            choices={
                key: ChoiceAnswer(
                    choice="unable_to_evaluate",
                    probabilities={
                        "pass": 0,
                        "fail": 0,
                        "unable_to_evaluate": 1,
                    },
                    confidence=1,
                )
                for key in questions
            }
        )

    module = importlib.import_module("deepeval.evaluate.expectations")
    monkeypatch.setattr(module, "initialize_model", initialize_model)
    monkeypatch.setattr(
        module,
        "initialize_system_one_model",
        lambda *args: FakeSystemOneModel(answer_fn=answer),
    )
    case = LLMTestCase(
        input="cancel",
        actual_output="done",
        expectations=Expectations(
            must=["Cancel"], model=ExplodingLLM(), eval_mode="system_one"
        ),
    )
    result = run(
        [case], run_async, error_config=ErrorConfig(ignore_errors=True)
    )[0]
    assert not result.success
    assert "Unable to evaluate expectations" in result.metrics_data[0].error


@pytest.mark.parametrize("run_async", [False, True])
def test_system_one_failure_does_not_fall_back(judge, monkeypatch, run_async):
    from deepeval.metrics.utils import initialize_model
    from tests.test_metrics.system_one_fakes import (
        CannedLLM,
        ExplodingSystemOneModel,
    )

    module = importlib.import_module("deepeval.evaluate.expectations")
    llm = CannedLLM('{"verdict":"pass","reason":"fallback"}')
    system_one = ExplodingSystemOneModel(ConnectionError("unavailable"))
    monkeypatch.setattr(module, "initialize_model", initialize_model)
    monkeypatch.setattr(
        module, "initialize_system_one_model", lambda *args: system_one
    )
    case = LLMTestCase(
        input="hello",
        actual_output="hello",
        expectations=Expectations(
            must=["Greet"], model=llm, eval_mode="system_one"
        ),
    )
    with pytest.raises(ConnectionError, match="unavailable"):
        run([case], run_async)
    assert not llm.prompts


def test_expectation_judge_config_validation_and_serialization():
    from tests.test_metrics.system_one_fakes import ExplodingLLM

    value = Expectations(
        must=["Greet"], model="model-name", eval_mode="system_one"
    )
    assert Expectations.model_validate_json(value.model_dump_json()) == value
    value.model = ExplodingLLM()
    assert json.loads(value.model_dump_json())["model"] is None
    for invalid in [
        {"model": object()},
        {"eval_mode": "typo"},
        {"eval_mode": "hybrid"},
    ]:
        with pytest.raises(ValidationError):
            Expectations(**invalid)


def test_cache_does_not_share_judge_modes(judge, monkeypatch):
    from deepeval.evaluate.expectations import _SingleTurnExpectations
    from deepeval.test_run.cache import Cache
    from tests.test_metrics.system_one_fakes import (
        FakeSystemOneModel,
        answer_everything,
    )

    module = importlib.import_module("deepeval.evaluate.expectations")
    monkeypatch.setattr(
        module,
        "initialize_system_one_model",
        lambda *args: FakeSystemOneModel(answer_fn=answer_everything()),
    )
    llm = _SingleTurnExpectations(Expectations(must=["Greet"], eval_mode="llm"))
    system_one = _SingleTurnExpectations(
        Expectations(must=["Greet"], eval_mode="system_one")
    )
    assert not Cache.same_metric_configs(
        llm, Cache.create_metric_configuration(system_one)
    )
