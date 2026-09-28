"""Judge provenance capture: exact bytes, logprobs, metadata, live/replay.

No network: the real OpenAIModel talks to an httpx.MockTransport, so the SDK's
parse path (the one that loses the wire text) is exercised as in production.
"""

import base64
import hashlib
import itertools
import json
import math
import shutil

import pytest

try:  # openai>=3 ships on httpx2
    import httpx2 as httpx
except ImportError:
    import httpx

import deepeval.templates
from deepeval import evaluate
from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.judge_capture import (
    LIVE,
    REPLAY,
    rebuild_chat_completion,
    metric_configuration_snapshot,
    template_fingerprint,
)
from deepeval.metrics import GEval, TaskCompletionMetric
from deepeval.metrics.g_eval import GEvalTemplate, Rubric
from deepeval.metrics.g_eval.utils import calculate_weighted_summed_score
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.test_run.cache import global_test_run_cache_manager

# Deliberately not what the SDK or pydantic would re-serialise to, so only a
# byte-exact capture matches.
STEPS_CONTENT = '{\n  "steps" : ["Check the answer.",  "Check the tone."]\n}'
SCORE_CONTENT = '{"reason":  "Mostly correct.", "score": 7}'


class FakeOpenAI:
    def __init__(self):
        self.ids = itertools.count(1)
        self.exchanges = []  # (request json, response body text, headers)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = next(self.ids)
        if not body.get("logprobs"):
            content, logprobs = STEPS_CONTENT, None
        else:
            top = [
                {"token": "7", "logprob": math.log(0.6), "bytes": None},
                {"token": "8", "logprob": math.log(0.4), "bytes": None},
            ]
            logprobs = {
                "content": [dict(top[0], top_logprobs=top)],
                "refusal": None,
            }
            content = SCORE_CONTENT
        payload = {
            "id": f"chatcmpl-{n:04d}",
            "object": "chat.completion",
            "created": 1790000000,
            "model": "gpt-4.1-mini-2025-04-14",
            "system_fingerprint": f"fp_{n:08d}",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                    },
                    "logprobs": logprobs,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100 + n,
                "completion_tokens": 20,
                "total_tokens": 120 + n,
            },
        }
        text = json.dumps(payload)
        headers = {
            "content-type": "application/json",
            "x-request-id": f"req_{n:06d}",
        }
        self.exchanges.append((body, text, headers))
        return httpx.Response(200, text=text, headers=headers)


@pytest.fixture
def fake():
    return FakeOpenAI()


def make_metric(fake, **geval_kwargs):
    model = OpenAIModel(
        model="gpt-4.1-mini",
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(fake)),
    )
    return GEval(
        name="Correctness",
        criteria="Is the actual output correct?",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        model=model,
        **{"async_mode": False, **geval_kwargs},
    )


TEST_CASE = LLMTestCase(input="What is 2+2?", actual_output="4")


def test_structured_and_raw_responses_are_byte_exact(fake):
    metric = make_metric(fake)
    metric.measure(TEST_CASE)

    assert len(fake.exchanges) == 2
    assert metric.judge_responses == [STEPS_CONTENT, SCORE_CONTENT]
    # Structured response round-trips to the schema's keys.
    assert set(json.loads(metric.judge_responses[0])) == {"steps"}
    for exchange, (_, body, headers) in zip(
        metric.judge_exchanges, fake.exchanges
    ):
        assert exchange["status"] == LIVE
        assert exchange["raw_response_body"] == body
        body_bytes = base64.b64decode(exchange["raw_response_body_base64"])
        assert body_bytes == body.encode()
        assert (
            exchange["raw_response_body_sha256"]
            == hashlib.sha256(body_bytes).hexdigest()
        )
        assert exchange["request_id"] == headers["x-request-id"]
    assert metric.judge_capture_status == LIVE


@pytest.mark.asyncio
async def test_async_path_captures_exact_bytes(fake):
    metric = make_metric(fake)
    metric.model.async_http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(fake)
    )
    await metric.a_measure(TEST_CASE, _show_indicator=False)

    assert metric.judge_responses == [STEPS_CONTENT, SCORE_CONTENT]
    assert [e["raw_response_body"] for e in metric.judge_exchanges] == [
        body for _, body, _ in fake.exchanges
    ]


def test_direct_model_calls_are_unchanged(fake):
    # No capture scope: the model does not switch to raw-response mode.
    metric = make_metric(fake)
    output, _ = metric.model.generate("hello")
    assert output == STEPS_CONTENT
    assert metric.judge_prompts is None


def test_provider_metadata_is_captured(fake):
    metric = make_metric(fake)
    metric.measure(TEST_CASE)

    for exchange, (_, body, _) in zip(metric.judge_exchanges, fake.exchanges):
        wire = json.loads(body)
        assert exchange["served_model"] == wire["model"]
        assert exchange["response_id"] == wire["id"]
        assert exchange["system_fingerprint"] == wire["system_fingerprint"]
        assert exchange["finish_reason"] == "stop"
        assert exchange["usage"] == wire["usage"]
        assert exchange["request_model"] == "gpt-4.1-mini"
        assert exchange["http_status"] == 200


def test_geval_score_recomputes_from_archive(fake):
    metric = make_metric(fake)
    metric.measure(TEST_CASE)

    exchange = metric.judge_exchanges[-1]
    assert exchange["logprobs"]["content"][0]["top_logprobs"]
    raw_score = json.loads(exchange["response_content"])["score"]
    weighted = calculate_weighted_summed_score(
        raw_score, rebuild_chat_completion(exchange)
    )
    low, high = metric.score_range
    assert weighted == pytest.approx(7.4)
    assert metric.score == pytest.approx((weighted - low) / (high - low))
    # The integer alone would not have recovered the reported score.
    assert metric.score != pytest.approx((raw_score - low) / (high - low))


def test_retry_keeps_last_attempt(fake):
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                text='{"not": "a chat completion"}',
                headers={"x-request-id": "req_bad"},
            )
        return fake(request)

    metric = make_metric(fake)
    metric.model.kwargs["http_client"] = httpx.Client(
        transport=httpx.MockTransport(flaky)
    )
    metric.measure(TEST_CASE)
    # The bad attempt reached the provider but its result was not kept.
    assert metric.judge_exchanges[0]["discarded_attempts"] == 1
    for exchange in metric.judge_exchanges:
        assert exchange["request_id"] != "req_bad"
        assert exchange["response_content"] is not None
    assert metric.judge_responses[0] == STEPS_CONTENT


def run_eval(fake, **geval_kwargs):
    metric = make_metric(fake, **geval_kwargs)
    result = evaluate(
        test_cases=[TEST_CASE],
        metrics=[metric],
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(use_cache=True, write_cache=True),
        display_config=DisplayConfig(
            show_indicator=False, print_results=False
        ),
    )
    return metric, result.test_results[0].metrics_data[0]


@pytest.fixture
def fresh_cache(monkeypatch):
    manager = global_test_run_cache_manager
    manager.cached_test_run = None
    manager.temp_cached_test_run = None
    manager.disable_write_cache = None
    yield manager
    manager.cached_test_run = None
    manager.temp_cached_test_run = None


def test_cache_hit_is_marked_replay(fake, fresh_cache):
    _, cold = run_eval(fake)
    assert cold.judge_capture_status == LIVE
    calls_after_cold = len(fake.exchanges)

    metric, warm = run_eval(fake)
    assert len(fake.exchanges) == calls_after_cold  # no judge was called
    assert warm.judge_capture_status == REPLAY
    assert all(e["status"] == REPLAY for e in warm.judge_exchanges)
    assert warm.judge_prompts == cold.judge_prompts
    assert metric.judge_capture_status == REPLAY


@pytest.fixture
def provenance_mode(monkeypatch):
    from deepeval.config.settings import reset_settings

    monkeypatch.setenv("DEEPEVAL_JUDGE_PROVENANCE", "1")
    reset_settings(reload_dotenv=False)
    yield


RUBRIC_A = [Rubric(score_range=(0, 10), expected_outcome="Fully correct.")]
RUBRIC_B = [
    Rubric(score_range=(0, 10), expected_outcome="Correct and concise.")
]


def score_prompt(fake):
    return fake.exchanges[-1][0]["messages"][-1]["content"][0]["text"]


def test_rubric_change_is_a_stale_hit_without_provenance_mode(
    fake, fresh_cache
):
    # Upstream behaviour, kept when the flag is off: the rubric is not part
    # of the cache key, so the result for rubric A is served for rubric B.
    run_eval(fake, rubric=RUBRIC_A)
    calls = len(fake.exchanges)
    _, warm = run_eval(fake, rubric=RUBRIC_B)
    assert len(fake.exchanges) == calls
    assert warm.judge_capture_status == REPLAY  # at least it says so


def test_rubric_change_misses_cache_in_provenance_mode(
    fake, fresh_cache, provenance_mode
):
    run_eval(fake, rubric=RUBRIC_A)
    prompt_a = score_prompt(fake)
    calls = len(fake.exchanges)

    _, same = run_eval(fake, rubric=RUBRIC_A)
    assert len(fake.exchanges) == calls
    assert same.judge_capture_status == REPLAY

    _, changed = run_eval(fake, rubric=RUBRIC_B)
    assert len(fake.exchanges) == calls + 2
    assert changed.judge_capture_status == LIVE
    prompt_b = score_prompt(fake)
    assert "Correct and concise." in prompt_b
    assert "Correct and concise." not in prompt_a
    assert changed.judge_prompts[-1] == prompt_b


def test_top_logprobs_change_misses_cache_in_provenance_mode(
    fake, fresh_cache, provenance_mode
):
    run_eval(fake, top_logprobs=20)
    calls = len(fake.exchanges)
    _, changed = run_eval(fake, top_logprobs=5)
    assert len(fake.exchanges) == calls + 2
    assert fake.exchanges[-1][0]["top_logprobs"] == 5
    assert changed.judge_capture_status == LIVE


# Two template classes in the same, unchanged file.
class TemplateA(GEvalTemplate):
    @staticmethod
    def generate_evaluation_results(
        evaluation_steps, test_case_content, parameters, **kwargs
    ):
        return (
            f"TEMPLATE A\n{evaluation_steps}\n{test_case_content}\n"
            'Return JSON {"reason": str, "score": int}.'
        )


class TemplateB(GEvalTemplate):
    @staticmethod
    def generate_evaluation_results(
        evaluation_steps, test_case_content, parameters, **kwargs
    ):
        return (
            f"TEMPLATE B: be strict\n{evaluation_steps}\n{test_case_content}\n"
            'Return JSON {"reason": str, "score": int}.'
        )


def test_template_class_change_misses_cache_in_provenance_mode(
    fake, fresh_cache, provenance_mode
):
    assert template_fingerprint(
        make_metric(fake, evaluation_template=TemplateA)
    ) != template_fingerprint(make_metric(fake, evaluation_template=TemplateB))

    _, cold = run_eval(fake, evaluation_template=TemplateA)
    assert cold.judge_prompts[-1].startswith("TEMPLATE A")
    calls = len(fake.exchanges)

    _, warm = run_eval(fake, evaluation_template=TemplateA)
    assert len(fake.exchanges) == calls
    assert warm.judge_capture_status == REPLAY

    _, changed = run_eval(fake, evaluation_template=TemplateB)
    assert len(fake.exchanges) == calls + 2
    assert changed.judge_capture_status == LIVE
    assert score_prompt(fake).startswith("TEMPLATE B")
    assert changed.judge_prompts[-1] == score_prompt(fake)


def test_fingerprint_ignores_values_derived_during_measure(fake):
    # GEval generates evaluation_steps during measure(); a warm lookup (fresh
    # instance) and a cache write (used instance) must still agree.
    metric = make_metric(fake)
    before = template_fingerprint(metric)
    metric.measure(TEST_CASE)
    assert metric.evaluation_steps
    assert template_fingerprint(metric) == before

    model = make_metric(fake).model
    inferred = TaskCompletionMetric(model=model)
    fresh = template_fingerprint(inferred)
    inferred.task = "a task the judge inferred"
    assert template_fingerprint(inferred) == fresh

    given = TaskCompletionMetric(model=model, task="Book a flight.")
    other = TaskCompletionMetric(model=model, task="Book a hotel.")
    assert template_fingerprint(given) != template_fingerprint(other)


def test_configuration_snapshot_covers_prompt_affecting_fields(fake):
    snapshot = metric_configuration_snapshot(
        make_metric(fake, rubric=RUBRIC_A, evaluation_template=TemplateA)
    )
    assert snapshot["rubric"][0]["expected_outcome"] == "Fully correct."
    assert snapshot["top_logprobs"] == 20
    assert snapshot["criteria"] == "Is the actual output correct?"
    assert snapshot["model"]["name"] == "gpt-4.1-mini"
    assert snapshot["model"]["temperature"] == 0.0
    template_names = [c["class"] for c in snapshot["evaluation_template"]]
    assert template_names[0].endswith(".TemplateA")
    assert "evaluation_steps" not in snapshot


@pytest.mark.parametrize("async_mode", [False, True])
def test_repeated_direct_measure_starts_fresh(fake, async_mode):
    metric = make_metric(fake, async_mode=async_mode)
    if async_mode:
        metric.model.async_http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(fake)
        )
    metric.measure(TEST_CASE)
    first = list(metric.judge_prompts)
    assert len(first) == 2

    metric.measure(TEST_CASE)
    # evaluation_steps are reused on the second run, so only the score call.
    assert len(metric.judge_prompts) == 1
    assert len(metric.judge_responses) == 1
    assert len(metric.judge_exchanges) == 1
    assert metric.judge_prompts[0] == first[1]
    assert metric._judge_capture_depth == 0


@pytest.mark.asyncio
async def test_repeated_direct_a_measure_starts_fresh(fake):
    metric = make_metric(fake, async_mode=True)
    metric.model.async_http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(fake)
    )
    await metric.a_measure(TEST_CASE, _show_indicator=False)
    await metric.a_measure(TEST_CASE, _show_indicator=False)
    assert len(metric.judge_prompts) == 1


def test_measure_that_raises_still_resets_depth(fake):
    metric = make_metric(fake)
    missing_output = LLMTestCase(input="x", actual_output=None)
    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(missing_output)
    assert metric._judge_capture_depth == 0
    metric.measure(TEST_CASE)
    assert len(metric.judge_prompts) == 2


def test_fingerprint_tracks_runtime_template_bundle(tmp_path, monkeypatch):
    metric = GEval(
        name="x",
        criteria="y",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        model=OpenAIModel(model="gpt-4.1-mini", api_key="sk-test"),
    )
    source = deepeval.templates.__file__.rsplit("/", 1)[0]
    copy = tmp_path / "deepeval" / "templates"
    shutil.copytree(source, copy)
    monkeypatch.setattr(
        deepeval.templates, "__file__", str(copy / "__init__.py")
    )
    before = template_fingerprint(metric)
    with open(copy / "metrics" / "templates.json", "a") as f:
        f.write(" ")
    assert template_fingerprint(metric) != before
