"""Provider configuration and real loopback HTTP; no external model calls."""

import asyncio
import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from deepeval.config.settings import Settings
from deepeval.errors import DeepEvalError
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, JevEval
from deepeval.metrics.jev_eval import Noul
from deepeval.metrics.jev_eval.utils import initialize_jev_model
from deepeval.metrics.utils.models import initialize_system_one_model
from deepeval.metrics.utils.system_one import _decide, _a_decide
from deepeval.metrics.utils.system_one_batch import (
    measure_system_one_batch,
    a_measure_system_one_batch,
)
from deepeval.models import SystemOneAPIModel
from deepeval.models.system_one import provider
from deepeval.models.system_one.limits import SystemOneContextLimitError
from deepeval.models.system_one.schema import (
    NoulQuestion,
    ChoiceQuestion,
    ScoreQuestion,
)
from deepeval.models.utils import EvaluationCost
from deepeval.test_case import LLMTestCase, SingleTurnParams
from tests.test_metrics.system_one_fakes import ExplodingLLM, FakeSystemOneModel


QUESTIONS = {
    "safe": NoulQuestion(
        instructions="Is the answer safe?", true="Safe", false="Unsafe"
    ),
    "category": ChoiceQuestion(
        instructions="Classify", options={"yes": "Relevant", "no": "Irrelevant"}
    ),
    "quality": ScoreQuestion(
        instructions="Rate", levels=["Poor", "Good", "Excellent"]
    ),
}
RESPONSE = {
    "answers": {
        "safe": {"type": "noul", "noul": 0.9},
        "category": {
            "type": "choice",
            "choice": "yes",
            "probabilities": {"yes": 0.8, "no": 0.2},
            "confidence": 0.6,
        },
        "quality": {
            "type": "score",
            "score": 1.5,
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            "confidence": 0.4,
        },
    },
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


def answer_payload(payload):
    answers = {}
    for key, question in payload["questions"].items():
        kind = question["type"]
        if kind == "noul":
            answers[key] = {"type": kind, "noul": 0.9}
        elif kind == "choice":
            options = list(question["criteria"])
            answers[key] = {
                "type": kind,
                "choice": options[0],
                "probabilities": {
                    option: float(i == 0) for i, option in enumerate(options)
                },
                "confidence": 1.0,
            }
        else:
            top = len(question["criteria"]) - 1
            answers[key] = {
                "type": kind,
                "score": top,
                "probabilities": {
                    str(i): float(i == top) for i in range(top + 1)
                },
                "confidence": 1.0,
            }
    return {"answers": answers}


@pytest.fixture
def server():
    config = {"status": 200, "body": RESPONSE, "calls": [], "delay": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            config["calls"].append((self.path, dict(self.headers), payload))
            time.sleep(config["delay"])
            body = config["body"]
            if callable(body):
                body = body(payload)
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(config["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Location", "/redirect-target")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    config["url"] = f"http://127.0.0.1:{httpd.server_port}"
    yield config
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def api(server=None, **kwargs):
    options = {
        "model": "local-decision",
        "base_url": server["url"] if server else "http://localhost:8009",
    }
    options.update(kwargs)
    return SystemOneAPIModel(**options)


def configure(monkeypatch, **kwargs):
    settings = Settings(_env_file=None, **kwargs)
    monkeypatch.setattr(provider, "get_settings", lambda: settings)
    return settings


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_http_contract_and_cost(server, is_async):
    model = api(
        server,
        api_key="test-secret",
        cost_per_input_token=0.01,
        cost_per_output_token=0.02,
    )
    answers, cost = (
        await model.a_decide({"text": "hello"}, QUESTIONS)
        if is_async
        else model.decide({"text": "hello"}, QUESTIONS)
    )
    assert answers.nouls["safe"].probability == 0.9
    assert answers.choices["category"].choice == "yes"
    assert answers.scores["quality"].normalized == 0.75
    assert isinstance(cost, EvaluationCost)
    assert cost == pytest.approx(0.14)
    assert (cost.input_tokens, cost.output_tokens) == (10, 2)
    path, headers, payload = server["calls"][0]
    assert path == "/v1/systemone"
    assert headers["Authorization"] == "Bearer test-secret"
    assert payload["model"] == "local-decision"
    assert payload["state"] == {"text": "hello"}
    assert payload["questions"]["safe"]["criteria"] == {
        "true": "Safe",
        "false": "Unsafe",
    }
    assert (
        payload["questions"]["category"]["criteria"]
        == QUESTIONS["category"].options
    )
    assert (
        payload["questions"]["quality"]["criteria"]
        == QUESTIONS["quality"].levels
    )


@pytest.mark.parametrize(
    "auth,key,header",
    [
        ("none", None, None),
        ("api_key", "secret", "X-API-Key"),
        ("bearer", "secret", "Authorization"),
    ],
)
def test_auth_and_endpoint(server, auth, key, header):
    api(server, auth=auth, api_key=key, endpoint="/decision").decide(
        {}, QUESTIONS
    )
    path, headers, _ = server["calls"][0]
    assert path == "/decision"
    assert (
        (header in headers)
        if header
        else not ({"Authorization", "X-API-Key"} & set(headers))
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "failure",
    ["redirect", "unauthorized", "server", "oversize", "json", "timeout"],
)
@pytest.mark.asyncio
async def test_transport_fails_without_retry_or_secret_leak(
    server, is_async, failure
):
    options = {}
    if failure in {"redirect", "unauthorized", "server"}:
        server["status"] = {
            "redirect": 307,
            "unauthorized": 401,
            "server": 503,
        }[failure]
        server["body"] = b"private response test-secret"
    elif failure == "oversize":
        options["max_response_bytes"] = 10
    elif failure == "json":
        server["body"] = b"private invalid JSON test-secret"
    else:
        options["timeout"] = 0.03
        server["delay"] = 0.1
    model = api(server, api_key="test-secret", **options)
    with pytest.raises(DeepEvalError) as error:
        if is_async:
            await model.a_decide({}, QUESTIONS)
        else:
            model.decide({}, QUESTIONS)
    assert len(server["calls"]) == 1
    assert "private" not in str(error.value)
    assert "test-secret" not in str(error.value)
    assert server["url"] not in str(error.value)


@pytest.mark.parametrize(
    "path,value",
    [
        (("answers",), {}),
        (("answers", "extra"), {}),
        (("answers", "safe", "type"), "choice"),
        (("answers", "safe", "noul"), True),
        (("answers", "safe", "noul"), float("nan")),
        (("answers", "safe", "noul"), -0.1),
        (("answers", "safe", "noul"), 1.1),
        (("answers", "safe", "noul"), "0.9"),
        (("answers", "category", "probabilities"), {"yes": 1.0}),
        (("answers", "category", "probabilities"), {"yes": 0.1, "no": 0.1}),
        (("answers", "category", "choice"), "no"),
        (("answers", "category", "confidence"), None),
        (("answers", "quality", "score"), 2),
        (
            ("answers", "quality", "probabilities"),
            {"0": 0.5, "01": 0.5, "2": 0},
        ),
        (("truncated",), True),
        (("truncated",), 0),
        (("usage", "input_tokens"), True),
        (("usage", "output_tokens"), -1),
        (("usage",), {"state_tokens": 10, "state_tokens_used": 5}),
    ],
)
def test_malformed_answers_are_rejected(path, value):
    body = copy.deepcopy(RESPONSE)
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(DeepEvalError):
        api()._parse(json.dumps(body).encode(), QUESTIONS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"base_url": "file:///tmp/model"},
        {"base_url": "https://user:secret@example.com"},
        {"base_url": "https://example.com?key=secret"},
        {"endpoint": "//other-host/path"},
        {"endpoint": "https://other-host"},
        {"auth": "unknown"},
        {"auth": "bearer"},
        {"auth": "none", "api_key": "secret"},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"max_state_tokens": -1},
        {"max_response_bytes": 0},
        {"question_types": []},
        {"question_types": ["chat"]},
        {"cost_per_input_token": float("nan")},
        {"cost_per_output_token": -1},
        {"api_key": "secret\nheader"},
        {"timeout": 10**1000},
    ],
)
def test_bad_configuration_fails_before_network(kwargs):
    with pytest.raises(DeepEvalError):
        api(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "http://localhost:8010"},
        {"endpoint": "/other"},
        {"api_key": "another-secret"},
        {"cost_per_input_token": 0},
        {"max_state_tokens": 200},
        {"question_types": ["noul"]},
    ],
)
def test_identity_is_stable_and_isolates_configuration(kwargs):
    first = api(api_key="secret-one")
    second = api(**dict({"api_key": "secret-one"}, **kwargs))
    assert first.get_model_name() == api(api_key="secret-one").get_model_name()
    assert first.get_model_name() != second.get_model_name()
    assert "secret-one" not in first.get_model_name()
    assert "localhost" not in first.get_model_name()


def test_unknown_cost_is_not_zero():
    assert api()._parse(json.dumps(RESPONSE).encode(), QUESTIONS)[1] is None
    model = api(cost_per_input_token=0, cost_per_output_token=0)
    assert model._parse(json.dumps(RESPONSE).encode(), QUESTIONS)[1] == 0
    body = dict(RESPONSE, usage={"input_tokens": 10})
    assert model._parse(json.dumps(body).encode(), QUESTIONS)[1] is None


def test_hybrid_cache_distinguishes_decision_providers_and_modes():
    from deepeval.test_run.cache import Cache

    def metric(decision, mode="hybrid"):
        return AnswerRelevancyMetric(
            model=ExplodingLLM(), system_one_model=decision, eval_mode=mode
        )

    first = metric(api())
    cached = Cache.create_metric_configuration(first)
    assert Cache.same_metric_configs(metric(api()), cached)
    assert not Cache.same_metric_configs(
        metric(api(base_url="http://localhost:8010")), cached
    )
    assert not Cache.same_metric_configs(metric(api(), "llm"), cached)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_distinct_provider_credentials_do_not_share_a_batch(
    server, is_async
):
    server["body"] = answer_payload
    case = LLMTestCase(
        input="Where?", actual_output="Paris", retrieval_context=["Paris"]
    )
    metrics = [
        cls(
            model=ExplodingLLM(),
            system_one_model=api(server, api_key=key),
            eval_mode="system_one",
        )
        for cls, key in (
            (AnswerRelevancyMetric, "tenant-a"),
            (FaithfulnessMetric, "tenant-b"),
        )
    ]
    run = a_measure_system_one_batch if is_async else measure_system_one_batch
    result = run(
        metrics, case, ignore_errors=False, skip_on_missing_params=False
    )
    if is_async:
        await result
    assert len(server["calls"]) == 2
    assert {call[1]["Authorization"] for call in server["calls"]} == {
        "Bearer tenant-a",
        "Bearer tenant-b",
    }


def test_limits_and_capabilities():
    api().check_request("x" * 200_000, QUESTIONS)
    with pytest.raises(SystemOneContextLimitError):
        api(max_state_tokens=5).check_request("x" * 100, QUESTIONS)
    with pytest.raises(SystemOneContextLimitError):
        api(max_request_tokens=5).check_request("x" * 100, QUESTIONS)
    with pytest.raises(DeepEvalError, match="does not support choice"):
        api(question_types=["noul"]).check_request({}, QUESTIONS)
    with pytest.raises(DeepEvalError, match="JSON serializable"):
        api()._payload({"invalid": float("nan")}, QUESTIONS)


def test_duplicate_answer_keys_are_rejected():
    with pytest.raises(DeepEvalError, match="duplicate"):
        api()._parse(b'{"answers":{},"answers":{}}', QUESTIONS)


def test_namespaced_model_name_is_not_rewritten():
    model = api(model="org/decision-model")
    assert (
        json.loads(model._payload({}, QUESTIONS))["model"]
        == "org/decision-model"
    )


@pytest.mark.parametrize("value", ["unknown", "openai"])
def test_unknown_provider_is_not_silently_replaced(value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, DEEPEVAL_SYSTEM_ONE_PROVIDER=value)


def test_default_provider_preserves_typesafe_and_model_precedence(monkeypatch):
    from deepeval.models.system_one import typesafe_model

    calls = []
    monkeypatch.setattr(
        typesafe_model,
        "TypeSafeModel",
        lambda **kw: calls.append(kw) or "typesafe",
    )
    configure(monkeypatch)
    assert provider.create_system_one_model() == "typesafe"
    assert calls[-1] == {"model": None}
    configure(monkeypatch, DEEPEVAL_SYSTEM_ONE_MODEL="configured")
    provider.create_system_one_model()
    provider.create_system_one_model("explicit")
    assert calls[-2:] == [{"model": "configured"}, {"model": "explicit"}]


def test_env_selection_and_explicit_instance(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_SYSTEM_ONE_PROVIDER", "systemone")
    monkeypatch.setenv("DEEPEVAL_SYSTEM_ONE_MODEL", "local-decision")
    monkeypatch.setenv("DEEPEVAL_SYSTEM_ONE_BASE_URL", "http://localhost:8009")
    monkeypatch.setenv("DEEPEVAL_SYSTEM_ONE_QUESTION_TYPES", '["noul"]')
    configure(monkeypatch)
    assert isinstance(
        initialize_system_one_model(eval_mode="system_one"), SystemOneAPIModel
    )
    assert isinstance(initialize_jev_model(), SystemOneAPIModel)
    assert provider.create_system_one_model("override").name == "override"
    custom = FakeSystemOneModel()
    assert provider.create_system_one_model(custom) is custom
    assert initialize_system_one_model(eval_mode="llm") is None
    assert provider.create_system_one_model().question_types == {"noul"}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_configured_jev_eval_and_built_in_batch(
    server, monkeypatch, is_async
):
    server["body"] = answer_payload
    configure(
        monkeypatch,
        DEEPEVAL_SYSTEM_ONE_PROVIDER="systemone",
        DEEPEVAL_SYSTEM_ONE_MODEL="local-decision",
        DEEPEVAL_SYSTEM_ONE_BASE_URL=server["url"],
    )
    case = LLMTestCase(
        input="Where?", actual_output="Paris", retrieval_context=["Paris"]
    )
    metric = JevEval(
        name="Safety",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        questions=[Noul("The answer is safe.")],
        async_mode=False,
    )
    if is_async:
        await metric.a_measure(case, _show_indicator=False)
    else:
        metric.measure(case, _show_indicator=False)
    assert metric.score == 0.9
    metrics = [
        cls(model=ExplodingLLM(), eval_mode="system_one")
        for cls in (AnswerRelevancyMetric, FaithfulnessMetric)
    ]
    run = a_measure_system_one_batch if is_async else measure_system_one_batch
    result = run(
        metrics, case, ignore_errors=False, skip_on_missing_params=False
    )
    if is_async:
        result = await result
    assert result == metrics
    assert len(server["calls"]) == 2
    assert all(m.score is not None for m in metrics)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_runners_honor_provider_budget(server, is_async):
    class Metric:
        system_one_model = api(server)

    metric = Metric()
    # Larger than Jev's default budget, but valid for an unbounded adapter.
    if is_async:
        await _a_decide(metric, "x" * 200_000, QUESTIONS)
    else:
        _decide(metric, "x" * 200_000, QUESTIONS)
    assert len(server["calls"]) == 1


@pytest.mark.asyncio
async def test_async_client_can_be_reused_across_event_loops(server):
    model = api(server)
    await model.a_decide({}, QUESTIONS)
    # A fresh event loop in another thread must not reuse a closed session.
    await asyncio.to_thread(lambda: asyncio.run(model.a_decide({}, QUESTIONS)))
    assert len(server["calls"]) == 2
