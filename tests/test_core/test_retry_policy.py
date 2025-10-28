import logging
import pytest
import tenacity
import time


from deepeval.models import retry_policy as rp
from deepeval.models.retry_policy import (
    create_retry_decorator,
    dynamic_wait,
    dynamic_stop,
    ErrorPolicy,
    extract_error_code,
    get_retry_policy_for,
    make_is_transient,
    sdk_retries_for,
)


##############################################
# Dummy exception shapes for offline testing #
##############################################


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class RaisingResponse:
    def json(self):
        raise ValueError("boom")


class AuthError(Exception): ...


class RateLimitError(Exception):
    def __init__(self, *, response=None, body=None, msg=""):
        super().__init__(msg)
        self.response = response
        self.body = body


class FakeClientError(Exception):
    def __init__(self, response):
        self.response = response


class NetTimeout(Exception): ...


class NetConn(Exception): ...


class HTTPStatusError(Exception):
    def __init__(self, status_code, *, msg=""):
        super().__init__(msg)
        self.status_code = status_code


OPENAI_MARKERS = {
    "insufficient_quota": ("insufficient_quota", "exceeded your current quota"),
}


def make_policy():
    return ErrorPolicy(
        auth_excs=(AuthError,),
        rate_limit_excs=(RateLimitError,),
        network_excs=(NetTimeout, NetConn),
        http_excs=(HTTPStatusError,),
        non_retryable_codes=frozenset({"insufficient_quota"}),
        message_markers=OPENAI_MARKERS,
    )


def RL(response=None, body=None, msg=""):
    """Helper to build a RateLimitError succinctly."""
    return RateLimitError(response=response, body=body, msg=msg)


################
# Fixtures
################


@pytest.fixture
def policy():
    return make_policy()


@pytest.fixture
def pred(policy):
    return make_is_transient(policy)


############################
# extract_error_code tests #
############################


@pytest.mark.parametrize(
    "response, body, msg, expected",
    [
        # response.json() -> structured code
        (
            DummyResponse({"error": {"code": "insufficient_quota"}}),
            None,
            "",
            "insufficient_quota",
        ),
        # body dict path
        (None, {"error": {"code": "throttle"}}, "", "throttle"),
        # numeric codes are stringified
        (DummyResponse({"error": {"code": 42}}), None, "", "42"),
        (DummyResponse({"error": {"code": 0}}), None, "", "0"),
        # message markers fallback
        (
            None,
            None,
            "You have exceeded your current quota.",
            "insufficient_quota",
        ),
        # missing -> empty
        (None, None, "", ""),
        # traversal breaks gracefully when shape is wrong
        (DummyResponse({"error": "oops"}), None, "", ""),
        # response.json() raises -> fall back to markers
        (
            RaisingResponse(),
            None,
            "exceeded your current quota",
            "insufficient_quota",
        ),
        # body not a dict -> ignored
        (None, ["not-a-dict"], "", ""),
    ],
    ids=[
        "resp-json",
        "body-dict",
        "numeric-42",
        "numeric-0",
        "markers-fallback",
        "missing",
        "bad-shape",
        "json-raises->markers",
        "body-not-dict",
    ],
)
def test_extract_error_code_variants(response, body, msg, expected):
    e = RL(response=response, body=body, msg=msg)
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == expected


def test_extract_code_botocore_shape():
    # extract code from response with "Error" -> "Code" (botocore ClientError)
    e = FakeClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "..."}}
    )
    assert extract_error_code(e) == "ThrottlingException"


def test_extract_error_code_prefers_response_over_markers():
    # Response has code, but message also contains marker text. Response should win.
    e = RL(
        response=DummyResponse({"error": {"code": "throttle"}}),
        msg="exceeded your current quota",
    )
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == "throttle"


def test_extract_error_code_grpc_code_lowercased():
    # Simulate grpc-style .code().name
    class DummyGrpcStatus:
        def __init__(self, name):
            self.name = name

    class DummyGrpcError(Exception):
        def code(self):
            return DummyGrpcStatus("UNAVAILABLE")

    assert extract_error_code(DummyGrpcError()) == "unavailable"


def test_extract_error_code_prefers_response_over_body():
    e = RL(
        response=DummyResponse({"error": {"code": "resp_code"}}),
        body={"error": {"code": "body_code"}},
    )
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == "resp_code"


##########################################
# make_is_transient classification tests #
##########################################


@pytest.mark.parametrize(
    "exc", [NetTimeout(), NetConn()], ids=["timeout", "conn"]
)
def test_network_is_retry(pred, exc):
    assert pred(exc) is True


@pytest.mark.parametrize(
    "exc, expected",
    [
        (HTTPStatusError(500), True),  # 5xx -> retry
        (HTTPStatusError(400), False),  # 4xx -> no retry
        (AuthError(), False),  # auth -> no retry
    ],
)
def test_core_paths(pred, exc, expected):
    assert pred(exc) is expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ("other", True),
        ("insufficient_quota", False),  # non-retryable by policy
    ],
)
def test_rate_limit_codes(policy, code, expected):
    pred = make_is_transient(policy)
    e = RL(response=DummyResponse({"error": {"code": code}}))
    assert pred(e) is expected


def test_extra_non_retryable_codes(policy):
    pred = make_is_transient(
        policy, extra_non_retryable_codes=("soft_throttle",)
    )
    e = RL(body={"error": {"code": "soft_throttle"}})
    assert pred(e) is False


def test_http_status_non_int_or_missing_means_no_retry(policy):
    class WeirdHTTP(Exception):
        pass

    # Treat WeirdHTTP as an HTTP error, but it lacks a `status_code` attribute.
    weird_policy = ErrorPolicy(
        auth_excs=policy.auth_excs,
        rate_limit_excs=policy.rate_limit_excs,
        network_excs=policy.network_excs,
        http_excs=(WeirdHTTP,),  # no status_code -> should not retry
        non_retryable_codes=policy.non_retryable_codes,
        retry_5xx=True,
        message_markers=policy.message_markers,
    )
    weird_pred = make_is_transient(weird_policy)
    assert weird_pred(WeirdHTTP()) is False


def test_retry_5xx_false_disables_server_retries(policy):
    p = ErrorPolicy(
        auth_excs=policy.auth_excs,
        rate_limit_excs=policy.rate_limit_excs,
        network_excs=policy.network_excs,
        http_excs=policy.http_excs,
        non_retryable_codes=policy.non_retryable_codes,
        retry_5xx=False,
        message_markers=policy.message_markers,
    )
    pred = make_is_transient(p)
    assert pred(HTTPStatusError(500)) is False


def test_message_markers_override_policy_markers(policy):
    custom_markers = {"custom_code": ("special sentinel",)}
    pred = make_is_transient(policy, message_markers=custom_markers)
    e = RL(msg="SPECIAL SENTINEL present")
    # Lowercasing inside extract => match
    assert (
        extract_error_code(e, message_markers=custom_markers) == "custom_code"
    )
    # Not in non-retryable set, so it retries
    assert pred(e) is True


############################################
# dynamic_wait / dynamic_stop construction #
############################################


def test_dynamic_wait_callable(monkeypatch):
    # sanity-check callability.
    w = dynamic_wait()
    assert callable(w)


def test_dynamic_wait_zeros_with_env(monkeypatch, settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_CAP_SECONDS = 0

    w = dynamic_wait()

    class RS:  # minimal retry state shape
        attempt_number = 1

    assert w(RS()) == 0


def test_dynamic_stop_callable():
    s = dynamic_stop()
    assert callable(s)


##############################################
# Retry decorator & dynamic policy tests     #
##############################################


def test_retry_respects_max_attempts_env(monkeypatch, policy, settings):
    slug = "max_attempts"
    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)
    monkeypatch.setitem(
        rp._STATIC_PRED_BY_SLUG, slug, rp.make_is_transient(policy)
    )
    # Ensure SDK retries are OFF so Tenacity predicate is used
    monkeypatch.setattr(rp, "sdk_retries_for", lambda s: False, raising=True)

    # Case 1
    # allow only 2 attempts, let the function fails twice, then cap is hit
    calls = {"n": 0}

    @create_retry_decorator(slug)
    def flaky_twice_then_ok():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise NetTimeout()
        return "ok"

    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 2

    with pytest.raises(tenacity.RetryError):
        flaky_twice_then_ok()
    assert calls["n"] == 2  # stopped at the cap

    # Case 2
    # allow 3 attempts, now it can succeed on the 3rd call because cap was increased
    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 3

    calls["n"] = 0

    assert flaky_twice_then_ok() == "ok"
    assert calls["n"] == 3


def test_create_retry_decorator_no_retry_when_sdk_enabled(monkeypatch, policy):
    """
    When SDK retries are enabled for the slug, our Tenacity predicate must
    short-circuit (no retries). We expect the original exception after exactly one call.
    """
    slug = "sdk_on"

    # Register a policy/predicate for the slug (not strictly needed, but harmless)
    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)
    monkeypatch.setitem(
        rp._STATIC_PRED_BY_SLUG, slug, rp.make_is_transient(policy)
    )

    # Critical: force the dynamic predicate to see SDK retries enabled
    monkeypatch.setattr(
        rp, "sdk_retries_for", lambda s: s == slug, raising=True
    )

    calls = {"n": 0}

    @create_retry_decorator(slug)
    def always_transient():
        calls["n"] += 1
        raise NetTimeout()

    with pytest.raises(NetTimeout):
        always_transient()

    # No retries performed: one call, inner exc is NetTimeout
    assert calls["n"] == 1


def test_dynamic_retry_no_policy_means_no_retry(monkeypatch):
    """
    If no policy exists (and SDK retries are not enabled), dynamic predicate
    must not retry. Expect the original exception after a single call.
    """
    slug = "no_policy"

    # Ensure no policy or static predicate registered
    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, None)
    monkeypatch.setitem(rp._STATIC_PRED_BY_SLUG, slug, None)

    # Ensure SDK retries are "off" for this slug
    monkeypatch.setattr(rp, "sdk_retries_for", lambda s: False, raising=True)

    calls = {"n": 0}

    @create_retry_decorator(slug)
    def fails():
        calls["n"] += 1
        raise NetTimeout()

    with pytest.raises(NetTimeout):
        fails()

    assert calls["n"] == 1


def test_get_retry_policy_for_respects_sdk_retries_for(monkeypatch, policy):
    slug = "any-slug"

    # Ensure policy is available for this slug
    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)

    # SDK disabled -> returns policy
    monkeypatch.setattr(rp, "sdk_retries_for", lambda s: False, raising=True)
    assert get_retry_policy_for(slug) is policy

    # SDK enabled for this slug -> returns None
    monkeypatch.setattr(
        rp, "sdk_retries_for", lambda s: s == slug, raising=True
    )
    assert get_retry_policy_for(slug) is None


def test_sdk_retries_for_wildcard(monkeypatch, settings):
    with settings.edit(persist=False):
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = ["*"]

    assert sdk_retries_for("anything") is True
    assert sdk_retries_for("azure") is True


def test_http_status_string_is_coerced_to_int(policy):
    # build a policy that treats StringStatus as an HTTP error with string status_code
    class StringStatus(Exception):
        def __init__(self, sc):
            self.status_code = sc

    p = ErrorPolicy(
        auth_excs=policy.auth_excs,
        rate_limit_excs=policy.rate_limit_excs,
        network_excs=policy.network_excs,
        http_excs=(StringStatus,),
        non_retryable_codes=policy.non_retryable_codes,
        retry_5xx=True,
        message_markers=policy.message_markers,
    )
    pred = rp.make_is_transient(p)
    assert pred(StringStatus("500")) is True
    assert pred(StringStatus("400")) is False


def test_dynamic_retry_invokes_static_predicate_when_sdk_off(
    monkeypatch, policy
):
    """
    Verify that when SDK is disabled, our dynamic predicate calls the static predicate.
    """
    slug = "static_pred_used"
    calls = {"seen": 0}

    def static_pred(exc: Exception) -> bool:
        calls["seen"] += 1
        # Pretend everything is transient (would cause retries if not limited)
        return True

    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)
    monkeypatch.setitem(rp._STATIC_PRED_BY_SLUG, slug, static_pred)
    monkeypatch.setattr(rp, "sdk_retries_for", lambda s: False, raising=True)

    @create_retry_decorator(slug)
    def boom():
        raise NetTimeout()

    with pytest.raises(tenacity.RetryError):
        boom()

    assert calls["seen"] >= 1  # static predicate was consulted


def test_dynamic_retry_does_not_call_static_predicate_when_sdk_on(
    monkeypatch, policy
):
    """
    Verify that when SDK is enabled, our static predicate is never consulted.
    """
    slug = "static_pred_bypassed"
    calls = {"seen": 0}

    def static_pred(_exc: Exception) -> bool:
        calls["seen"] += 1
        return True

    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)
    monkeypatch.setitem(rp._STATIC_PRED_BY_SLUG, slug, static_pred)
    monkeypatch.setattr(
        rp,
        "sdk_retries_for",
        lambda s: True if s == slug else False,
        raising=True,
    )

    @create_retry_decorator(slug)
    def boom():
        raise NetTimeout()

    with pytest.raises(NetTimeout):
        boom()

    assert calls["seen"] == 0  # never consulted


def test_sync_timeout_is_retryable_and_capped(monkeypatch, policy, settings):
    slug = "openai"
    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)
    monkeypatch.setitem(
        rp._STATIC_PRED_BY_SLUG, slug, make_is_transient(policy)
    )

    calls = {"n": 0}

    @create_retry_decorator(slug)
    def slow():
        calls["n"] += 1
        time.sleep(0.05)  # longer than per-attempt timeout

    with settings.edit(persist=False):
        settings.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = (
            0.01  # force per-attempt timeout
        )
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 3
        settings.DEEPEVAL_RETRY_CAP_SECONDS = 0  # keep the test fast

    with pytest.raises(tenacity.RetryError):
        slow()

    # We should have hit the cap: 1 initial + (max_attempts-1) retries => attempts == 3
    assert calls["n"] == 3


def test_dynamic_toggle_sdk_retries_runtime(monkeypatch, policy, settings):
    slug = "openai"
    # register policy + static predicate
    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)
    monkeypatch.setitem(
        rp._STATIC_PRED_BY_SLUG, slug, make_is_transient(policy)
    )

    calls = {"n": 0}

    @create_retry_decorator(slug)
    def flaky():
        calls["n"] += 1
        raise NetTimeout()

    # SDK off -> Tenacity should retry up to cap
    with settings.edit(persist=False):
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = []
        settings.DEEPEVAL_RETRY_MAX_ATTEMPTS = 3
        settings.DEEPEVAL_RETRY_CAP_SECONDS = 0

    with pytest.raises(tenacity.RetryError):
        flaky()
    assert calls["n"] == 3

    # SDK on -> no retries; same wrapped function
    calls["n"] = 0
    with settings.edit(persist=False):
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = ["openai"]  # on for this slug

    with pytest.raises(NetTimeout):
        flaky()
    assert calls["n"] == 1


###############
# Diagnostics #
###############


@pytest.mark.skip(
    reason="Needs update: exc_info now controlled by settings.DEEPEVAL_LOG_STACK_TRACES (not log level)."
)
def test_retry_logging_levels_change_at_runtime(
    monkeypatch, caplog, policy, settings
):
    slug = "log_levels"
    monkeypatch.setitem(rp._POLICY_BY_SLUG, slug, policy)
    monkeypatch.setitem(
        rp._STATIC_PRED_BY_SLUG, slug, rp.make_is_transient(policy)
    )
    monkeypatch.setattr(rp, "sdk_retries_for", lambda s: False, raising=True)

    @create_retry_decorator(slug)
    def boom():
        raise NetTimeout()

    # Before: WARNING for before-sleep, ERROR for after
    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL = logging.WARNING
        settings.DEEPEVAL_RETRY_AFTER_LOG_LEVEL = logging.ERROR

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=f"deepeval.retry.{slug}"):
        with pytest.raises(tenacity.RetryError):  # <- expect RetryError
            boom()

    # There should be an ERROR "after" record, and no INFO-level records
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert not any(r.levelno == logging.INFO for r in caplog.records)
    assert not any(r.levelno == logging.DEBUG for r in caplog.records)
    assert all(
        (r.exc_info is None) == (r.levelno < logging.ERROR)
        for r in caplog.records
    )

    # After: INFO for before-sleep, DEBUG for after (no traceback at DEBUG)
    with settings.edit(persist=False):
        settings.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL = logging.INFO
        settings.DEEPEVAL_RETRY_AFTER_LOG_LEVEL = logging.DEBUG

    caplog.clear()
    # Ensure we have at least 2 attempts so before_sleep runs.
    monkeypatch.setenv("DEEPEVAL_RETRY_MAX_ATTEMPTS", "2")
    with caplog.at_level(logging.DEBUG, logger=f"deepeval.retry.{slug}"):
        with pytest.raises(tenacity.RetryError):
            boom()

    # Both INFO (before) and DEBUG (after) should appear
    assert any(r.levelno == logging.INFO for r in caplog.records)
    assert any(r.levelno == logging.DEBUG for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    assert all(r.exc_info is None for r in caplog.records)
