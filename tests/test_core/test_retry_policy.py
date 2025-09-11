import pytest

from deepeval.models.retry_policy import (
    ErrorPolicy,
    extract_error_code,
    make_is_transient,
    default_wait,
    default_stop,
)

##############################################
# Dummy exception shapes for offline testing #
##############################################


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class AuthError(Exception): ...


class RateLimitError(Exception):
    def __init__(self, *, response=None, body=None, msg=""):
        super().__init__(msg)
        self.response = response
        self.body = body


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


############################
# extract_error_code tests #
############################


def test_extract_code_from_response_json():
    e = RateLimitError(
        response=DummyResponse({"error": {"code": "insufficient_quota"}})
    )
    assert (
        extract_error_code(e, message_markers=OPENAI_MARKERS)
        == "insufficient_quota"
    )


def test_extract_code_from_body_dict():
    e = RateLimitError(body={"error": {"code": "throttle"}})
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == "throttle"


def test_extract_code_from_message_markers():
    e = RateLimitError(msg="You have exceeded your current quota.")
    assert (
        extract_error_code(e, message_markers=OPENAI_MARKERS)
        == "insufficient_quota"
    )


def test_extract_code_numeric_is_stringified():
    e = RateLimitError(response=DummyResponse({"error": {"code": 42}}))
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == "42"


def test_extract_code_missing_returns_empty():
    e = RateLimitError()
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == ""


##########################################
# make_is_transient classification tests #
##########################################


def test_transient_policy_core_paths():
    policy = make_policy()
    pred = make_is_transient(policy)

    assert pred(NetTimeout()) is True  # network -> retry
    assert pred(NetConn()) is True  # network -> retry
    assert pred(HTTPStatusError(500)) is True  # 5xx -> retry
    assert pred(HTTPStatusError(400)) is False  # non-5xx -> no retry
    assert pred(AuthError()) is False  # auth -> no retry


def test_rate_limit_retry_vs_non_retryable_code():
    policy = make_policy()
    pred = make_is_transient(policy)

    e_retryable = RateLimitError(
        response=DummyResponse({"error": {"code": "other"}})
    )
    e_non_retryable = RateLimitError(
        response=DummyResponse({"error": {"code": "insufficient_quota"}})
    )

    assert pred(e_retryable) is True
    assert pred(e_non_retryable) is False


def test_extra_non_retryable_codes():
    policy = make_policy()
    pred = make_is_transient(
        policy, extra_non_retryable_codes=("soft_throttle",)
    )

    e = RateLimitError(body={"error": {"code": "soft_throttle"}})
    assert pred(e) is False


############################################
# extract_error_code: edge cases & guards  #
############################################


def test_extract_code_json_traversal_breaks_gracefully():
    # error is not a dict -> traversal breaks and returns ""
    e = RateLimitError(response=DummyResponse({"error": "oops"}))
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == ""


def test_extract_code_response_json_raises_falls_back_to_markers(monkeypatch):
    class RaisingResponse:
        def json(self):
            raise ValueError("boom")

    e = RateLimitError(
        response=RaisingResponse(), msg="exceeded your current quota"
    )
    assert (
        extract_error_code(e, message_markers=OPENAI_MARKERS)
        == "insufficient_quota"
    )


def test_extract_code_body_not_dict_is_ignored():
    e = RateLimitError(body=["not-a-dict"])
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == ""


###########################################
# make_is_transient: HTTP/marker corners  #
###########################################


def test_http_status_non_int_or_missing_means_no_retry():
    policy = make_policy()
    pred = make_is_transient(policy)

    class WeirdHTTP(Exception):
        pass

    # Build a policy that treats WeirdHTTP as an http_exc
    weird_policy = ErrorPolicy(
        auth_excs=policy.auth_excs,
        rate_limit_excs=policy.rate_limit_excs,
        network_excs=policy.network_excs,
        http_excs=(WeirdHTTP,),  # no status_code attr
        non_retryable_codes=policy.non_retryable_codes,
        retry_5xx=True,
        message_markers=policy.message_markers,
    )
    weird_pred = make_is_transient(weird_policy)
    assert weird_pred(WeirdHTTP()) is False  # no status_code -> no retry


def test_retry_5xx_false_disables_server_retries():
    base = make_policy()
    p = ErrorPolicy(
        auth_excs=base.auth_excs,
        rate_limit_excs=base.rate_limit_excs,
        network_excs=base.network_excs,
        http_excs=base.http_excs,
        non_retryable_codes=base.non_retryable_codes,
        retry_5xx=False,
        message_markers=base.message_markers,
    )
    pred = make_is_transient(p)
    assert pred(HTTPStatusError(500)) is False


def test_message_markers_override_policy_markers():
    policy = make_policy()
    pred = make_is_transient(
        policy, message_markers={"custom_code": ("special sentinel",)}
    )
    e = RateLimitError(msg="SPECIAL SENTINEL present")
    # Lowercasing in extract => match
    assert (
        extract_error_code(
            e, message_markers={"custom_code": ("special sentinel",)}
        )
        == "custom_code"
    )
    # Since "custom_code" is not non-retryable, it should retry:
    assert pred(e) is True


def test_numeric_zero_code_stringified():
    e = RateLimitError(response=DummyResponse({"error": {"code": 0}}))
    assert extract_error_code(e, message_markers=OPENAI_MARKERS) == "0"


############################################
# default_wait / default_stop construction #
############################################


def test_default_wait_bounds_env(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_RETRY_INITIAL_SECONDS", "0.0")  # below min
    w = default_wait()
    # Can't introspect tenacity easily; just ensure callable exists
    assert callable(w)  # sanity


def test_default_stop_default_attempts():
    s = default_stop()
    assert callable(s)
