import deepeval.models.llms.azure_model as azure_model
from deepeval.models.retry_policy import (
    ErrorPolicy,
    AZURE_OPENAI_ERROR_POLICY,
    OPENAI_MESSAGE_MARKERS,
    make_is_transient,
    get_retry_policy_for,
)


assert AZURE_OPENAI_ERROR_POLICY is not None, "OpenAI is a required dependency"


def test_azure_retry_predicate_present():
    # 1) We have a retry policy wired for 'azure' (unless the user explicitly opts into SDK retries)
    assert get_retry_policy_for("azure") is not None

    # 2) All Azure model call sites are decorated with Tenacity (@retry_azure),
    #    which Tenacity exposes as a 'retry' attribute on the wrapped function.
    decorated_methods = (
        "generate",
        "a_generate",
        "generate_raw_response",
        "a_generate_raw_response",
    )
    for name in decorated_methods:
        fn = getattr(azure_model.AzureOpenAIModel, name)
        assert hasattr(
            fn, "retry"
        ), f"{name} should be decorated with Tenacity retry"


def test_azure_sdk_retries_disabled(monkeypatch):
    # build model with conflicting kwargs, then our override should win.
    m = azure_model.AzureOpenAIModel(
        deployment_name="dummy",
        model_name="gpt-4o-mini",
        azure_openai_api_key="x",
        openai_api_version="2024-02-01",
        azure_endpoint="https://example",
        max_retries=5,
    )
    client = m.load_model(async_mode=False)
    assert client.max_retries == 0


def test_azure_hard_quota_marker_is_non_retryable():
    class RateLimitError(Exception):
        def __init__(self, msg="", response=None, body=None):
            super().__init__(msg)
            self.response = response
            self.body = body

    policy = ErrorPolicy(
        auth_excs=(),
        rate_limit_excs=(RateLimitError,),
        network_excs=(),
        http_excs=(),
        non_retryable_codes=frozenset({"insufficient_quota"}),
        message_markers=OPENAI_MESSAGE_MARKERS,
    )
    pred = make_is_transient(policy)

    e = RateLimitError(body={"error": {"code": "insufficient_quota"}})
    assert pred(e) is False


def test_length_finish_reason_is_non_retryable():
    class LengthFinishReasonError(Exception): ...

    policy = ErrorPolicy(
        auth_excs=(),
        rate_limit_excs=(),
        network_excs=(),
        http_excs=(),
        non_retryable_codes=frozenset({"insufficient_quota"}),
        message_markers={},
    )
    pred = make_is_transient(policy)
    assert pred(LengthFinishReasonError()) is False


def test_azure_sdk_retries_opt_in_respects_user_max_retries():
    from deepeval.config.settings import get_settings

    s = get_settings()

    # turn ON SDK managed retries for Azure
    with s.edit(persist=False):
        s.DEEPEVAL_SDK_RETRY_PROVIDERS = ["azure"]

    try:
        m = azure_model.AzureOpenAIModel(
            deployment_name="dummy",
            model_name="gpt-4o-mini",
            azure_openai_api_key="x",
            openai_api_version="2024-02-01",
            azure_endpoint="https://example",
            max_retries=5,  # should be honored when SDK retries are enabled
        )
        client = m.load_model(async_mode=False)
        assert client.max_retries == 5
    finally:
        # clean up to avoid bleeding state into other tests
        with s.edit(persist=False):
            s.DEEPEVAL_SDK_RETRY_PROVIDERS = []
