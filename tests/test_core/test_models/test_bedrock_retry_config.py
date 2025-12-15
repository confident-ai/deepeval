from types import SimpleNamespace
from unittest.mock import patch

import deepeval.models.llms.amazon_bedrock_model as mod


class DummyConfig:
    def __init__(self, *, retries=None, **kw):
        self.retries = retries or {}
        self.kw = kw


class DummyClient:
    # minimal response shape the model expects
    async def converse(self, **kwargs):
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 3, "outputTokens": 7},
        }


class DummyCM:
    def __init__(self, session, service_name, **kw):
        self.session = session
        self.kw = kw

    async def __aenter__(self):
        # record the Config used for later assertions
        self.session.last_config = self.kw.get("config")
        return DummyClient()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def __init__(self):
        self.last_config = None
        self.created = 0

    def create_client(self, service_name, **kw):
        self.created += 1
        return DummyCM(self, service_name, **kw)


def test_bedrock_retry_predicate_present():
    from deepeval.models.retry_policy import (
        BEDROCK_ERROR_POLICY,
        make_is_transient,
    )

    # If botocore isn't installed, the Bedrock policy is None and we skip.
    if BEDROCK_ERROR_POLICY is None:
        return

    # Only import botocore when we know it's available.
    from botocore.exceptions import ClientError

    pred = make_is_transient(BEDROCK_ERROR_POLICY)

    # ThrottlingException should be treated as retriable.
    throttling_exc = ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded",
            }
        },
        operation_name="Converse",
    )
    assert pred(throttling_exc) is True

    # AccessDeniedException: should not be retried.
    access_denied_exc = ClientError(
        error_response={
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Access denied",
            }
        },
        operation_name="Converse",
    )
    assert pred(access_denied_exc) is False


@patch("deepeval.models.llms.amazon_bedrock_model.require_dependency")
def test_bedrock_sdk_toggle(mock_require_dep, settings):

    # fake session instance so we can inspect its state
    sess = DummySession()

    # Fake modules returned by require_dependency inside AmazonBedrockModel
    fake_aiobotocore_session_module = SimpleNamespace(
        get_session=lambda: sess,
    )

    class DummyBotocoreModule:
        class config:
            Config = DummyConfig

    def fake_require_dependency(name, provider_label=None, install_hint=None):
        if name == "aiobotocore.session":
            return fake_aiobotocore_session_module
        if name == "botocore":
            return DummyBotocoreModule
        raise AssertionError(f"Unexpected dependency requested: {name}")

    # Patch the require_dependency used by amazon_bedrock_model
    mock_require_dep.side_effect = fake_require_dependency

    # SDK control ON means adaptive mode, max_attempts=5
    with settings.edit(persist=False):
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = ["bedrock"]
        settings.AWS_BEDROCK_COST_PER_INPUT_TOKEN = 1e-6
        settings.AWS_BEDROCK_COST_PER_OUTPUT_TOKEN = 1e-6

    m = mod.AmazonBedrockModel(model="id", region_name="us-east-1")
    # triggers client build
    m.generate("ping")
    assert m._sdk_retry_mode is True
    assert isinstance(sess.last_config, DummyConfig)
    assert sess.last_config.retries.get("max_attempts") == 5
    assert sess.last_config.retries.get("mode") == "adaptive"

    # flip to Tenacity control, expect max_attempts=1
    with settings.edit(persist=False):
        settings.DEEPEVAL_SDK_RETRY_PROVIDERS = []

    # Next call should rebuild the client with new retry config
    m.generate("ping2")
    assert m._sdk_retry_mode is False
    assert isinstance(sess.last_config, DummyConfig)
    assert sess.last_config.retries.get("max_attempts") == 1
    # no 'mode' key when we drive Tenacity
    assert "mode" not in sess.last_config.retries
