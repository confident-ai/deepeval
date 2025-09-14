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

    if BEDROCK_ERROR_POLICY is None:
        return  # botocore not installed in this env, then skip
    pred = make_is_transient(BEDROCK_ERROR_POLICY)

    # throttling by message: should retry
    class ClientError(Exception): ...

    assert (
        pred(
            ClientError(
                "An error occurred (ThrottlingException) when calling Converse: Rate exceeded"
            )
        )
        is True
    )
    # accessDenied: should NOT retry
    assert (
        pred(
            ClientError(
                "An error occurred (AccessDeniedException) when calling Converse: Access denied"
            )
        )
        is False
    )


def test_bedrock_sdk_toggle(monkeypatch):
    from deepeval.config.settings import get_settings

    # fake session instance so we can inspect its state
    sess = DummySession()

    # fake aiobotocore availability and configure module to use our fakes
    monkeypatch.setattr(mod, "aiobotocore_available", True)
    monkeypatch.setattr(mod, "get_session", lambda: sess, raising=False)
    monkeypatch.setattr(mod, "Config", DummyConfig, raising=False)

    # use settings.edit() so the runtime toggle takes effect
    s = get_settings()

    # SDK control ON means adaptive mode, max_attempts=5
    with s.edit(persist=False):
        s.DEEPEVAL_SDK_RETRY_PROVIDERS = ["bedrock"]

    m = mod.AmazonBedrockModel(model_id="id", region_name="us-east-1")
    # triggers client build
    m.generate("ping")
    assert m._sdk_retry_mode is True
    assert isinstance(sess.last_config, DummyConfig)
    assert sess.last_config.retries.get("max_attempts") == 5
    assert sess.last_config.retries.get("mode") == "adaptive"

    # flip to Tenacity control, expect max_attempts=1
    with s.edit(persist=False):
        s.DEEPEVAL_SDK_RETRY_PROVIDERS = []

    # Next call should rebuild the client with new retry config
    m.generate("ping2")
    assert m._sdk_retry_mode is False
    assert isinstance(sess.last_config, DummyConfig)
    assert sess.last_config.retries.get("max_attempts") == 1
    # no 'mode' key when we drive Tenacity
    assert "mode" not in sess.last_config.retries
