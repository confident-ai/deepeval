"""Non-2xx Confident AI responses must surface as errors.

A gateway, proxy, or framework-level rejection (401/404/422/5xx) commonly
answers with a JSON body that is not the Confident ``{"success": ...}``
envelope, for example ``{"detail": "Not authenticated"}``. ``send_request``
and ``a_send_request`` must raise in that case instead of handing the error
body back to the caller as if it were successful ``data``.
"""

import pytest
from pydantic import SecretStr

from deepeval.confident import api as confident_api
from deepeval.confident.types import ConfidentApiError


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeAsyncResponse:
    def __init__(self, status: int, payload):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeClientSession:
    """Stands in for ``aiohttp.ClientSession`` and returns one canned reply."""

    response = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, *args, **kwargs):
        return type(self).response


@pytest.fixture
def api(monkeypatch):
    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_REGION = None
        CONFIDENT_API_KEY = SecretStr("confident_us_dummy")
        DEEPEVAL_DEFAULT_SAVE = None
        CONFIDENT_DISABLE_SSL = None

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: None,
    )
    return confident_api.Api()


def _install_sync_response(monkeypatch, status_code: int, payload):
    def fake_http_request(method, url, headers=None, json=None, params=None):
        return _FakeResponse(status_code, payload)

    monkeypatch.setattr(
        confident_api.Api, "_http_request", staticmethod(fake_http_request)
    )


def _install_async_response(monkeypatch, status: int, payload):
    _FakeClientSession.response = _FakeAsyncResponse(status, payload)
    monkeypatch.setattr(
        confident_api.aiohttp, "ClientSession", _FakeClientSession
    )


@pytest.mark.parametrize(
    "status_code, payload",
    [
        (401, {"detail": "Not authenticated"}),
        (404, {"message": "Dataset not found"}),
        (500, {"error": "Internal Server Error"}),
        (502, {"code": 502}),
    ],
)
def test_send_request_raises_on_non_envelope_error_body(
    monkeypatch, api, status_code, payload
):
    _install_sync_response(monkeypatch, status_code, payload)

    with pytest.raises(ConfidentApiError, match=str(status_code)):
        api.send_request(
            method=confident_api.HttpMethods.GET,
            endpoint=confident_api.Endpoints.DATASET_ALIAS_ENDPOINT,
            url_params={"alias": "missing"},
        )


def test_send_request_error_message_carries_body_detail(monkeypatch, api):
    _install_sync_response(monkeypatch, 401, {"detail": "Not authenticated"})

    with pytest.raises(ConfidentApiError, match="Not authenticated"):
        api.send_request(
            method=confident_api.HttpMethods.GET,
            endpoint=confident_api.Endpoints.DATASET_ALIAS_ENDPOINT,
            url_params={"alias": "missing"},
        )


def test_send_request_still_raises_confident_envelope_error(monkeypatch, api):
    """The existing ``{"success": false}`` path keeps its error message."""
    _install_sync_response(
        monkeypatch, 401, {"success": False, "error": "Invalid API key"}
    )

    with pytest.raises(ConfidentApiError, match="Invalid API key"):
        api.send_request(
            method=confident_api.HttpMethods.GET,
            endpoint=confident_api.Endpoints.DATASET_ALIAS_ENDPOINT,
            url_params={"alias": "missing"},
        )


def test_send_request_200_non_envelope_body_is_still_returned(monkeypatch, api):
    """Legacy endpoints answer 200 with a bare payload; keep passing it back."""
    _install_sync_response(monkeypatch, 200, {"id": "run_123"})

    data, link = api.send_request(
        method=confident_api.HttpMethods.POST,
        endpoint=confident_api.Endpoints.TEST_RUN_ENDPOINT,
        body={},
    )

    assert data == {"id": "run_123"}
    assert link is None


@pytest.mark.parametrize(
    "status, payload",
    [
        (401, {"detail": "Not authenticated"}),
        (500, {"error": "Internal Server Error"}),
    ],
)
async def test_a_send_request_raises_on_non_envelope_error_body(
    monkeypatch, api, status, payload
):
    _install_async_response(monkeypatch, status, payload)

    with pytest.raises(ConfidentApiError, match=str(status)):
        await api.a_send_request(
            method=confident_api.HttpMethods.GET,
            endpoint=confident_api.Endpoints.DATASET_ALIAS_ENDPOINT,
            url_params={"alias": "missing"},
        )


async def test_a_send_request_still_raises_confident_envelope_error(
    monkeypatch, api
):
    _install_async_response(
        monkeypatch, 401, {"success": False, "error": "Invalid API key"}
    )

    with pytest.raises(ConfidentApiError, match="Invalid API key"):
        await api.a_send_request(
            method=confident_api.HttpMethods.GET,
            endpoint=confident_api.Endpoints.DATASET_ALIAS_ENDPOINT,
            url_params={"alias": "missing"},
        )
