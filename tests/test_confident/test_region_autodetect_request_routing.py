from pydantic import SecretStr


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_request_succeeds_by_auto_routing_eu_key_when_region_unset(monkeypatch):
    """
    Red today:
      - Region unset => get_base_api_url() defaults to US
      - EU key used against US endpoint => 401 Invalid API key => raises ConfidentApiError

    Green after fix:
      - Region unset + api key prefix confident_eu_ => route EU
      - Request succeeds (200) against EU endpoint
    """
    from deepeval.confident import api as confident_api

    # Settings: no explicit base url override; EU api key present
    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_API_KEY = SecretStr("confident_eu_6M_dummy")
        API_KEY = None
        DEEPEVAL_DEFAULT_SAVE = None

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())

    # Region is not set by user
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: None,
    )

    # Fake HTTP behavior:
    # - If it goes to US base URL => return 401 Invalid API key
    # - If it goes to EU base URL => return 200 success
    def fake_http_request(
        method: str, url: str, headers=None, json=None, params=None
    ):
        if url.startswith(confident_api.API_BASE_URL_EU):
            return _FakeResponse(
                200,
                {"success": True, "data": {"ok": True}, "deprecated": False},
            )
        return _FakeResponse(
            401,
            {"success": False, "error": "Invalid API key", "deprecated": False},
        )

    monkeypatch.setattr(
        confident_api.Api, "_http_request", staticmethod(fake_http_request)
    )

    api = (
        confident_api.Api()
    )  # uses get_confident_api_key() + get_base_api_url()

    data, link = api.send_request(
        method=confident_api.HttpMethods.POST,
        endpoint=confident_api.Endpoints.TEST_RUN_ENDPOINT,
        body={"dummy": True},
    )

    assert data == {"ok": True}
    assert link is None


def test_request_succeeds_by_auto_routing_au_key_when_region_unset(monkeypatch):
    """
    Red today:
      - Region unset => get_base_api_url() defaults to US
      - AU key used against US endpoint => 401 Invalid API key => raises ConfidentApiError

    Green after fix:
      - Region unset + api key prefix confident_au_ => route AU
      - Request succeeds (200) against AU endpoint
    """
    from deepeval.confident import api as confident_api

    # Settings: no explicit base url override; AU api key present
    class DummySettings:
        CONFIDENT_BASE_URL = None
        CONFIDENT_API_KEY = SecretStr("confident_au_7M_dummy")
        API_KEY = None
        DEEPEVAL_DEFAULT_SAVE = None

    monkeypatch.setattr(confident_api, "get_settings", lambda: DummySettings())

    # Region is not set by user
    monkeypatch.setattr(
        confident_api.KEY_FILE_HANDLER,
        "fetch_data",
        lambda *args, **kwargs: None,
    )

    # Fake HTTP behavior:
    # - If it goes to US base URL => return 401 Invalid API key
    # - If it goes to AU base URL => return 200 success
    def fake_http_request(
        method: str, url: str, headers=None, json=None, params=None
    ):
        if url.startswith(confident_api.API_BASE_URL_AU):
            return _FakeResponse(
                200,
                {"success": True, "data": {"ok": True}, "deprecated": False},
            )
        return _FakeResponse(
            401,
            {"success": False, "error": "Invalid API key", "deprecated": False},
        )

    monkeypatch.setattr(
        confident_api.Api, "_http_request", staticmethod(fake_http_request)
    )

    api = (
        confident_api.Api()
    )  # uses get_confident_api_key() + get_base_api_url()

    data, link = api.send_request(
        method=confident_api.HttpMethods.POST,
        endpoint=confident_api.Endpoints.TEST_RUN_ENDPOINT,
        body={"dummy": True},
    )

    assert data == {"ok": True}
    assert link is None
