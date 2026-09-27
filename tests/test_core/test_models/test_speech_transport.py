"""`SpeechTransport` against a real local HTTP server.

The provider tests stub the transport out, which leaves the transport itself
unexercised: whether it actually reaches the network, encodes a multipart body
the way a server can parse it, and turns status codes into the exception classes
the retry policy classifies on.
"""

import asyncio
import json as jsonlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from deepeval.models.retry_policy import SPEECH_ERROR_POLICY, make_is_transient
from deepeval.models.speech import (
    Multipart,
    SpeechAuthError,
    SpeechHTTPError,
    SpeechRateLimitError,
    SpeechTransport,
)

_received = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output clean
        pass

    def _respond(
        self, status: int, body: bytes, content_type="application/json"
    ):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/status/"):
            status = int(self.path.rsplit("/", 1)[1])
            self._respond(status, b'{"error":"boom"}')
            return
        if self.path == "/not-json":
            self._respond(200, b"<html>nope</html>", "text/html")
            return
        _received["query"] = self.path
        self._respond(200, jsonlib.dumps({"path": self.path}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        _received["headers"] = dict(self.headers)
        _received["body"] = body
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm")
            self.end_headers()
            for _ in range(4):
                self.wfile.write(b"\x01\x02" * 512)
            return
        self._respond(200, jsonlib.dumps({"ok": True}).encode())


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()


def _transport(base_url: str, **kwargs) -> SpeechTransport:
    return SpeechTransport(
        provider_label="Test",
        base_url=base_url,
        headers={"X-Api-Key": "secret", "X-Skipped": None},
        **kwargs,
    )


def test_none_valued_headers_are_dropped(server):
    assert "X-Skipped" not in _transport(server).headers


def test_absolute_urls_bypass_the_base_url(server):
    transport = _transport(server)

    assert transport.url("/v1/x") == f"{server}/v1/x"
    assert transport.url("https://other.example/y") == "https://other.example/y"


#
# Encoding
#


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"flag": True}, "flag=true"),
        ({"flag": False}, "flag=false"),
        ({"skip": None, "keep": "1"}, "keep=1"),
        ({"n": 24000}, "n=24000"),
    ],
)
def test_booleans_and_none_are_encoded_the_way_these_apis_expect(
    server, params, expected
):
    # `?flag=True` is not the `true` any of these providers look for.
    transport = _transport(server)

    payload = transport.request_json("GET", "/echo", params=params)

    assert expected in payload["path"]
    if "skip" in params:
        assert "skip" not in payload["path"]


def test_a_repeated_param_is_sent_once_per_value(server):
    # Deepgram spells multiple keyterms as the same key repeated, which a dict
    # cannot express.
    transport = _transport(server)

    payload = transport.request_json(
        "GET", "/echo", params={"keyterm": ["alpha", "beta"]}
    )

    assert "keyterm=alpha" in payload["path"]
    assert "keyterm=beta" in payload["path"]


def test_multipart_bodies_are_parseable_and_carry_scalar_fields(server):
    transport = _transport(server)

    transport.request_json(
        "POST",
        "/upload",
        multipart=Multipart(
            file_field="audio",
            filename="turn.wav",
            content=b"RIFFDATA",
            content_type="audio/wav",
            fields={"model": "m-1", "diarize": True, "skip": None},
        ),
    )

    body = _received["body"]
    assert b'name="audio"' in body
    assert b'filename="turn.wav"' in body
    assert b"RIFFDATA" in body
    assert b'name="model"' in body and b"m-1" in body
    # Booleans have to reach the server in the JSON spelling, not Python's.
    assert b"true" in body and b"True" not in body
    assert b'name="skip"' not in body


@pytest.mark.asyncio
async def test_the_async_client_encodes_multipart_the_same_way(server):
    transport = _transport(server)

    await transport.a_request_json(
        "POST",
        "/upload",
        multipart=Multipart(
            file_field="audio",
            filename="turn.wav",
            content=b"RIFFDATA",
            content_type="audio/wav",
            fields={"model": "m-1"},
        ),
    )

    body = _received["body"]
    assert b'filename="turn.wav"' in body
    assert b"RIFFDATA" in body
    assert b'name="model"' in body


def test_configured_headers_reach_the_server(server):
    _transport(server).request_json("POST", "/upload", content=b"x")

    assert _received["headers"]["X-Api-Key"] == "secret"


def test_per_request_headers_are_merged_over_the_defaults(server):
    _transport(server).request_json(
        "POST", "/upload", content=b"x", headers={"X-AAI-Model": "m"}
    )

    assert _received["headers"]["X-AAI-Model"] == "m"
    assert _received["headers"]["X-Api-Key"] == "secret"


#
# Streaming
#


@pytest.mark.asyncio
async def test_streaming_yields_the_body_in_pieces(server):
    transport = _transport(server)

    reads = [
        data
        async for data in transport.a_stream_bytes(
            "POST", "/stream", content=b"go", chunk_size=256
        )
    ]

    assert len(reads) > 1
    assert b"".join(reads) == b"\x01\x02" * 512 * 4


@pytest.mark.asyncio
async def test_many_sequential_requests_do_not_leak_sessions(server):
    # A cached session would be bound to a loop nobody closes; this checks the
    # per-request session actually works under repetition.
    transport = _transport(server)

    for _ in range(5):
        assert await transport.a_request_json("POST", "/upload", content=b"x")


def test_a_model_can_be_used_from_more_than_one_event_loop(server):
    # Each `asyncio.run` makes a fresh loop, and an aiohttp session bound to a
    # previous one would raise here.
    transport = _transport(server)

    for _ in range(3):
        payload = asyncio.run(
            transport.a_request_json("POST", "/upload", content=b"x")
        )
        assert payload == {"ok": True}


#
# Error translation
#


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, SpeechAuthError),
        (403, SpeechAuthError),
        (429, SpeechRateLimitError),
        (400, SpeechHTTPError),
        (500, SpeechHTTPError),
        (503, SpeechHTTPError),
    ],
)
def test_status_codes_become_the_classes_the_retry_policy_reads(
    server, status, expected
):
    transport = _transport(server)

    with pytest.raises(expected) as excinfo:
        transport.request_json("GET", f"/status/{status}")

    assert excinfo.value.status_code == status
    # The body explains what went wrong, so it belongs in the message.
    assert "boom" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,expected",
    [
        (401, SpeechAuthError),
        (429, SpeechRateLimitError),
        (500, SpeechHTTPError),
    ],
)
async def test_the_async_client_translates_errors_the_same_way(
    server, status, expected
):
    with pytest.raises(expected):
        await _transport(server).a_request_json("GET", f"/status/{status}")


@pytest.mark.asyncio
async def test_a_streaming_error_is_raised_with_its_body(server):
    with pytest.raises(SpeechHTTPError, match="boom"):
        async for _ in _transport(server).a_stream_bytes("GET", "/status/500"):
            pass


def test_an_auth_failure_says_to_check_the_key(server):
    with pytest.raises(SpeechAuthError, match="API key"):
        _transport(server).request_json("GET", "/status/401")


def test_a_non_json_response_is_reported_as_such(server):
    with pytest.raises(SpeechHTTPError, match="not valid JSON"):
        _transport(server).request_json("GET", "/not-json")


@pytest.mark.parametrize(
    "status,retryable",
    [
        # Bad credentials will be just as bad next time.
        (401, False),
        (403, False),
        # Rate limits and provider faults are worth another attempt.
        (429, True),
        (500, True),
        (503, True),
        # A malformed request is permanent.
        (400, False),
        (404, False),
    ],
)
def test_the_retry_policy_classifies_speech_failures(server, status, retryable):
    is_transient = make_is_transient(SPEECH_ERROR_POLICY)
    transport = _transport(server)

    try:
        transport.request_json("GET", f"/status/{status}")
        raise AssertionError("expected a failure")
    except SpeechHTTPError as exc:
        assert is_transient(exc) is retryable


def test_a_connection_failure_is_treated_as_transient():
    is_transient = make_is_transient(SPEECH_ERROR_POLICY)
    # Port 1 is not listening, so this is a genuine connection error rather
    # than a simulated one.
    transport = _transport("http://127.0.0.1:1", timeout=2.0)

    with pytest.raises(Exception) as excinfo:
        transport.request_json("GET", "/echo")

    assert is_transient(excinfo.value) is True
