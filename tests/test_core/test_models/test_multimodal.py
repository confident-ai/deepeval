import base64
from types import SimpleNamespace

import pytest
import requests

from deepeval.config.settings import get_settings
from deepeval.errors import DeepEvalError
from deepeval.models.llms import (
    AnthropicModel,
    DeepSeekModel,
    GeminiModel,
    GrokModel,
    KimiModel,
    OllamaModel,
    OpenAIModel,
    OpenRouterModel,
)
from deepeval.models.llms.multimodal import (
    audio_base64,
    audio_format,
    decode_image,
    fetch_remote_image,
    multimodal_parts,
    reject_audio,
    to_openai_content,
)
from deepeval.test_case import Audio, MLLMImage

IMAGE_BYTES = b"\x89PNG-bytes"
IMAGE_B64 = base64.b64encode(IMAGE_BYTES).decode()
WAV_BYTES = b"RIFF-wav-bytes"
WAV_B64 = base64.b64encode(WAV_BYTES).decode()


def wav_clip():
    return Audio.from_bytes(WAV_BYTES, "audio/x-wav")


class FakeResponse:
    def __init__(self, headers):
        self.content = IMAGE_BYTES
        self.headers = headers

    def raise_for_status(self):
        pass


@pytest.fixture
def fetches(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({"content-type": "image/webp"})

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_multimodal_parts_is_none_for_text_only_prompt():
    assert multimodal_parts("just text") is None


def test_multimodal_parts_splits_text_and_images_in_order():
    first = MLLMImage(dataBase64=IMAGE_B64, mimeType="image/png")
    second = MLLMImage(url="https://example.com/cat.png")

    parts = multimodal_parts(f"A {first} B {second}")

    assert parts == ["A ", first, " B ", second]


def test_to_openai_content_wraps_text_only_prompt():
    assert to_openai_content("hello") == [{"type": "text", "text": "hello"}]


def test_to_openai_content_text_only_as_string():
    assert to_openai_content("hello", text_only_as_string=True) == "hello"


def test_to_openai_content_formats_remote_and_inline_images(fetches):
    remote = MLLMImage(url="https://example.com/cat.png")
    inline = MLLMImage(dataBase64=IMAGE_B64, mimeType="image/png")

    content = to_openai_content(f"Compare {remote} with {inline}")

    assert content == [
        {"type": "text", "text": "Compare "},
        {"type": "image_url", "image_url": {"url": remote.url}},
        {"type": "text", "text": " with "},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{IMAGE_B64}"},
        },
    ]
    assert fetches == []


def test_to_openai_content_inlines_remote_images_when_asked(fetches):
    remote = MLLMImage(url="https://example.com/cat.png")

    content = to_openai_content(f"{remote}", inline_remote_images=True)

    assert content == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/webp;base64,{IMAGE_B64}"},
        }
    ]
    assert [url for url, _ in fetches] == [remote.url]


def test_to_openai_content_reports_failed_remote_fetch(monkeypatch):
    def failing_get(url, **kwargs):
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(requests, "get", failing_get)
    remote = MLLMImage(url="https://example.com/cat.png")

    with pytest.raises(ValueError, match="Failed to fetch remote image"):
        to_openai_content(f"{remote}", inline_remote_images=True)


def test_fetch_remote_image_uses_media_timeouts(fetches):
    settings = get_settings()
    remote = MLLMImage(url="https://example.com/cat.png")

    data, mime_type = fetch_remote_image(remote)

    assert (data, mime_type) == (IMAGE_BYTES, "image/webp")
    assert fetches[0][1]["timeout"] == (
        settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
        settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
    )


def test_fetch_remote_image_falls_back_to_jpeg_without_content_type(
    monkeypatch,
):
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResponse({}))
    remote = MLLMImage(url="https://example.com/image")

    assert fetch_remote_image(remote) == (IMAGE_BYTES, "image/jpeg")


def test_decode_image_returns_raw_bytes():
    image = MLLMImage(dataBase64=IMAGE_B64, mimeType="image/png")

    assert decode_image(image) == IMAGE_BYTES


@pytest.mark.parametrize("error_cls", [ValueError, DeepEvalError])
def test_decode_image_raises_given_error_on_invalid_base64(error_cls):
    image = MLLMImage(dataBase64="not base64!!", mimeType="image/png")

    with pytest.raises(error_cls, match="Invalid base64 data"):
        decode_image(image, error_cls=error_cls)


def test_multimodal_parts_puts_audio_after_the_text():
    clip = wav_clip()

    assert multimodal_parts("Rate the reply.", [clip]) == [
        "Rate the reply.",
        clip,
    ]


def test_multimodal_parts_puts_audio_after_images():
    image = MLLMImage(dataBase64=IMAGE_B64, mimeType="image/png")
    clip = wav_clip()

    assert multimodal_parts(f"A {image}", [clip]) == ["A ", image, clip]


def test_multimodal_parts_treats_empty_audio_as_text_only():
    assert multimodal_parts("just text", []) is None


def test_to_openai_content_sends_audio_as_base64_input_audio():
    content = to_openai_content("Rate the reply.", [wav_clip()])

    assert content == [
        {"type": "text", "text": "Rate the reply."},
        {
            "type": "input_audio",
            "input_audio": {"data": WAV_B64, "format": "wav"},
        },
    ]


@pytest.mark.parametrize(
    "mime_type, expected",
    [
        ("audio/wav", "wav"),
        ("audio/x-wav", "wav"),
        ("audio/wave", "wav"),
        ("audio/vnd.wave", "wav"),
        ("audio/mpeg", "mp3"),
        ("audio/mp3", "mp3"),
        ("audio/flac", "flac"),
    ],
)
def test_audio_format_normalizes_mime_types(mime_type, expected):
    assert audio_format(Audio.from_bytes(WAV_BYTES, mime_type)) == expected


def test_audio_base64_downloads_a_remote_clip(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **kwargs: FakeResponse({"content-type": "audio/wav"}),
    )
    clip = Audio(url="https://example.com/call.wav")

    assert audio_base64(clip) == IMAGE_B64


@pytest.mark.parametrize("audio", [None, []])
def test_reject_audio_allows_calls_without_audio(audio):
    reject_audio(audio, "SomeModel")


def test_reject_audio_raises_when_audio_is_given():
    with pytest.raises(DeepEvalError, match="SomeModel doesn't accept audio"):
        reject_audio([wav_clip()], "SomeModel")


def test_gemini_sends_audio_as_an_inline_part():
    model = GeminiModel.__new__(GeminiModel)
    model._module = SimpleNamespace(
        types=SimpleNamespace(
            Part=SimpleNamespace(
                from_bytes=lambda data, mime_type: (data, mime_type)
            )
        )
    )

    contents = model._prompt_contents("Rate the reply.", [wav_clip()])

    assert contents == ["Rate the reply.", (WAV_BYTES, "audio/wav")]


@pytest.mark.parametrize(
    "send_audio",
    [
        lambda audio: AnthropicModel.__new__(AnthropicModel)._prompt_content(
            "Rate", audio
        ),
        lambda audio: OllamaModel.__new__(OllamaModel)._prompt_messages(
            "Rate", audio
        ),
        lambda audio: DeepSeekModel.__new__(DeepSeekModel).generate(
            "Rate", audio=audio
        ),
        lambda audio: GrokModel.__new__(GrokModel)._prompt_content(
            "Rate", audio
        ),
        lambda audio: KimiModel.__new__(KimiModel)._prompt_content(
            "Rate", audio
        ),
    ],
    ids=["anthropic", "ollama", "deepseek", "grok", "kimi"],
)
def test_providers_without_audio_input_reject_audio(send_audio):
    with pytest.raises(DeepEvalError, match="doesn't accept audio input"):
        send_audio([wav_clip()])


@pytest.mark.parametrize(
    "make_model, supports, requires",
    [
        (lambda: OpenAIModel(model="gpt-audio", api_key="k"), True, True),
        (lambda: OpenAIModel(model="gpt-audio-mini", api_key="k"), True, True),
        (lambda: OpenAIModel(model="gpt-4.1", api_key="k"), False, None),
        (
            lambda: GeminiModel(model="gemini-2.5-flash", api_key="k"),
            True,
            None,
        ),
        (lambda: GeminiModel(model="gemini-pro", api_key="k"), False, None),
        (lambda: AnthropicModel.__new__(AnthropicModel), False, None),
        (lambda: OllamaModel.__new__(OllamaModel), False, None),
        (lambda: OpenRouterModel(model="x", api_key="k"), None, None),
    ],
    ids=[
        "gpt-audio",
        "gpt-audio-mini",
        "gpt-4.1",
        "gemini-2.5-flash",
        "gemini-pro",
        "anthropic",
        "ollama",
        "openrouter",
    ],
)
def test_models_report_audio_support(make_model, supports, requires):
    model = make_model()

    assert model.supports_audio() is supports
    assert model.requires_audio() is requires


def test_audio_only_openai_model_rejects_text_only_prompts():
    model = OpenAIModel(model="gpt-audio", api_key="k")

    with pytest.raises(DeepEvalError, match="only accepts requests that"):
        model._prompt_content("Rate the reply.")
    assert model._prompt_content("Rate the reply.", [wav_clip()])[1] == {
        "type": "input_audio",
        "input_audio": {"data": WAV_B64, "format": "wav"},
    }


def test_openai_generate_bills_audio_tokens_at_the_audio_rate():
    model = OpenAIModel(model="gpt-audio", api_key="k")
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(audio_tokens=600),
        ),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: completion)
        )
    )
    model.load_model = lambda async_mode=False: client

    _, cost = model.generate("Rate the reply.", audio=[wav_clip()])

    assert cost == pytest.approx(400 * 2.5e-6 + 600 * 32e-6 + 100 * 10e-6)


def test_gemini_bills_audio_modality_tokens_at_the_audio_rate():
    model = GeminiModel(model="gemini-2.5-flash", api_key="k")
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=100,
        thoughts_token_count=None,
        prompt_tokens_details=[
            SimpleNamespace(modality="TEXT", token_count=400),
            SimpleNamespace(modality="AUDIO", token_count=600),
        ],
    )

    cost = model._token_cost(SimpleNamespace(usage_metadata=usage))

    assert cost == pytest.approx(400 * 0.30e-6 + 600 * 1.00e-6 + 100 * 2.5e-6)
