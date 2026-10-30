"""NaceModel (Drex) against a recorded-shape fake of the Drex API.

Every request goes through the real ``nace-sdk`` client into an
``httpx.MockTransport`` that answers like ``POST /v1/systemone`` does, so the
wire format (questions, images, headers, model ids) is what is asserted. No
network and no key are needed.
"""

import base64
import json
from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("nace_sdk")

from deepeval._version import __version__  # noqa: E402
from deepeval.errors import DeepEvalError  # noqa: E402
from deepeval.models import NaceModel, TypeSafeModel  # noqa: E402
from deepeval.models.system_one.constants import (  # noqa: E402
    DREX_MAX_BODY_BYTES,
)
from deepeval.models.system_one.limits import (  # noqa: E402
    SystemOneContextLimitError,
    SystemOneQuestionLimitError,
)
from deepeval.models.system_one.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)
from deepeval.test_case import MLLMImage  # noqa: E402
from tests.test_metrics.system_one_fakes import (  # noqa: E402
    BLUE,
    RED,
    FakeDrex,
    drex_error as error,
    image,
)


@pytest.fixture
def drex():
    return FakeDrex()


def nace(drex: FakeDrex, **kwargs) -> NaceModel:
    kwargs.setdefault("api_key", "nace_sk_test")
    return NaceModel(transport=httpx.MockTransport(drex), **kwargs)


NOUL = {"q": NoulQuestion(instructions="Is the customer asking for a refund?")}
CHOICE = {
    "q": ChoiceQuestion(
        instructions="What does the customer want?",
        options={"refund": "Money back", "replace": None},
    )
}
SCORE = {
    "q": ScoreQuestion(
        instructions="How urgent is this?", levels=["Low", "Medium", "High"]
    )
}


@pytest.fixture(autouse=True)
def _no_nace_env(monkeypatch):
    for key in (
        "NACE_API_KEY",
        "OPENROUTER_API_KEY",
        "NACE_MODEL_NAME",
        "NACE_BASE_URL",
        "NACE_COST_PER_INPUT_TOKEN",
        "DEEPEVAL_DECISION_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


###############################################
# Questions and answers
###############################################


@pytest.mark.parametrize("model", ["drex-v2.0", "drex-max"])
@pytest.mark.parametrize(
    "questions, kind, wire_type",
    [
        (NOUL, "nouls", "noul"),
        (CHOICE, "choices", "choice"),
        (SCORE, "scores", "score"),
    ],
    ids=["noul", "choice", "score"],
)
def test_single_question_on_each_model(drex, model, questions, kind, wire_type):
    answers, cost = nace(drex, model=model).decide("state", questions)

    assert list(getattr(answers, kind)) == ["q"]
    [body] = drex.bodies
    assert body["model"] == model
    assert body["questions"]["q"]["type"] == wire_type
    assert cost.input_tokens == 1000


def test_choice_and_score_keep_full_probabilities(drex):
    answers, _ = nace(drex).decide("s", {**CHOICE, "s": SCORE["q"]})

    assert answers.choices["q"].probabilities == {
        "refund": 0.8,
        "replace": pytest.approx(0.2),
    }
    assert answers.scores["s"].probabilities == {0: 0.0, 1: 0.0, 2: 1.0}


def test_mixed_batch_is_one_request(drex):
    questions = {"a": NOUL["q"], "b": CHOICE["q"], "c": SCORE["q"]}
    answers, _ = nace(drex).decide({"ticket": "lamp cracked"}, questions)

    assert len(drex.requests) == 1
    assert set(answers.nouls) == {"a"}
    assert set(answers.choices) == {"b"}
    assert set(answers.scores) == {"c"}
    assert drex.bodies[0]["state"] == {"ticket": "lamp cracked"}


def test_noul_with_criteria_only_and_choice_null_description(drex):
    questions = {
        "upset": NoulQuestion(
            instructions=None, true="The customer is upset", false=None
        ),
        "intent": CHOICE["q"],
    }
    nace(drex).decide("s", questions)

    wire = drex.bodies[0]["questions"]
    # Drex rejects `instructions: null`, so it is left out, never sent null.
    assert wire["upset"] == {
        "type": "noul",
        "criteria": {"true": "The customer is upset", "false": None},
    }
    assert wire["intent"]["criteria"] == {
        "refund": "Money back",
        "replace": None,
    }


def test_non_text_instructions_are_sent_as_text(drex):
    nace(drex).decide(
        "s", {"q": NoulQuestion(instructions={"rule": "refunds only"})}
    )

    assert drex.bodies[0]["questions"]["q"]["instructions"] == (
        '{"rule": "refunds only"}'
    )


def test_default_model_is_drex_max_1_1(drex):
    model = nace(drex)

    assert model.name == "drex-max-v1.1"
    assert model.get_model_name() == "drex-max-v1.1 (Nace)"
    assert model.supports_images()
    _, cost = model.decide("s", NOUL)
    assert drex.bodies[0]["model"] == "drex-max-v1.1"
    assert cost.value == pytest.approx(1000 * 0.03 / 1e6)


def test_openrouter_model_id_names_the_same_model(drex):
    model = nace(drex, model="nace-ai/drex-max-v1.1")

    assert model.name == "drex-max-v1.1"
    assert model.supports_images()
    model.decide("s", NOUL)
    assert drex.bodies[0]["model"] == "drex-max-v1.1"


def test_cost_uses_the_price_of_the_model(drex):
    _, cost = nace(drex, model="drex-max").decide("s", NOUL)
    _, cost_v2 = nace(drex, model="drex-v2.0").decide("s", NOUL)

    assert cost.value == pytest.approx(1000 * 0.03 / 1e6)
    assert cost_v2.value == pytest.approx(1000 * 0.06 / 1e6)


def test_cost_override(drex):
    _, cost = nace(drex, cost_per_input_token=1e-6).decide("s", NOUL)

    assert cost.value == pytest.approx(1000 * 1e-6)


###############################################
# Images (Drex Max)
###############################################


@pytest.mark.parametrize("order", [(RED, BLUE), (BLUE, RED)], ids=["rb", "br"])
def test_two_images_keep_their_order(drex, order):
    question = ChoiceQuestion(
        instructions="Which image is red?",
        options={"Image 1": None, "Image 2": None},
    )
    nace(drex).decide(
        "Look at [Image 1] and [Image 2].",
        {"which": question},
        images=[image(data) for data in order],
    )

    [body] = drex.bodies
    assert body["model"] == "drex-max-v1.1"
    sent = [base64.b64decode(url.split(",", 1)[1]) for url in body["images"]]
    assert sent == list(order)


def test_images_on_text_model_refused_client_side(drex):
    model = nace(drex, model="drex-v2.0")

    assert not model.supports_images()
    with pytest.raises(DeepEvalError, match="drex-v2.0 reads text only"):
        model.decide("[Image 1]", NOUL, images=[image(RED)])
    assert drex.requests == []


def test_non_image_bytes_are_refused(drex):
    gif = MLLMImage(
        dataBase64=base64.b64encode(b"GIF89a....").decode(),
        mimeType="image/gif",
    )
    with pytest.raises(DeepEvalError, match="PNG, JPEG or WebP"):
        nace(drex, model="drex-max").decide("[Image 1]", NOUL, images=[gif])
    assert drex.requests == []


def test_more_than_eight_images_refused(drex):
    with pytest.raises(DeepEvalError, match="at most 8 images"):
        nace(drex, model="drex-max").decide("s", NOUL, images=[image(RED)] * 9)


def test_images_too_large_to_inline_are_uploaded_then_deleted(drex):
    model = nace(drex, model="drex-max")
    client = model.load_model()
    uploaded, deleted = [], []

    def upload(data, file_name):
        uploaded.append((len(data), file_name))
        return SimpleNamespace(
            workspace_id="ws_1", file_id=f"f_{len(uploaded)}"
        )

    client.documents.upload = upload
    client.documents.delete_file = lambda ws, fid: deleted.append((ws, fid))
    # 3 MiB of bytes is 4 MiB as base64: over the body limit inline.
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (DREX_MAX_BODY_BYTES * 3 // 4)

    model.decide("[Image 1] [Image 2]", NOUL, images=[image(big), image(RED)])

    # The big image goes up as a file; the small one still fits inline.
    assert len(uploaded) == 1 and uploaded[0][1].endswith(".png")
    first, second = drex.bodies[0]["images"]
    assert first == {
        "type": "workspace_file",
        "workspace_id": "ws_1",
        "file_id": "f_1",
    }
    assert second.startswith("data:image/png;base64,")
    assert deleted == [("ws_1", "f_1")]


def test_upload_is_deleted_even_when_the_request_fails():
    drex = FakeDrex([error(422, "images.0: unusable image")] * 2)
    model = nace(drex, model="drex-max")
    client = model.load_model()
    deleted = []
    client.documents.upload = lambda data, file_name: SimpleNamespace(
        workspace_id="ws", file_id="f"
    )
    client.documents.delete_file = lambda ws, fid: deleted.append(fid)
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * DREX_MAX_BODY_BYTES

    with pytest.raises(DeepEvalError):
        model.decide("[Image 1]", NOUL, images=[image(big)])
    assert deleted == ["f"]


###############################################
# Limits
###############################################


@pytest.mark.parametrize("model", ["drex-max", "drex-max-v1.1", "drex-v2.0"])
def test_more_than_512_questions_refused(drex, model):
    questions = {f"q{i}": NOUL["q"] for i in range(513)}

    with pytest.raises(SystemOneQuestionLimitError, match="at most 512") as e:
        nace(drex, model=model).decide("s", questions)
    assert e.value.question_count == 513
    assert drex.requests == []


def test_state_over_drex_2_limit_refused_before_sending(drex):
    with pytest.raises(SystemOneContextLimitError, match="state limit"):
        nace(drex, model="drex-v2.0").decide("x" * 3 * 140_000, NOUL)
    assert drex.requests == []


def test_body_over_4_mib_refused_before_sending(drex):
    state = "x" * (DREX_MAX_BODY_BYTES + 1)
    with pytest.raises(SystemOneContextLimitError):
        nace(drex, model="drex-max").decide(state, NOUL)
    assert drex.requests == []


def test_jev_budget_is_the_default_for_other_models():
    # A custom System One model keeps the conservative Jev budget it had.
    from deepeval.models.base_model import DeepEvalBaseSystemOneModel

    with pytest.raises(SystemOneContextLimitError):
        DeepEvalBaseSystemOneModel.check_budget(None, "x" * 200_000, NOUL)


###############################################
# Errors and retries
###############################################


def test_422_lists_each_issue_path_and_message():
    drex = FakeDrex(
        [
            error(
                422,
                "questions.q.criteria: score needs at least 1 level",
                type="invalid_request_error",
                issues=[
                    {
                        "path": "questions.q.criteria",
                        "message": "score needs at least 1 level",
                    },
                    {"path": "model", "message": "unknown model"},
                ],
            )
        ]
    )

    with pytest.raises(DeepEvalError) as info:
        nace(drex).decide("s", NOUL)

    text = str(info.value)
    assert "questions.q.criteria: score needs at least 1 level" in text
    assert "model: unknown model" in text
    assert "req_err" in text
    assert len(drex.requests) == 1  # a 422 is never retried


def test_request_too_long_is_a_context_limit():
    drex = FakeDrex(
        [
            error(
                422,
                "Request too long: the state exceeds the 131,072-token state "
                "limit of drex-v2.0. Shorten the state.",
                type="invalid_request_error",
                issues=[{"path": "state", "message": "exceeds the limit"}],
            )
        ]
    )

    with pytest.raises(SystemOneContextLimitError, match="too large"):
        nace(drex).decide("s", NOUL)


def test_413_from_the_hosting_layer_is_a_context_limit():
    drex = FakeDrex([httpx.Response(413, text="413 Request Entity Too Large")])

    with pytest.raises(SystemOneContextLimitError):
        nace(drex).decide("s", NOUL)


def test_402_points_at_billing():
    drex = FakeDrex(
        [error(402, "This account has insufficient available funds.")]
    )

    with pytest.raises(DeepEvalError, match="dashboard/billing"):
        nace(drex).decide("s", NOUL)
    assert len(drex.requests) == 1


def test_402_on_an_image_upload_points_at_billing():
    drex = FakeDrex(
        [error(402, "This account has insufficient available funds.")]
    )
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * DREX_MAX_BODY_BYTES

    with pytest.raises(DeepEvalError, match="dashboard/billing"):
        nace(drex).decide("[Image 1]", NOUL, images=[image(big)])
    assert "/systemone" not in str(drex.requests[0].url)


def test_401_is_not_retried():
    import nace_sdk

    drex = FakeDrex([error(401, "Invalid API key.")])

    with pytest.raises(nace_sdk.NaceAuthenticationError):
        nace(drex).decide("s", NOUL)
    assert len(drex.requests) == 1


@pytest.mark.parametrize("status", [429, 529])
def test_rate_limit_and_overload_are_retried(status, monkeypatch):
    monkeypatch.setenv("DEEPEVAL_RETRY_CAP_SECONDS", "0")
    drex = FakeDrex([error(status, "Retry shortly.")])

    answers, _ = nace(drex).decide("s", NOUL)

    assert len(drex.requests) == 2
    assert answers.nouls["q"].probability == 0.9


def test_missing_key_fails_at_construction():
    with pytest.raises(DeepEvalError, match="NACE_API_KEY"):
        NaceModel()


###############################################
# Headers and backends
###############################################


def test_requests_carry_key_and_deepeval_attribution(drex):
    nace(drex).decide("s", NOUL)

    [request] = drex.requests
    assert request.url == "https://console.nace.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer nace_sk_test"
    assert request.headers["x-nace-client"] == f"deepeval/{__version__}"


def test_base_url_is_configurable(drex):
    nace(drex, base_url="https://drex.example.com/").decide("s", NOUL)

    assert str(drex.requests[0].url) == "https://drex.example.com/v1/systemone"


@pytest.mark.parametrize("model_id", ["drex-max", "nace-ai/drex-max"])
def test_openrouter_backend(drex, model_id):
    model = nace(drex, base_url="https://openrouter.ai/api", model=model_id)
    model.decide(
        {"ticket": "see [Image 1]"},
        {"q": NoulQuestion(instructions=None, true="Shows damage")},
        images=[image(RED)],
    )

    [request] = drex.requests
    body = json.loads(request.content)
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert body["model"] == "nace-ai/drex-max"
    # OpenRouter needs instructions on every question and takes images in
    # a state content array, never a top-level `images`.
    assert body["questions"]["q"]["instructions"] == ""
    assert "images" not in body
    text, picture = body["state"]
    assert text == {"type": "text", "text": '{"ticket": "see [Image 1]"}'}
    assert picture["type"] == "image_url"
    assert picture["image_url"]["url"].startswith("data:image/png;base64,")
    assert request.headers["http-referer"] == "https://nace.ai"
    assert request.headers["x-openrouter-title"] == "Drex for deepeval"


@pytest.mark.parametrize(
    "base_url, env, api_key, sent",
    [
        ("https://openrouter.ai/api", {}, None, None),
        (
            "https://openrouter.ai/api",
            {"OPENROUTER_API_KEY": "sk-or-env"},
            None,
            "sk-or-env",
        ),
        ("https://openrouter.ai/api", {}, "sk-or-arg", "sk-or-arg"),
        ("https://drex.example.com", {}, None, "nace_sk_env"),
    ],
)
def test_openrouter_never_gets_the_nace_key(
    monkeypatch, drex, base_url, env, api_key, sent
):
    from deepeval.config.settings import reset_settings

    monkeypatch.setenv("NACE_API_KEY", "nace_sk_env")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reset_settings(reload_dotenv=False)

    def build():
        return NaceModel(
            base_url=base_url,
            api_key=api_key,
            transport=httpx.MockTransport(drex),
        )

    if sent is None:
        with pytest.raises(DeepEvalError, match="OPENROUTER_API_KEY"):
            build()
    else:
        build().decide("s", NOUL)
        assert drex.requests[0].headers["authorization"] == f"Bearer {sent}"


def test_openrouter_refuses_images_too_large_to_inline(drex):
    model = nace(drex, base_url="https://openrouter.ai/api", model="drex-max")
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * DREX_MAX_BODY_BYTES

    with pytest.raises(DeepEvalError, match="Downscale"):
        model.decide("s", NOUL, images=[image(big)])
    assert drex.requests == []


###############################################
# Provider selection
###############################################


def test_decision_provider_selects_nace(monkeypatch):
    from deepeval.config.settings import reset_settings
    from deepeval.metrics.utils import initialize_system_one_model

    monkeypatch.setenv("DEEPEVAL_DECISION_PROVIDER", "nace")
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_test")
    monkeypatch.setenv("NACE_MODEL_NAME", "drex-max")
    reset_settings(reload_dotenv=False)

    model = initialize_system_one_model(None, "system_one")

    assert isinstance(model, NaceModel)
    assert model.name == "drex-max"


def test_decision_provider_defaults_to_typesafe(monkeypatch):
    pytest.importorskip("typesafe_sdk")
    from deepeval.config.settings import reset_settings
    from deepeval.metrics.utils import initialize_system_one_model

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test")
    reset_settings(reload_dotenv=False)

    assert isinstance(
        initialize_system_one_model(None, "system_one"), TypeSafeModel
    )


def test_unusable_nace_says_how_to_configure_it(monkeypatch):
    from deepeval.config.settings import reset_settings
    from deepeval.metrics.utils import initialize_system_one_model

    monkeypatch.setenv("DEEPEVAL_DECISION_PROVIDER", "nace")
    reset_settings(reload_dotenv=False)

    with pytest.raises(DeepEvalError, match="deepeval set-nace"):
        initialize_system_one_model(None, "hybrid")
