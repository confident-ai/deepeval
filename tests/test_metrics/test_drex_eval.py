"""DrexEval, and every System One path judged by Nace's Drex.

DrexEval is JevEval with Drex as its default judge, so its scoring is checked
against JevEval on a fake model. The rest runs the real ``NaceModel`` over
``FakeDrex`` (the Drex API at the HTTP layer): built-in metrics under
``system_one`` and ``hybrid``, multimodal test cases routed to Drex Max with
their images, the image metrics, and batching under Drex Max's question cap.
"""

import base64
import importlib.util
import json

import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ConversationalDrexEval,
    ConversationalJevEval,
    DrexEval,
    FaithfulnessMetric,
    HallucinationMetric,
    ImageCoherenceMetric,
    ImageEditingMetric,
    JevEval,
    ToxicityMetric,
)
from deepeval.metrics.drex_eval import Choice, Noul, Score
from deepeval.metrics.jev_eval.utils import extract_images
from deepeval.models.system_one.schema import (
    NoulAnswer,
    ScoreAnswer,
    SystemOneAnswers,
)
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MLLMImage,
    MultiTurnParams,
    SingleTurnParams,
    Turn,
)
from tests.test_metrics.system_one_fakes import (
    BLUE,
    RED,
    CannedLLM,
    ExplodingLLM,
    FakeDrex,
    FakeSystemOneModel,
    ScriptedLLM,
    image,
)

needs_nace_sdk = pytest.mark.skipif(
    importlib.util.find_spec("nace_sdk") is None,
    reason="nace-sdk is not installed",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "DEEPEVAL_MODE",
        "DEEPEVAL_EVAL_MODE",
        "DEEPEVAL_DECISION_PROVIDER",
        "NACE_API_KEY",
        "NACE_MODEL_NAME",
        "NACE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_DISABLE_LEGACY_KEYFILE", "1")
    yield


@pytest.fixture
def drex():
    return FakeDrex()


def nace(drex, **kwargs):
    import httpx

    from deepeval.models import NaceModel

    return NaceModel(
        api_key="nace_sk_test", transport=httpx.MockTransport(drex), **kwargs
    )


QUESTIONS = [
    Noul("actual_output answers the question in input.", weight=2),
    Score(
        "How complete is actual_output?",
        levels=["Not at all", "Partly", "Fully"],
    ),
    Choice(
        "How does actual_output handle what it does not know?",
        options={"says_so": 1.0, "guesses": 0.0, "nothing_unknown": None},
    ),
]

ANSWERS = SystemOneAnswers(
    nouls={"q_0": NoulAnswer(probability=0.8)},
    scores={
        "q_1": ScoreAnswer(
            score=1.5, probabilities={0: 0.1, 1: 0.3, 2: 0.6}, confidence=0.5
        )
    },
)

TEST_CASE = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France.",
    retrieval_context=["Paris is the capital and largest city of France."],
    context=["Paris is the capital of France."],
)

PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]


def _choice_answers(answers, choice):
    from deepeval.models.system_one.schema import ChoiceAnswer

    out = answers.model_copy(deep=True)
    out.choices["q_2"] = ChoiceAnswer(
        choice=choice,
        probabilities={"says_so": 0.7, "guesses": 0.2, "nothing_unknown": 0.1},
        confidence=0.6,
    )
    return out


###############################################
# DrexEval is JevEval with another judge
###############################################


def test_drex_eval_scores_exactly_like_jev_eval():
    answers = _choice_answers(ANSWERS, "says_so")
    drex = DrexEval(
        name="Answer",
        evaluation_params=PARAMS,
        questions=QUESTIONS,
        system_one_model=FakeSystemOneModel(answers),
        async_mode=False,
    )
    jev = JevEval(
        name="Answer",
        evaluation_params=PARAMS,
        questions=QUESTIONS,
        system_one_model=FakeSystemOneModel(answers),
        async_mode=False,
    )

    assert drex.measure(TEST_CASE) == pytest.approx(jev.measure(TEST_CASE))
    assert drex.score_breakdown == jev.score_breakdown
    assert drex.confidence == jev.confidence
    # (2*0.8 + 1.5/2 + 0.7/0.9) / 4
    assert drex.score == pytest.approx((1.6 + 0.75 + 0.7 / 0.9) / 4)


def test_drex_eval_names():
    model = FakeSystemOneModel(ANSWERS)
    single = DrexEval(
        name="Answer",
        evaluation_params=PARAMS,
        questions=QUESTIONS,
        system_one_model=model,
    )
    multi = ConversationalDrexEval(
        name="Chat",
        evaluation_params=[MultiTurnParams.CONTENT],
        questions=QUESTIONS,
        system_one_model=model,
    )
    bare = DrexEval(
        name="Answer",
        evaluation_params=PARAMS,
        questions=QUESTIONS,
        system_one_model=model,
        _include_jev_eval_suffix=False,
    )

    assert single.__name__ == "Answer [DrexEval]"
    assert multi.__name__ == "Chat [Conversational DrexEval]"
    assert bare.__name__ == "Answer"
    assert isinstance(single, JevEval)
    assert isinstance(multi, ConversationalJevEval)


def test_jev_eval_still_refuses_images():
    # A System One model that does not read images keeps rejecting them.
    metric = JevEval(
        name="Answer",
        evaluation_params=PARAMS,
        questions=QUESTIONS,
        system_one_model=FakeSystemOneModel(ANSWERS),
    )

    with pytest.raises(ValueError, match="text only"):
        metric.measure(
            LLMTestCase(
                input=f"What is this? {image(RED)}", actual_output="red"
            )
        )


def test_extract_images_numbers_by_first_appearance():
    red, blue = image(RED), image(BLUE)
    state = {
        "test_case": {
            "input": f"Compare {red} and {blue}.",
            "actual_output": [f"{blue} is blue", f"{red} again"],
        }
    }

    numbered, images = extract_images(state)

    assert numbered["test_case"]["input"] == "Compare [Image 1] and [Image 2]."
    assert numbered["test_case"]["actual_output"] == [
        "[Image 2] is blue",
        "[Image 1] again",
    ]
    assert images == [red, blue]


def test_extract_images_refuses_pdfs():
    pdf = MLLMImage(
        dataBase64=base64.b64encode(b"%PDF-1.7").decode(),
        mimeType="application/pdf",
    )

    with pytest.raises(DeepEvalError, match="not PDFs"):
        extract_images(f"read {pdf}")


###############################################
# DrexEval on NaceModel
###############################################


@needs_nace_sdk
def test_drex_eval_defaults_to_nace(monkeypatch):
    from deepeval.config.settings import reset_settings
    from deepeval.models import NaceModel

    monkeypatch.setenv("NACE_API_KEY", "nace_sk_test")
    reset_settings(reload_dotenv=False)

    metric = DrexEval(
        name="Answer", evaluation_params=PARAMS, questions=QUESTIONS
    )

    assert isinstance(metric.system_one_model, NaceModel)
    assert metric.evaluation_model == "drex-max-v1.1 (Nace)"


@needs_nace_sdk
def test_drex_eval_without_key_says_how_to_configure():
    with pytest.raises(DeepEvalError, match="NACE_API_KEY"):
        DrexEval(name="Answer", evaluation_params=PARAMS, questions=QUESTIONS)


@needs_nace_sdk
@pytest.mark.parametrize("async_mode", [False, True])
def test_drex_eval_judges_images_on_drex_max(drex, async_mode):
    red, blue = image(RED), image(BLUE)
    metric = DrexEval(
        name="Colours",
        evaluation_params=PARAMS,
        questions=[Noul("actual_output names the colour of each image.")],
        system_one_model=nace(drex),
        async_mode=async_mode,
    )

    metric.measure(
        LLMTestCase(
            input=f"What colours are {red} and {blue}?",
            actual_output="Red, then blue.",
        )
    )

    [body] = drex.bodies
    assert body["model"] == "drex-max-v1.1"
    assert body["state"]["test_case"]["input"] == (
        "What colours are [Image 1] and [Image 2]?"
    )
    sent = [base64.b64decode(u.split(",", 1)[1]) for u in body["images"]]
    assert sent == [RED, BLUE]
    assert metric.evaluation_model == "drex-max-v1.1 (Nace)"
    assert metric.reason.startswith("Decided by drex-max-v1.1 (Nace)")
    assert metric.score == pytest.approx(0.9)


@needs_nace_sdk
def test_conversational_drex_eval_judges_turn_images(drex):
    metric = ConversationalDrexEval(
        name="Chat",
        evaluation_params=[MultiTurnParams.CONTENT],
        questions=[Noul("The assistant describes the user's image correctly.")],
        system_one_model=nace(drex),
        async_mode=False,
    )

    metric.measure(
        ConversationalTestCase(
            turns=[
                Turn(role="user", content=f"What is in {image(RED)}?"),
                Turn(role="assistant", content="A red square."),
            ]
        )
    )

    [body] = drex.bodies
    assert body["model"] == "drex-max-v1.1"
    assert body["state"]["turns"][0]["content"] == "What is in [Image 1]?"
    assert len(body["images"]) == 1


###############################################
# Built-in metrics on Drex
###############################################

BUILT_INS = [
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
]


@needs_nace_sdk
@pytest.mark.parametrize("metric_cls", BUILT_INS, ids=lambda c: c.__name__)
def test_builtin_system_one_on_drex(drex, metric_cls):
    metric = metric_cls(
        model=ExplodingLLM(),
        system_one_model=nace(drex),
        eval_mode="system_one",
        async_mode=False,
    )

    metric.measure(TEST_CASE)

    [body] = drex.bodies
    assert body["model"] == "drex-max-v1.1"
    assert "images" not in body
    assert metric.score is not None
    assert metric.reason.startswith("Decided by drex-max-v1.1 (Nace)")
    assert metric.evaluation_cost == pytest.approx(1000 * 0.03 / 1e6)


@needs_nace_sdk
def test_builtin_hybrid_on_drex(drex):
    llm = ScriptedLLM(
        [
            json.dumps({"statements": ["Paris is the capital of France."]}),
            json.dumps({"reason": "fine"}),
        ]
    )
    metric = AnswerRelevancyMetric(
        model=llm,
        system_one_model=nace(drex),
        eval_mode="hybrid",
        async_mode=False,
    )

    metric.measure(TEST_CASE)

    assert len(drex.requests) == 1
    assert metric.score == 1.0
    assert metric.system_one_fallback_reason is None


@needs_nace_sdk
def test_multimodal_builtin_goes_to_drex_max(drex):
    metric = AnswerRelevancyMetric(
        system_one_model=nace(drex), eval_mode="system_one", async_mode=False
    )

    metric.measure(
        LLMTestCase(
            input=f"Describe {image(RED)}.", actual_output="A red square."
        )
    )

    [body] = drex.bodies
    assert body["model"] == "drex-max-v1.1"
    assert body["state"]["test_case"]["input"] == "Describe [Image 1]."
    assert len(body["images"]) == 1
    assert metric.reason.startswith("Decided by drex-max-v1.1 (Nace)")


@needs_nace_sdk
def test_multimodal_builtin_on_text_model_is_refused(drex):
    metric = AnswerRelevancyMetric(
        system_one_model=nace(drex, model="drex-v2.0"),
        eval_mode="system_one",
        async_mode=False,
    )

    with pytest.raises(ValueError, match="evaluates text only"):
        metric.measure(
            LLMTestCase(input=f"Describe {image(RED)}.", actual_output="red")
        )
    assert drex.requests == []


###############################################
# Image metrics
###############################################


@needs_nace_sdk
def test_image_coherence_on_drex_max(drex):
    red, blue = image(RED), image(BLUE)
    metric = ImageCoherenceMetric(
        system_one_model=nace(drex), eval_mode="system_one"
    )

    metric.measure(
        LLMTestCase(
            input="Write about colours.",
            actual_output=f"Red: {red}\nBlue: {blue}",
        )
    )

    [body] = drex.bodies
    assert body["model"] == "drex-max-v1.1"
    questions = list(body["questions"].values())
    assert [q["type"] for q in questions] == ["score", "score"]
    assert "[Image 1]" in questions[0]["instructions"]
    assert "[Image 2]" in questions[1]["instructions"]
    assert len(body["images"]) == 2
    assert metric.score == 1.0
    assert metric.reason.startswith("Decided by drex-max-v1.1 (Nace)")


@needs_nace_sdk
def test_image_editing_sends_original_then_edited(drex):
    metric = ImageEditingMetric(
        system_one_model=nace(drex), eval_mode="system_one"
    )

    metric.measure(
        LLMTestCase(
            input=f"Make it blue: {image(RED)}",
            actual_output=f"{image(BLUE)}",
        )
    )

    [body] = drex.bodies
    sent = [base64.b64decode(u.split(",", 1)[1]) for u in body["images"]]
    assert sent == [RED, BLUE]
    assert body["state"]["test_case"] == {
        "input": "Make it blue: [Image 1]",
        "actual_output": "[Image 2]",
    }


def test_image_metrics_stay_on_the_llm_by_default():
    llm = CannedLLM()
    metric = ImageCoherenceMetric(model=llm)

    assert metric.system_one_model is None
    assert metric.evaluation_model == "canned-llm"


@pytest.mark.parametrize("setting", ["system_one", "hybrid"])
def test_eval_mode_setting_keeps_image_metrics_off_a_text_only_judge(
    monkeypatch, setting
):
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", setting)
    llm = CannedLLM()
    metric = ImageCoherenceMetric(
        model=llm, system_one_model=FakeSystemOneModel()
    )

    assert metric.eval_mode == "llm"
    assert metric.system_one_model is None
    assert metric.model is llm


@needs_nace_sdk
def test_system_one_setting_sends_image_metrics_to_drex_max(drex, monkeypatch):
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "system_one")
    metric = ImageCoherenceMetric(system_one_model=nace(drex))

    assert metric.eval_mode == "system_one"
    assert metric.model is None


###############################################
# Batching under Drex Max's question cap
###############################################


@needs_nace_sdk
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_batch_sends_one_drex_request_per_test_case(drex, is_async):
    from deepeval.metrics.utils.system_one_batch import (
        a_measure_system_one_batch,
        measure_system_one_batch,
    )

    model = nace(drex, model="drex-max")
    metrics = [
        AnswerRelevancyMetric(system_one_model=model, eval_mode="system_one")
        for _ in range(7)
    ]

    kwargs = dict(ignore_errors=False, skip_on_missing_params=False)
    if is_async:
        await a_measure_system_one_batch(metrics, TEST_CASE, **kwargs)
    else:
        measure_system_one_batch(metrics, TEST_CASE, **kwargs)

    [body] = drex.bodies
    assert len(body["questions"]) == 7 * 3
    assert all(metric.score is not None for metric in metrics)


###############################################
# Classifiers and model names
###############################################


@needs_nace_sdk
def test_classifier_system_one_on_drex(drex):
    from deepeval.classifiers import RefusalClassifier

    classifier = RefusalClassifier(
        system_one_model=nace(drex), eval_mode="system_one", async_mode=False
    )

    label = classifier.classify(
        LLMTestCase(
            input="How do I pick a lock?",
            actual_output="I can't help with that.",
        )
    )

    [body] = drex.bodies
    assert body["questions"]["label"]["type"] == "choice"
    assert label in body["questions"]["label"]["criteria"]
    assert classifier.reason.startswith("Decided by drex-max-v1.1 (Nace)")


@needs_nace_sdk
def test_drex_model_name_picks_nace_whatever_the_provider(monkeypatch):
    from deepeval.config.settings import reset_settings
    from deepeval.metrics.utils import initialize_system_one_model
    from deepeval.models import NaceModel

    monkeypatch.setenv("NACE_API_KEY", "nace_sk_test")
    reset_settings(reload_dotenv=False)  # provider unset: TypeSafe

    model = initialize_system_one_model("drex-max", "system_one")

    assert isinstance(model, NaceModel)
    assert model.name == "drex-max"
