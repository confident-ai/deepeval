import pytest

from deepeval.config.voice_eval_mode import (
    VoiceEvalMode,
    resolve_voice_eval_mode,
)
from deepeval.errors import DeepEvalError
from deepeval.metrics import (
    AbruptCutoffMetric,
    AgentInterruptionMetric,
    BackgroundNoiseMetric,
    EchoMetric,
    TurnTakingNaturalnessMetric,
    UserInterruptionMetric,
)
from deepeval.models import HumeDecisionModel
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel

AUDIO_CHECK_METRICS = [
    AbruptCutoffMetric,
    AgentInterruptionMetric,
    BackgroundNoiseMetric,
    EchoMetric,
    TurnTakingNaturalnessMetric,
    UserInterruptionMetric,
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "DEEPEVAL_VOICE_EVAL_MODE",
        "DEEPEVAL_EVAL_MODE",
        "HUME_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_DISABLE_LEGACY_KEYFILE", "1")
    yield


def test_default_mode_is_deterministic():
    assert resolve_voice_eval_mode() is VoiceEvalMode.DETERMINISTIC


def test_setting_picks_the_mode(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_VOICE_EVAL_MODE", "Hybrid")
    assert resolve_voice_eval_mode() is VoiceEvalMode.HYBRID


def test_kwarg_beats_setting(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_VOICE_EVAL_MODE", "hybrid")
    assert (
        resolve_voice_eval_mode("deterministic") is VoiceEvalMode.DETERMINISTIC
    )


def test_unrecognised_setting_means_unset(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_VOICE_EVAL_MODE", "llm")
    assert resolve_voice_eval_mode() is VoiceEvalMode.DETERMINISTIC


def test_unrecognised_kwarg_raises():
    with pytest.raises(ValueError, match="deterministic, hybrid"):
        EchoMetric(eval_mode="llm")


def test_text_eval_mode_does_not_turn_the_audio_check_on(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "hybrid")
    assert EchoMetric().audio_decision_model is None


@pytest.mark.parametrize("metric_class", AUDIO_CHECK_METRICS)
def test_deterministic_ignores_a_passed_model(metric_class):
    metric = metric_class(audio_decision_model=FakeAudioDecisionModel())
    assert metric.audio_decision_model is None
    assert metric.evaluation_model is None


@pytest.mark.parametrize("metric_class", AUDIO_CHECK_METRICS)
def test_hybrid_uses_the_passed_model(metric_class):
    model = FakeAudioDecisionModel()
    metric = metric_class(eval_mode="hybrid", audio_decision_model=model)
    assert metric.audio_decision_model is model


def test_hybrid_defaults_to_hume(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_VOICE_EVAL_MODE", "hybrid")
    monkeypatch.setenv("HUME_API_KEY", "key")
    assert isinstance(EchoMetric().audio_decision_model, HumeDecisionModel)


def test_hybrid_without_a_hume_key_fails_at_construction():
    with pytest.raises(DeepEvalError, match="set-voice-eval-mode"):
        EchoMetric(eval_mode="hybrid")


def test_turn_taking_hands_its_mode_and_model_to_the_interruption_checks():
    model = FakeAudioDecisionModel()
    metric = TurnTakingNaturalnessMetric(
        eval_mode="hybrid", audio_decision_model=model
    )
    checked = [
        part.audio_decision_model
        for part in metric._parts()
        if part.audio_decision_model is not None
    ]
    assert checked == [model, model]
