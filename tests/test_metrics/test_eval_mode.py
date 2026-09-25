"""Eval modes (`llm` / `hybrid` / `system_one`) and the System One fallback
switches, exercised on a small metric that has both a whole-chain System One
form and a legacy QAG chain. No network: Jev and the LLM are fakes."""

import json
from typing import List, Optional

import pytest

from deepeval.config.eval_mode import (
    EvalMode,
    resolve_eval_mode,
)
from deepeval.errors import DeepEvalError
from deepeval.metrics import BaseMetric
from deepeval.metrics.base_metric import YES_NO, Verdict
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.jev_eval import Noul
from deepeval.metrics.utils import (
    SystemOneEvalSpec,
    SystemOneVerdictSpec,
    check_llm_test_case_params,
    generate_qag_verdicts,
    initialize_model,
    initialize_system_one_model,
    run_system_one_eval,
    a_run_system_one_eval,
    score_qag_verdicts,
)
from deepeval.metrics.utils.decision import _system_one_active
from deepeval.models.system_one.limits import SystemOneContextLimitError
from deepeval.models.system_one.schema import (
    NoulAnswer,
    ScoreAnswer,
    SystemOneAnswers,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

from tests.test_metrics.system_one_fakes import (
    CannedLLM,
    ExplodingLLM,
    ExplodingSystemOneModel,
    FakeSystemOneModel,
    ScriptedLLM,
    noul_answers,
)


###############################################
# A toy metric with both chains
###############################################


class ToyVerdict(BaseModel):
    verdict: Verdict
    reason: Optional[str] = None


class ToyVerdicts(BaseModel):
    verdicts: List[ToyVerdict]


TOY_QUESTIONS = [
    Noul("actual_output answers input.", weight=2),
    Noul("actual_output is polite."),
]


class ToyMetric(BaseMetric):
    """Whole-chain form: two Nouls over input/actual_output. Legacy chain:
    split actual_output into sentences (in code, to stand in for the LLM
    extraction step) and ask the judge for one yes/no verdict per sentence.
    Under `hybrid` the verdicts come from Jev (the LLM covering a failed
    Jev call); under `llm` from the LLM; under `system_one` there is no LLM
    and only the whole-chain form runs."""

    _required_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

    def __init__(
        self,
        model=None,
        system_one_model=None,
        eval_mode=None,
        include_reason=True,
        strict_mode=False,
        threshold=0.5,
        spec=True,
    ):
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.threshold = 1 if strict_mode else threshold
        self.async_mode = False
        self.verbose_mode = False
        self._spec = spec
        self.path = None

    def _system_one_eval_spec(self, test_case):
        if not self._spec:
            return None
        return SystemOneEvalSpec(
            evaluation_params=self._required_params, questions=TOY_QUESTIONS
        )

    def measure(self, test_case, _show_indicator=False, _in_component=False):
        check_llm_test_case_params(
            test_case, self._required_params, None, None, self, self.model
        )
        self.evaluation_cost = 0
        with metric_progress_indicator(self, _show_indicator=False):
            if run_system_one_eval(self, test_case):
                self.path = "system_one"
                return self.score
            self.path = "legacy"
            self._legacy(test_case)
            return self.score

    async def a_measure(
        self, test_case, _show_indicator=False, _in_component=False
    ):
        check_llm_test_case_params(
            test_case, self._required_params, None, None, self, self.model
        )
        self.evaluation_cost = 0
        with metric_progress_indicator(self, _show_indicator=False):
            if await a_run_system_one_eval(self, test_case):
                self.path = "system_one"
                return self.score
            self.path = "legacy"
            self._legacy(test_case)
            return self.score

    def _legacy(self, test_case):
        sentences = [
            s.strip() for s in test_case.actual_output.split(".") if s.strip()
        ]
        self.verdicts = generate_qag_verdicts(
            metric=self,
            prompt="judge these",
            verdict_cls=ToyVerdict,
            verdicts_cls=ToyVerdicts,
            allowed=YES_NO,
            system_one=SystemOneVerdictSpec(
                instructions="Does `sentence` answer `input`?",
                items=sentences,
                item_key="sentence",
                state={"input": test_case.input},
            ),
        )
        self.score = score_qag_verdicts(
            self, self.verdicts, passing=(Verdict.YES,)
        )
        self.reason = (
            "LLM reason" if self.include_reason and self.verdicts else None
        )
        self.success = self.is_successful()

    @property
    def __name__(self):
        return "Toy"


TEST_CASE = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France. Have a nice day.",
)

CONFIDENT = SystemOneAnswers(
    nouls={
        "q_0": NoulAnswer(probability=0.95),  # confidence 0.90
        "q_1": NoulAnswer(probability=0.85),  # confidence 0.70
    }
)
UNSURE = SystemOneAnswers(
    nouls={
        "q_0": NoulAnswer(probability=0.95),
        "q_1": NoulAnswer(probability=0.55),  # confidence 0.10
    }
)

LLM_VERDICTS = json.dumps({"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "DEEPEVAL_MODE",
        "DEEPEVAL_EVAL_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_DISABLE_LEGACY_KEYFILE", "1")
    yield


###############################################
# Mode resolution
###############################################


def test_default_mode_is_llm():
    assert resolve_eval_mode() is EvalMode.LLM


def test_feature_channel_does_not_turn_jev_on(monkeypatch):
    # The `experimental` channel used to imply `hybrid`; Jev is now opted
    # into through the eval mode alone.
    monkeypatch.setenv("DEEPEVAL_MODE", "experimental")
    assert resolve_eval_mode() is EvalMode.LLM


def test_setting_is_independent_of_feature_channel(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_MODE", "experimental")
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "llm")
    assert resolve_eval_mode() is EvalMode.LLM
    monkeypatch.setenv("DEEPEVAL_MODE", "stable")
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "system_one")
    assert resolve_eval_mode() is EvalMode.SYSTEM_ONE
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "hybrid")
    assert resolve_eval_mode() is EvalMode.HYBRID


def test_kwarg_beats_setting(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "system_one")
    assert resolve_eval_mode("llm") is EvalMode.LLM
    assert resolve_eval_mode(EvalMode.HYBRID) is EvalMode.HYBRID
    assert resolve_eval_mode("SYSTEM_ONE") is EvalMode.SYSTEM_ONE
    # One spelling per mode: no aliases on the kwarg.
    with pytest.raises(ValueError):
        resolve_eval_mode("system-one")
    with pytest.raises(ValueError):
        resolve_eval_mode("jev")


def test_unrecognised_setting_means_unset(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "banana")
    assert resolve_eval_mode() is EvalMode.LLM


def test_unrecognised_kwarg_raises():
    with pytest.raises(ValueError, match="banana"):
        resolve_eval_mode("banana")


def test_llm_mode_builds_no_system_one_model():
    metric = ToyMetric(model=CannedLLM(), eval_mode="llm")
    assert metric.system_one_model is None
    assert not _system_one_active(metric, object())


def test_noul_confidence_is_symmetric():
    assert NoulAnswer(probability=0.95).confidence == pytest.approx(0.9)
    assert NoulAnswer(probability=0.05).confidence == pytest.approx(0.9)
    assert NoulAnswer(probability=0.5).confidence == pytest.approx(0.0)
    assert CONFIDENT.min_confidence() == pytest.approx(0.7)


###############################################
# `llm`: nothing changes
###############################################


def test_llm_mode_runs_legacy_chain_only():
    llm = CannedLLM(LLM_VERDICTS)
    metric = ToyMetric(model=llm, eval_mode="llm")
    metric.measure(TEST_CASE)
    assert metric.path == "legacy"
    assert metric.score == 0.5
    assert metric.reason == "LLM reason"
    assert metric.confidence is None
    assert metric.system_one_fallback_reason is None
    assert len(llm.prompts) == 1


###############################################
# `hybrid`: Jev at the decision step, LLM covers failed calls
###############################################


def test_hybrid_uses_jev_for_verdicts_and_llm_for_reason():
    jev = FakeSystemOneModel(answer_fn=noul_answers(0.9))
    metric = ToyMetric(
        model=ExplodingLLM(), system_one_model=jev, eval_mode="hybrid"
    )
    metric.measure(TEST_CASE)
    assert metric.path == "legacy"  # the chain ran; Jev took the verdicts
    assert metric.score == 1.0
    assert len(jev.calls) == 1
    _, questions = jev.calls[0]
    assert list(questions) == ["sentence_0", "sentence_1"]
    assert metric.confidence == pytest.approx(0.8)
    assert metric.reason == "LLM reason"


def test_hybrid_ignores_whole_chain_spec():
    jev = FakeSystemOneModel(answer_fn=noul_answers(0.9))
    metric = ToyMetric(
        model=CannedLLM(LLM_VERDICTS),
        system_one_model=jev,
        eval_mode="hybrid",
    )
    metric.measure(TEST_CASE)
    # Not the two TOY_QUESTIONS: the per-sentence Nouls of the legacy chain.
    assert list(jev.calls[0][1]) == ["sentence_0", "sentence_1"]


def test_hybrid_falls_back_to_llm_when_jev_call_fails():
    jev = ExplodingSystemOneModel(ConnectionError("jev is down"))
    llm = CannedLLM(LLM_VERDICTS)
    metric = ToyMetric(model=llm, system_one_model=jev, eval_mode="hybrid")
    metric.measure(TEST_CASE)
    assert metric.path == "legacy"
    assert metric.score == 0.5  # the LLM's verdicts
    assert len(llm.prompts) == 1
    assert "ConnectionError: jev is down" in metric.system_one_fallback_reason


def test_hybrid_falls_back_on_context_limit():
    jev = ExplodingSystemOneModel(
        SystemOneContextLimitError(
            "too big", estimated_tokens=99_999, limit_tokens=32_000
        )
    )
    metric = ToyMetric(
        model=CannedLLM(LLM_VERDICTS),
        system_one_model=jev,
        eval_mode="hybrid",
    )
    metric.measure(TEST_CASE)
    assert metric.score == 0.5
    assert (
        "context limit (99999 est. tokens > 32000)"
        in metric.system_one_fallback_reason
    )


def test_hybrid_permanent_error_keeps_measure_on_llm():
    jev = ExplodingSystemOneModel(
        DeepEvalError("TypeSafe AI API key is missing")
    )
    metric = ToyMetric(
        model=CannedLLM(LLM_VERDICTS),
        system_one_model=jev,
        eval_mode="hybrid",
    )
    metric.measure(TEST_CASE)
    assert metric._system_one_disabled is True
    metric.measure(TEST_CASE)  # re-enabled for the next measure
    assert len(jev.calls) == 2


def test_hybrid_never_swallows_programming_errors():
    jev = ExplodingSystemOneModel(KeyError("sentence_7"))
    metric = ToyMetric(
        model=CannedLLM(LLM_VERDICTS),
        system_one_model=jev,
        eval_mode="hybrid",
    )
    with pytest.raises(KeyError):
        metric.measure(TEST_CASE)


###############################################
# `system_one`: whole chain on Jev
###############################################


def test_system_one_decides_whole_chain_with_no_llm_call():
    jev = FakeSystemOneModel(CONFIDENT)
    metric = ToyMetric(
        model=ExplodingLLM(), system_one_model=jev, eval_mode="system_one"
    )
    score = metric.measure(TEST_CASE)
    assert metric.path == "system_one"
    assert score == pytest.approx((2 * 0.95 + 0.85) / 3)
    assert metric.success is True
    assert metric.confidence == pytest.approx(0.7)
    assert metric.system_one_fallback_reason is None
    assert metric.evaluation_model == "fake-jev"
    assert len(metric.score_breakdown) == 2
    # One request, over the raw test case, with the metric's questions.
    assert len(jev.calls) == 1
    state, questions = jev.calls[0]
    assert set(state["test_case"]) == {"input", "actual_output"}
    assert list(questions) == ["q_0", "q_1"]


def test_system_one_reason_is_deterministic():
    metric = ToyMetric(
        model=ExplodingLLM(),
        system_one_model=FakeSystemOneModel(CONFIDENT),
        eval_mode="system_one",
    )
    metric.measure(TEST_CASE)
    reason = metric.reason
    assert reason.startswith("Decided by fake-jev, minimum confidence 0.70.")
    assert (
        "1. actual_output answers input. -> clearly holds (P(yes)=0.95, "
        "confidence=0.90, weight=2)" in reason
    )
    assert reason.endswith(
        "Score: 0.92 (weighted mean of 2 applicable questions)."
    )


def test_system_one_include_reason_false():
    metric = ToyMetric(
        model=ExplodingLLM(),
        system_one_model=FakeSystemOneModel(CONFIDENT),
        eval_mode="system_one",
        include_reason=False,
    )
    metric.measure(TEST_CASE)
    assert metric.path == "system_one"
    assert metric.reason is None


def test_system_one_strict_mode():
    metric = ToyMetric(
        model=ExplodingLLM(),
        system_one_model=FakeSystemOneModel(UNSURE),
        eval_mode="system_one",
        strict_mode=True,
    )
    metric.measure(TEST_CASE)
    assert metric.path == "system_one"
    assert metric.threshold == 1
    assert metric.score == 1.0  # both Nouls >= 0.5
    assert all(o["passed"] for o in metric.score_breakdown)
    assert "strict=pass" in metric.reason


@pytest.mark.asyncio
async def test_system_one_async_path():
    jev = FakeSystemOneModel(CONFIDENT)
    metric = ToyMetric(
        model=ExplodingLLM(), system_one_model=jev, eval_mode="system_one"
    )
    score = await metric.a_measure(TEST_CASE)
    assert metric.path == "system_one"
    assert score == pytest.approx((2 * 0.95 + 0.85) / 3)


def test_system_one_builds_no_llm():
    # No LLM key is needed: `initialize_model` skips the LLM under
    # `system_one`, whatever `model` was passed.
    metric = ToyMetric(
        model=ExplodingLLM(),
        system_one_model=FakeSystemOneModel(CONFIDENT),
        eval_mode="system_one",
    )
    assert metric.model is None
    assert metric.evaluation_model == "fake-jev"


def test_system_one_keeps_low_confidence_result():
    metric = ToyMetric(
        system_one_model=FakeSystemOneModel(UNSURE), eval_mode="system_one"
    )
    metric.measure(TEST_CASE)
    assert metric.path == "system_one"
    assert metric.confidence == pytest.approx(0.1)
    assert metric.system_one_fallback_reason is None


def test_system_one_surfaces_jev_errors_unchanged():
    jev = ExplodingSystemOneModel(ConnectionError("jev is down"))
    metric = ToyMetric(system_one_model=jev, eval_mode="system_one")
    with pytest.raises(ConnectionError, match="jev is down"):
        metric.measure(TEST_CASE)


def test_system_one_context_limit_says_switch_to_llm():
    jev = ExplodingSystemOneModel(
        SystemOneContextLimitError(
            "too big", estimated_tokens=99_999, limit_tokens=32_000
        )
    )
    metric = ToyMetric(system_one_model=jev, eval_mode="system_one")
    with pytest.raises(DeepEvalError, match='eval_mode="llm"') as info:
        metric.measure(TEST_CASE)
    assert isinstance(info.value.__cause__, SystemOneContextLimitError)


def test_system_one_context_precheck_runs_before_the_request():
    jev = FakeSystemOneModel(CONFIDENT)
    metric = ToyMetric(system_one_model=jev, eval_mode="system_one")
    huge = LLMTestCase(input="q", actual_output="x" * 200_000)
    with pytest.raises(DeepEvalError, match="set-eval-mode llm"):
        metric.measure(huge)
    assert jev.calls == []


def test_system_one_without_whole_chain_form_raises():
    metric = ToyMetric(
        system_one_model=FakeSystemOneModel(CONFIDENT),
        eval_mode="system_one",
        spec=False,
    )
    with pytest.raises(DeepEvalError, match="cannot judge this test case"):
        metric.measure(TEST_CASE)


###############################################
# `system_one`: no fallback
###############################################


###############################################
# Built-in metrics
###############################################


def _answer_relevancy(**kwargs):
    from deepeval.metrics import AnswerRelevancyMetric

    kwargs.setdefault("async_mode", False)
    return AnswerRelevancyMetric(**kwargs)


def test_builtin_system_one_whole_chain():
    jev = FakeSystemOneModel(
        answer_fn=lambda qs: SystemOneAnswers(
            nouls={
                "q_0": NoulAnswer(probability=0.9),
                "q_1": NoulAnswer(probability=0.8),
            },
            scores={
                "q_2": ScoreAnswer(
                    score=2.7,
                    probabilities={0: 0.0, 1: 0.1, 2: 0.1, 3: 0.8},
                    confidence=0.73,
                )
            },
        )
    )
    metric = _answer_relevancy(
        model=ExplodingLLM(), system_one_model=jev, eval_mode="system_one"
    )
    score = metric.measure(TEST_CASE)
    # (2*0.9 + 0.8 + 2.7/3) / 4
    assert score == pytest.approx((1.8 + 0.8 + 0.9) / 4)
    assert metric.reason.startswith("Decided by fake-jev")
    assert "Every statement in `actual_output` is relevant" in metric.reason
    assert metric.confidence == pytest.approx(0.6)  # the 0.8 Noul
    assert metric.evaluation_model == "fake-jev"
    assert len(jev.calls) == 1
    state, questions = jev.calls[0]
    assert set(state["test_case"]) == {"input", "actual_output"}
    assert len(questions) == 3
    assert "System One" in metric.verbose_logs


def test_builtin_system_one_surfaces_errors():
    jev = ExplodingSystemOneModel(ConnectionError("down"))
    metric = _answer_relevancy(system_one_model=jev, eval_mode="system_one")
    with pytest.raises(ConnectionError, match="down"):
        metric.measure(TEST_CASE)
    assert len(jev.calls) == 1


def test_builtin_hybrid_unchanged():
    jev = FakeSystemOneModel(answer_fn=noul_answers(0.9))
    llm = ScriptedLLM(
        [
            json.dumps({"statements": ["Paris is the capital of France."]}),
            json.dumps({"reason": "fine"}),
        ]
    )
    metric = _answer_relevancy(
        model=llm, system_one_model=jev, eval_mode="hybrid"
    )
    assert metric.measure(TEST_CASE) == 1.0
    assert metric.reason == "fine"
    assert len(jev.calls) == 1
    assert list(jev.calls[0][1]) == ["statement_0"]
    assert metric.confidence == pytest.approx(0.8)
    assert metric.system_one_fallback_reason is None


def test_builtin_llm_mode_never_touches_jev():
    llm = ScriptedLLM(
        [
            json.dumps({"statements": ["Paris is the capital of France."]}),
            json.dumps({"verdicts": [{"verdict": "yes"}]}),
            json.dumps({"reason": "fine"}),
        ]
    )
    metric = _answer_relevancy(model=llm, eval_mode="llm")
    assert metric.system_one_model is None
    assert metric.measure(TEST_CASE) == 1.0
    assert metric.confidence is None


def test_builtin_system_one_rejects_multimodal():
    jev = FakeSystemOneModel(answer_fn=noul_answers(0.9))
    metric = _answer_relevancy(system_one_model=jev, eval_mode="system_one")
    multimodal = LLMTestCase(
        input="what is this? [DEEPEVAL:IMAGE:https://x/y.png]",
        actual_output="a cat",
        multimodal=True,
    )
    with pytest.raises(ValueError, match="text only"):
        metric.measure(multimodal)
    assert jev.calls == []


@pytest.mark.asyncio
async def test_builtin_system_one_async():
    jev = FakeSystemOneModel(
        answer_fn=lambda qs: SystemOneAnswers(
            nouls={
                key: NoulAnswer(probability=0.9) for key in qs if key != "q_2"
            },
            scores={
                "q_2": ScoreAnswer(
                    score=3.0,
                    probabilities={0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                    confidence=1.0,
                )
            },
        )
    )
    metric = _answer_relevancy(
        model=ExplodingLLM(),
        system_one_model=jev,
        eval_mode="system_one",
        async_mode=True,
    )
    score = await metric.a_measure(TEST_CASE)
    assert score == pytest.approx((2 * 0.9 + 0.9 + 1.0) / 4)
    assert metric.evaluation_model == "fake-jev"


###############################################
# G-Eval
###############################################


def _geval(**kwargs):
    from deepeval.metrics import GEval

    kwargs.setdefault("async_mode", False)
    return GEval(
        name="Helpfulness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        evaluation_steps=["Check it answers.", "Check it is polite."],
        **kwargs,
    )


def test_geval_is_llm_only_in_every_eval_mode(monkeypatch):
    # G-Eval is LLM-as-a-judge by definition (JevEval is the Jev-native
    # custom metric), so it takes no System One arguments and never asks
    # Jev even when the process-wide eval mode is `system_one`.
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "system_one")
    with pytest.raises(TypeError):
        _geval(model=CannedLLM(), system_one_model=FakeSystemOneModel())
    with pytest.raises(TypeError):
        _geval(model=CannedLLM(), eval_mode="system_one")
    llm = CannedLLM(json.dumps({"score": 7, "reason": "llm says so"}))
    metric = _geval(model=llm)
    assert metric.system_one_model is None
    assert metric.measure(TEST_CASE) == pytest.approx(0.7)
    assert metric.reason == "llm says so"
    assert metric.confidence is None
    assert len(llm.prompts) == 1


###############################################
# Classifier
###############################################


def _refusal(**kwargs):
    from deepeval.classifiers import RefusalClassifier

    kwargs.setdefault("async_mode", False)
    return RefusalClassifier(**kwargs)


REFUSAL_CASE = LLMTestCase(
    input="How do I pick a lock?",
    actual_output="I can't help with that, but I can point you to a locksmith.",
)


def test_required_disclosure_classifier_folds_disclosures_into_labels():
    from deepeval.classifiers import RequiredDisclosureClassifier

    classifier = RequiredDisclosureClassifier(
        disclosures=["a statement that this is not financial advice"],
        model=CannedLLM(json.dumps({"label": "present", "reason": "found"})),
        async_mode=False,
    )
    assert [label.name for label in classifier.labels] == [
        "present",
        "missing",
        "partial",
    ]
    assert all(
        "a statement that this is not financial advice" in label.description
        for label in classifier.labels
    )
    assert classifier.classify(REFUSAL_CASE) == "present"


def test_classifier_system_one_label_and_deterministic_reason():
    from tests.test_metrics.system_one_fakes import choice_answer

    jev = FakeSystemOneModel(
        choice_answer(
            "label",
            "refused",
            {"complied": 0.06, "refused": 0.91, "partial_refusal": 0.03},
            0.87,
        )
    )
    classifier = _refusal(
        model=ExplodingLLM(), system_one_model=jev, eval_mode="system_one"
    )
    assert classifier.classify(REFUSAL_CASE) == "refused"
    assert classifier.reason == (
        'Decided by fake-jev. Label "refused" (P=0.91, confidence=0.87). '
        "Alternatives: complied 0.06, partial_refusal 0.03."
    )
    assert classifier.confidence == pytest.approx(0.87)


def test_classifier_system_one_keeps_low_confidence_label():
    from tests.test_metrics.system_one_fakes import choice_answer

    jev = FakeSystemOneModel(
        choice_answer(
            "label",
            "refused",
            {"complied": 0.40, "refused": 0.45, "partial_refusal": 0.15},
            0.2,
        )
    )
    classifier = _refusal(system_one_model=jev, eval_mode="system_one")
    assert classifier.model is None
    assert classifier.classify(REFUSAL_CASE) == "refused"
    assert classifier.confidence == pytest.approx(0.2)


def test_classifier_system_one_raises_jev_errors():
    jev = ExplodingSystemOneModel(ConnectionError("down"))
    classifier = _refusal(system_one_model=jev, eval_mode="system_one")
    with pytest.raises(ConnectionError):
        classifier.classify(REFUSAL_CASE)


@pytest.mark.parametrize("eval_mode", ["hybrid", EvalMode.HYBRID])
def test_classifier_hybrid_runs_as_llm(monkeypatch, eval_mode):
    # Classifiers have no `hybrid`: `deepeval set-eval-mode hybrid` (and a
    # stray `eval_mode="hybrid"` that bypasses the type hint) run them as
    # `llm`, silently, and never touch Jev.
    llm = CannedLLM(json.dumps({"label": "refused", "reason": "llm reason"}))
    jev = ExplodingSystemOneModel(ConnectionError("should not be called"))

    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "hybrid")
    from_setting = _refusal(model=llm, system_one_model=jev)
    assert from_setting.eval_mode is EvalMode.LLM
    assert from_setting.classify(REFUSAL_CASE) == "refused"
    assert from_setting.reason == "llm reason"

    from_kwarg = _refusal(model=llm, system_one_model=jev, eval_mode=eval_mode)
    assert from_kwarg.eval_mode is EvalMode.LLM
    assert from_kwarg.classify(REFUSAL_CASE) == "refused"


def test_classifier_copy_keeps_eval_mode():
    jev = FakeSystemOneModel(SystemOneAnswers())
    classifier = _refusal(
        model=CannedLLM(), system_one_model=jev, eval_mode="system_one"
    )
    copied = classifier.copy()
    assert copied.eval_mode is EvalMode.SYSTEM_ONE
    assert copied.system_one_model is jev


def test_per_measure_state_is_reset():
    # Measure 1: the Jev call fails, the LLM covers it. Measure 2: Jev
    # answers, so nothing from measure 1 should linger.
    failures = iter([ConnectionError("blip"), None])

    def answer(questions):
        exc = next(failures)
        if exc is not None:
            raise exc
        return noul_answers(0.9)(questions)

    jev = FakeSystemOneModel(answer_fn=answer)
    metric = ToyMetric(
        model=CannedLLM(LLM_VERDICTS),
        system_one_model=jev,
        eval_mode="hybrid",
    )
    metric.measure(TEST_CASE)
    assert metric.system_one_fallback_reason is not None
    assert metric.confidence is None
    metric.measure(TEST_CASE)
    assert metric.system_one_fallback_reason is None
    assert metric.confidence == pytest.approx(0.8)
