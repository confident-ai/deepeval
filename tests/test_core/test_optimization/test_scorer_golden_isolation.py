import copy

import pytest

from deepeval.dataset import Golden, ConversationalGolden
from deepeval.metrics import BaseMetric, BaseConversationalMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.optimizer.scorer.scorer import Scorer
from deepeval.optimizer.types import PromptConfiguration
from deepeval.prompt import Prompt
from deepeval.test_case import Turn


class OfflineModel(DeepEvalBaseLLM):
    def load_model(self):
        return None

    def generate(self, *args, **kwargs):
        raise AssertionError("No judge calls are needed for this test")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("No judge calls are needed for this test")

    def get_model_name(self):
        return "offline"


class OutputMetric(BaseMetric):
    def measure(self, test_case, **kwargs):
        self.score = float(test_case.actual_output == "correct")
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case)


class ConversationOutputMetric(BaseConversationalMetric):
    def measure(self, test_case, **kwargs):
        self.score = float(test_case.turns[-1].content == "correct")
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case)


class DiagnosisModel(OfflineModel):
    def generate(self, prompt, schema):
        return schema(failures="", successes="Correct answer", analysis="")

    async def a_generate(self, prompt, schema):
        return self.generate(prompt, schema)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", ["single", "user_last", "assistant_last"])
async def test_candidate_scoring_preserves_golden_inputs(kind, async_mode):
    if kind == "single":
        golden = Golden(
            input="Question",
            actual_output="original",
            expected_output="correct",
        )
        metric = OutputMetric()
    else:
        turns = [Turn(role="user", content="Question")]
        if kind == "assistant_last":
            turns.append(Turn(role="assistant", content="original"))
        golden = ConversationalGolden(
            scenario="Answer the question", turns=turns
        )
        metric = ConversationOutputMetric()

    original = copy.deepcopy(golden)
    observed_inputs = []
    retained_inputs = []

    def model_callback(prompt, example):
        observed_inputs.append(copy.deepcopy(example))
        retained_inputs.append(example)
        return prompt.text_template

    scorer = Scorer(
        model_callback=model_callback,
        metrics=[metric],
        max_concurrent=1,
        throttle_seconds=0,
        optimizer_model=OfflineModel(),
    )
    scores = []
    for output in ["correct", "incorrect"]:
        configuration = PromptConfiguration.new(
            {Scorer.DEFAULT_MODULE_ID: Prompt(text_template=output)}
        )
        if async_mode:
            scores.extend(await scorer.a_score_pareto(configuration, [golden]))
        else:
            scores.extend(scorer.score_pareto(configuration, [golden]))

    assert scores == [1.0, 0.0]
    assert observed_inputs == [original, original]
    assert all(example is golden for example in retained_inputs)
    assert golden == original


@pytest.mark.parametrize("async_mode", [False, True])
async def test_feedback_preserves_golden_without_turns(async_mode):
    golden = ConversationalGolden(scenario="Answer a question")
    original = copy.deepcopy(golden)
    scorer = Scorer(
        model_callback=lambda prompt, example: "correct",
        metrics=[ConversationOutputMetric()],
        max_concurrent=1,
        throttle_seconds=0,
        optimizer_model=DiagnosisModel(),
    )
    configuration = PromptConfiguration.new(
        {Scorer.DEFAULT_MODULE_ID: Prompt(text_template="Answer the question")}
    )
    if async_mode:
        feedback = await scorer.a_get_minibatch_feedback(
            configuration, Scorer.DEFAULT_MODULE_ID, [golden]
        )
    else:
        feedback = scorer.get_minibatch_feedback(
            configuration, Scorer.DEFAULT_MODULE_ID, [golden]
        )

    assert len(feedback.results) == 1
    assert "[Actual Model Output]: correct" in feedback.results[0]
    assert "Score: 1.0" in feedback.results[0]
    assert golden == original
