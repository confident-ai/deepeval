import asyncio

import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import AgentJudgeMetric
from deepeval.metrics.community.agent_judge.agent_judge import (
    AgentJudgeMetric as AgentJudgeMetricDirect,
)
from deepeval.metrics.community.agent_judge.schema import (
    AgentJudgment,
    AgentMilestones,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall


class StaticJudgeModel(DeepEvalBaseLLM):
    """Key-free judge returning canned milestones and verdicts."""

    def __init__(self, milestones=None, scores=None):
        self.milestones = milestones or ["Plan the work", "Do the work"]
        self.scores = scores if scores is not None else [1.0, 0.0]
        self.schema_calls = []
        super().__init__(model="static-judge-model")

    def load_model(self):
        return self

    def get_model_name(self):
        return "static-judge-model"

    def generate(self, prompt: str, schema=None):
        return self.generate_with_schema(prompt, schema=schema)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate_with_schema(prompt, schema=schema)

    def generate_with_schema(self, prompt: str, schema=None):
        if schema.__name__ == "AgentMilestones":
            return AgentMilestones(milestones=list(self.milestones))
        if schema.__name__ == "AgentJudgment":
            return AgentJudgment(scores=list(self.scores), reason="ok")
        raise ValueError(f"Unexpected schema: {schema}")

    async def a_generate_with_schema(self, prompt: str, schema=None):
        return self.generate_with_schema(prompt, schema=schema)


def _test_case():
    return LLMTestCase(
        input="Fetch the weather and summarise it",
        actual_output="It is raining in Paris.",
        tools_called=[
            ToolCall(
                name="get_weather",
                input_parameters={"city": "Paris"},
                output={"condition": "rain", "temp": 12},
            )
        ],
    )


class TestAgentJudgeMetric:
    """Runs fully key-free via StaticJudgeModel (no OPENAI_API_KEY)."""

    def test_measure_with_supplied_requirements(self):
        metric = AgentJudgeMetric(
            requirements=["Get the weather", "Mention Paris"],
            model=StaticJudgeModel(scores=[1.0, 0.5]),
        )
        score = metric.measure(_test_case())
        assert score == pytest.approx(0.75)
        assert metric.milestones == ["Get the weather", "Mention Paris"]
        assert metric.milestone_scores == [1.0, 0.5]
        assert metric.is_successful() is True
        assert metric.reason == "ok"

    def test_measure_decomposes_task_into_milestones(self):
        model = StaticJudgeModel(
            milestones=["Get the weather", "Mention Paris"],
            scores=[0.0, 0.0],
        )
        metric = AgentJudgeMetric(threshold=0.5, model=model)
        score = metric.measure(_test_case())
        assert score == pytest.approx(0.0)
        assert metric.milestones == ["Get the weather", "Mention Paris"]
        assert metric.is_successful() is False

    def test_a_measure_matches_sync_score(self):
        metric = AgentJudgeMetric(
            requirements=["Get the weather", "Mention Paris"],
            model=StaticJudgeModel(scores=[1.0, 0.5]),
        )
        score = asyncio.run(metric.a_measure(_test_case()))
        assert score == pytest.approx(0.75)

    def test_sync_mode_measure(self):
        metric = AgentJudgeMetric(
            requirements=["Only milestone"],
            model=StaticJudgeModel(scores=[0.4]),
            async_mode=False,
        )
        score = metric.measure(_test_case())
        assert score == pytest.approx(0.4)
        assert metric.is_successful() is False

    def test_fill_pads_missing_and_drops_extra_scores(self):
        metric = AgentJudgeMetric(
            requirements=["a", "b", "c"], model=StaticJudgeModel()
        )
        metric._fill(
            _test_case(),
            ["a", "b", "c"],
            AgentJudgment(scores=[0.8], reason="short"),
        )
        assert metric.milestone_scores == [0.8, 0.0, 0.0]
        assert metric.score == pytest.approx(0.8 / 3)

    def test_align_scores_clamps_and_pads(self):
        aligned = AgentJudgeMetric._align_scores([2.0, -1.0], 3)
        assert aligned == [1.0, 0.0, 0.0]
        assert AgentJudgeMetric._align_scores(None, 2) == [0.0, 0.0]

    def test_fill_strict_mode_zeroes_below_threshold(self):
        metric = AgentJudgeMetric(
            requirements=["a", "b"],
            model=StaticJudgeModel(),
            strict_mode=True,
        )
        metric._fill(
            _test_case(),
            ["a", "b"],
            AgentJudgment(scores=[1.0, 0.5], reason="partial"),
        )
        assert metric.score == 0
        assert metric.is_successful() is False
        assert metric.reason == "partial"

    def test_judgment_prompt_contains_evidence(self):
        prompt = AgentJudgeMetric._build_judgment_prompt(
            _test_case(), ["Get the weather"]
        )
        assert "Fetch the weather and summarise it" in prompt
        assert "Get the weather" in prompt
        assert "get_weather" in prompt
        assert "Paris" in prompt
        assert '"scores"' in prompt

    def test_milestones_prompt_contains_task(self):
        prompt = AgentJudgeMetric._build_milestones_prompt(_test_case())
        assert "Fetch the weather and summarise it" in prompt
        assert '"milestones"' in prompt

    def test_trajectory_formats_and_truncates_tool_output(self):
        trajectory = AgentJudgeMetric._trajectory(_test_case())
        assert "It is raining in Paris." in trajectory
        assert "get_weather" in trajectory
        assert '"city": "Paris"' in trajectory
        long_case = LLMTestCase(
            input="task",
            actual_output="out",
            tools_called=[
                ToolCall(name="dump", input_parameters={}, output="x" * 900)
            ],
        )
        assert "x" * 900 not in AgentJudgeMetric._trajectory(long_case)

    def test_normalize_requirements_strips_and_drops_blanks(self):
        assert AgentJudgeMetric._normalize_requirements(
            ["  a ", "", "  ", "b"]
        ) == ["a", "b"]

    def test_empty_requirements_raise(self):
        with pytest.raises(ValueError):
            AgentJudgeMetric(requirements=["   "], model=StaticJudgeModel())

    def test_empty_llm_milestones_raise(self):
        with pytest.raises(ValueError):
            AgentJudgeMetric._normalize_milestones(["", "  "])

    def test_missing_actual_output_raises(self):
        metric = AgentJudgeMetric(requirements=["a"], model=StaticJudgeModel())
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(
                LLMTestCase(input="Fetch the weather", actual_output=None)
            )

    def test_exported_from_community_package(self):
        assert AgentJudgeMetric is AgentJudgeMetricDirect
        from deepeval.metrics import community

        assert "AgentJudgeMetric" in community.__all__
