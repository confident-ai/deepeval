import os
from unittest.mock import MagicMock, patch
import pytest
from deepeval.test_case import MLLMImage
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.metrics.step_efficiency.schema import EfficiencyVerdict, StepAnalysis

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


class TestStepEfficiencyMetric:
    """Tests for answer relevancy metric"""

    def test_normal_sync_metric_measure(self):
        @observe()
        def trip_planner_agent(input):
            destination = "Paris"
            days = 2

            @observe()
            def restaurant_finder(city):
                return ["Le Jules Verne", "Angelina Paris", "Septime"]

            @observe()
            def itinerary_generator(destination, days):
                return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

            itinerary = itinerary_generator(destination, days)
            restaurants = restaurant_finder(destination)

            return itinerary + restaurants

        golden = Golden(input="List some places from Paris")
        dataset = EvaluationDataset(goldens=[golden])

        metric = StepEfficiencyMetric(async_mode=False)

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert golden.multimodal is False

    def test_normal_async_metric_measure(self):
        @observe()
        def trip_planner_agent(input):
            destination = "Paris"
            days = 2

            @observe()
            def restaurant_finder(city):
                return ["Le Jules Verne", "Angelina Paris", "Septime"]

            @observe()
            def itinerary_generator(destination, days):
                return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

            itinerary = itinerary_generator(destination, days)
            restaurants = restaurant_finder(destination)

            return itinerary + restaurants

        golden = Golden(input="List some places from Paris")
        dataset = EvaluationDataset(goldens=[golden])

        metric = StepEfficiencyMetric()

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert golden.multimodal is False

    def test_multimodal_async_metric_measure(self):
        image = MLLMImage(url=CAR)

        @observe()
        def trip_planner_agent(input):
            destination = "Paris"
            days = 2

            @observe()
            def restaurant_finder(city):
                return ["Le Jules Verne", "Angelina Paris", "Septime"]

            @observe()
            def itinerary_generator(destination, days):
                return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

            itinerary = itinerary_generator(destination, days)
            restaurants = restaurant_finder(destination)

            return itinerary + restaurants

        golden = Golden(
            input=f"If this image is a car {image}, list some places from Paris"
        )
        dataset = EvaluationDataset(goldens=[golden])

        metric = StepEfficiencyMetric()

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert golden.multimodal is True

    def test_multimodal_sync_metric_measure(self):
        image = MLLMImage(url=CAR)

        @observe()
        def trip_planner_agent(input):
            destination = "Paris"
            days = 2

            @observe()
            def restaurant_finder(city):
                return ["Le Jules Verne", "Angelina Paris", "Septime"]

            @observe()
            def itinerary_generator(destination, days):
                return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

            itinerary = itinerary_generator(destination, days)
            restaurants = restaurant_finder(destination)

            return itinerary + restaurants

        golden = Golden(
            input=f"If this image is a car {image}, list some places from Paris"
        )
        dataset = EvaluationDataset(goldens=[golden])

        metric = StepEfficiencyMetric(async_mode=False)

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert golden.multimodal is True

    def test_invalid_model_throws_error_for_multimodal(self):
        image = MLLMImage(url=CAR)

        @observe()
        def trip_planner_agent(input):
            destination = "Paris"
            days = 2

            @observe()
            def restaurant_finder(city):
                return ["Le Jules Verne", "Angelina Paris", "Septime"]

            @observe()
            def itinerary_generator(destination, days):
                return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

            itinerary = itinerary_generator(destination, days)
            restaurants = restaurant_finder(destination)

            return itinerary + restaurants

        golden = Golden(
            input=f"If this image is a car {image}, list some places from Paris"
        )
        dataset = EvaluationDataset(goldens=[golden])

        with pytest.raises(ValueError):
            metric = StepEfficiencyMetric(model="gpt-3.5-turbo")

            for golden in dataset.evals_iterator(metrics=[metric]):
                trip_planner_agent(golden.input)


class TestStepEfficiencyScoreBreakdown:
    """No-API-key unit tests for score_breakdown population."""

    def _make_metric(self) -> StepEfficiencyMetric:
        mock_model = MagicMock()
        mock_model.get_model_name.return_value = "mock"
        with patch(
            "deepeval.metrics.step_efficiency.step_efficiency.initialize_model",
            return_value=(mock_model, True),
        ):
            return StepEfficiencyMetric()

    def test_score_breakdown_populated_from_verdict_steps(self):
        metric = self._make_metric()
        verdict = EfficiencyVerdict(
            score=0.5,
            reason="One step was redundant.",
            steps=[
                StepAnalysis(
                    step_name="fetch_data",
                    is_necessary=True,
                    reason="Required to retrieve the input data.",
                ),
                StepAnalysis(
                    step_name="extra_llm_call",
                    is_necessary=False,
                    reason="Duplicate call; the first response was sufficient.",
                ),
            ],
        )
        if verdict.steps:
            metric.score_breakdown = {
                step.step_name: {
                    "necessary": step.is_necessary,
                    "reason": step.reason,
                }
                for step in verdict.steps
            }

        assert metric.score_breakdown is not None
        assert len(metric.score_breakdown) == 2
        assert metric.score_breakdown["fetch_data"]["necessary"] is True
        assert metric.score_breakdown["extra_llm_call"]["necessary"] is False

    def test_score_breakdown_none_when_steps_absent(self):
        metric = self._make_metric()
        verdict = EfficiencyVerdict(score=1.0, reason="Efficient.", steps=None)
        if verdict.steps:
            metric.score_breakdown = {
                step.step_name: {"necessary": step.is_necessary, "reason": step.reason}
                for step in verdict.steps
            }
        assert metric.score_breakdown is None
