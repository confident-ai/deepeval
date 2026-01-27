import os
import pytest
from deepeval.test_case import MLLMImage
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(current_dir, "images/car.png")


class TestTaskCompletionMetric:
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

        metric = TaskCompletionMetric(async_mode=False)

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert metric.score is not None
        assert metric.reason is not None
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

        metric = TaskCompletionMetric()

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert metric.score is not None
        assert metric.reason is not None
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

        metric = TaskCompletionMetric()

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert metric.score is not None
        assert metric.reason is not None
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

        metric = TaskCompletionMetric(async_mode=False)

        for golden in dataset.evals_iterator(metrics=[metric]):
            trip_planner_agent(golden.input)

        assert metric.score is not None
        assert metric.reason is not None
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
            metric = TaskCompletionMetric(model="gpt-3.5-turbo")

            for golden in dataset.evals_iterator(metrics=[metric]):
                trip_planner_agent(golden.input)
