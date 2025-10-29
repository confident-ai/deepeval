import pytest
from unittest.mock import AsyncMock, Mock, patch
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics.task_completion.schema import (
    TaskAndOutcome,
    TaskCompletionVerdict,
)


class TestTaskCompletionTaskAssignment:
    """Test that _is_task_provided correctly controls task assignment in measure and a_measure."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns structured responses."""
        model = Mock()
        model.get_model_name.return_value = "mock-model"
        return model

    @pytest.fixture
    def test_case(self):
        """Create a test case with trace data."""
        return LLMTestCase(
            input="Test input",
            actual_output="Test output",
            tools_called=[ToolCall(name="tool1")],
            _trace_dict={"some": "trace_data"},
        )

    def test_measure_assigns_task_when_not_provided(
        self, mock_model, test_case
    ):
        """Test that measure assigns extracted task when task is not provided in __init__."""
        # Patch initialize_model to return our mock
        with patch(
            "deepeval.metrics.task_completion.task_completion.initialize_model",
            return_value=(mock_model, True),
        ):
            # Create metric without task
            metric = TaskCompletionMetric(
                model=mock_model, async_mode=False, threshold=0.5
            )

            # Verify _is_task_provided is False
            assert metric._is_task_provided is False
            assert metric.task is None

            # Run measure
            with patch.object(
                metric,
                "_extract_task_and_outcome",
                return_value=("Extracted task", "Some outcome"),
            ):
                with patch.object(
                    metric,
                    "_generate_verdicts",
                    return_value=(0.8, "Test reason"),
                ):
                    metric.measure(test_case, _show_indicator=False)

            # Verify task was assigned from extraction
            assert metric.task == "Extracted task"

    def test_measure_does_not_overwrite_provided_task(
        self, mock_model, test_case
    ):
        """Test that measure does NOT overwrite task when it was provided in __init__."""
        provided_task = "Original provided task"

        # Patch initialize_model to return our mock
        with patch(
            "deepeval.metrics.task_completion.task_completion.initialize_model",
            return_value=(mock_model, True),
        ):
            # Create metric with task
            metric = TaskCompletionMetric(
                task=provided_task,
                model=mock_model,
                async_mode=False,
                threshold=0.5,
            )

            # Verify _is_task_provided is True
            assert metric._is_task_provided is True
            assert metric.task == provided_task

            # Run measure - should NOT overwrite task even if extraction returns different task
            with patch.object(
                metric,
                "_extract_task_and_outcome",
                return_value=("Different extracted task", "Some outcome"),
            ):
                with patch.object(
                    metric,
                    "_generate_verdicts",
                    return_value=(0.8, "Test reason"),
                ):
                    metric.measure(test_case, _show_indicator=False)

            # Verify task was NOT overwritten
            assert metric.task == provided_task

    @pytest.mark.asyncio
    async def test_a_measure_assigns_task_when_not_provided(
        self, mock_model, test_case
    ):
        """Test that a_measure assigns extracted task when task is not provided in __init__."""
        # Patch initialize_model to return our mock
        with patch(
            "deepeval.metrics.task_completion.task_completion.initialize_model",
            return_value=(mock_model, True),
        ):
            # Create metric without task
            metric = TaskCompletionMetric(model=mock_model, threshold=0.5)

            # Verify _is_task_provided is False
            assert metric._is_task_provided is False
            assert metric.task is None

            # Run a_measure
            with patch.object(
                metric,
                "_a_extract_task_and_outcome",
                new_callable=AsyncMock,
                return_value=("Async extracted task", "Some outcome"),
            ):
                with patch.object(
                    metric,
                    "_a_generate_verdicts",
                    new_callable=AsyncMock,
                    return_value=(0.8, "Test reason"),
                ):
                    await metric.a_measure(test_case, _show_indicator=False)

            # Verify task was assigned from extraction
            assert metric.task == "Async extracted task"

    @pytest.mark.asyncio
    async def test_a_measure_does_not_overwrite_provided_task(
        self, mock_model, test_case
    ):
        """Test that a_measure does NOT overwrite task when it was provided in __init__."""
        provided_task = "Original provided task for async"

        # Patch initialize_model to return our mock
        with patch(
            "deepeval.metrics.task_completion.task_completion.initialize_model",
            return_value=(mock_model, True),
        ):
            # Create metric with task
            metric = TaskCompletionMetric(
                task=provided_task, model=mock_model, threshold=0.5
            )

            # Verify _is_task_provided is True
            assert metric._is_task_provided is True
            assert metric.task == provided_task

            # Run a_measure - should NOT overwrite task even if extraction returns different task
            with patch.object(
                metric,
                "_a_extract_task_and_outcome",
                new_callable=AsyncMock,
                return_value=("Different async extracted task", "Some outcome"),
            ):
                with patch.object(
                    metric,
                    "_a_generate_verdicts",
                    new_callable=AsyncMock,
                    return_value=(0.8, "Test reason"),
                ):
                    await metric.a_measure(test_case, _show_indicator=False)

            # Verify task was NOT overwritten
            assert metric.task == provided_task

    def test_is_task_provided_flag_correctly_set(self):
        """Test that _is_task_provided flag is set correctly based on task parameter."""
        mock_model = Mock()
        mock_model.get_model_name.return_value = "mock-model"

        with patch(
            "deepeval.metrics.task_completion.task_completion.initialize_model",
            return_value=(mock_model, True),
        ):
            # When task is None
            metric_no_task = TaskCompletionMetric(task=None)
            assert metric_no_task._is_task_provided is False

            # When task is provided
            metric_with_task = TaskCompletionMetric(task="Some task")
            assert metric_with_task._is_task_provided is True

    @pytest.mark.asyncio
    async def test_both_methods_handle_task_consistently(
        self, mock_model, test_case
    ):
        """Test that both measure and a_measure handle task assignment consistently."""
        extracted_task = "Consistently extracted task"

        with patch(
            "deepeval.metrics.task_completion.task_completion.initialize_model",
            return_value=(mock_model, True),
        ):
            # Test with task not provided
            metric1 = TaskCompletionMetric(model=mock_model, async_mode=False)
            metric2 = TaskCompletionMetric(model=mock_model, async_mode=True)

            with patch.object(
                metric1,
                "_extract_task_and_outcome",
                return_value=(extracted_task, "outcome"),
            ):
                with patch.object(
                    metric1, "_generate_verdicts", return_value=(0.8, "reason")
                ):
                    metric1.measure(test_case, _show_indicator=False)

            with patch.object(
                metric2,
                "_a_extract_task_and_outcome",
                new_callable=AsyncMock,
                return_value=(extracted_task, "outcome"),
            ):
                with patch.object(
                    metric2,
                    "_a_generate_verdicts",
                    new_callable=AsyncMock,
                    return_value=(0.8, "reason"),
                ):
                    await metric2.a_measure(test_case, _show_indicator=False)

            # Both should have the same task assigned
            assert metric1.task == extracted_task
            assert metric2.task == extracted_task

            # Test with task provided
            provided_task = "Provided task"
            metric3 = TaskCompletionMetric(
                task=provided_task, model=mock_model, async_mode=False
            )
            metric4 = TaskCompletionMetric(
                task=provided_task, model=mock_model, async_mode=True
            )

            with patch.object(
                metric3,
                "_extract_task_and_outcome",
                return_value=("different", "outcome"),
            ):
                with patch.object(
                    metric3, "_generate_verdicts", return_value=(0.8, "reason")
                ):
                    metric3.measure(test_case, _show_indicator=False)

            with patch.object(
                metric4,
                "_a_extract_task_and_outcome",
                new_callable=AsyncMock,
                return_value=("different", "outcome"),
            ):
                with patch.object(
                    metric4,
                    "_a_generate_verdicts",
                    new_callable=AsyncMock,
                    return_value=(0.8, "reason"),
                ):
                    await metric4.a_measure(test_case, _show_indicator=False)

            # Both should retain the provided task
            assert metric3.task == provided_task
            assert metric4.task == provided_task
