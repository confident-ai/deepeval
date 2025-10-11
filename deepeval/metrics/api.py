from typing import Optional, Set, Any, Dict, List, Union
import threading
import asyncio
import queue
import atexit
from time import perf_counter
from enum import Enum
from pydantic import Field
from rich.console import Console

from deepeval.confident.api import Api, HttpMethods, Endpoints, is_confident
from deepeval.constants import (
    CONFIDENT_METRIC_LOGGING_FLUSH,
    CONFIDENT_METRIC_LOGGING_VERBOSE,
)
from deepeval.metrics.base_metric import BaseConversationalMetric, BaseMetric
from deepeval.test_case.conversational_test_case import ConversationalTestCase
from deepeval.test_case.llm_test_case import LLMTestCase
from deepeval.test_case.api import create_api_test_case
from deepeval.test_run.api import LLMApiTestCase, ConversationalApiTestCase
from deepeval.tracing.api import MetricData
from deepeval.config.settings import get_settings


class MetricWorkerStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"


class ApiMetricData(MetricData):
    llm_test_case: Optional[LLMApiTestCase] = Field(None, alias="llmTestCase")
    conversational_test_case: Optional[ConversationalApiTestCase] = Field(
        None, alias="conversationalTestCase"
    )


class MetricDataManager:
    """Manager for posting metric data asynchronously in background thread."""

    def __init__(self):
        settings = get_settings()
        # Initialize queue and worker thread for metric posting
        self._metric_queue = queue.Queue()
        self._worker_thread = None
        self._min_interval = 0.2  # Minimum time between API calls (seconds)
        self._last_post_time = 0
        self._in_flight_tasks: Set[asyncio.Task[Any]] = set()
        self._flush_enabled = bool(settings.CONFIDENT_METRIC_LOGGING_FLUSH)
        self._daemon = not self._flush_enabled
        self._thread_lock = threading.Lock()
        self.metric_logging_enabled = bool(
            settings.CONFIDENT_METRIC_LOGGING_ENABLED
        )

        # Register an exit handler to warn about unprocessed metrics
        atexit.register(self._warn_on_exit)

    def post_metric_if_enabled(
        self,
        metric: Union[BaseMetric, BaseConversationalMetric],
        test_case: Optional[Union[LLMTestCase, ConversationalTestCase]] = None,
    ):
        """Post metric data asynchronously in a background thread."""
        if not self.metric_logging_enabled or not is_confident():
            return

        from deepeval.evaluate.utils import create_metric_data

        metric_data = create_metric_data(metric)
        api_metric_data = ApiMetricData(
            **metric_data.model_dump(by_alias=True, exclude_none=True)
        )

        if isinstance(test_case, LLMTestCase):
            api_metric_data.llm_test_case = create_api_test_case(test_case)
        elif isinstance(test_case, ConversationalTestCase):
            api_metric_data.conversational_test_case = create_api_test_case(
                test_case
            )

        self._ensure_worker_thread_running()
        self._metric_queue.put(api_metric_data)

    def _warn_on_exit(self):
        """Warn if there are unprocessed metrics on exit."""
        queue_size = self._metric_queue.qsize()
        in_flight = len(self._in_flight_tasks)
        remaining_tasks = queue_size + in_flight

        if not self._flush_enabled and remaining_tasks > 0:
            self._print_metric_data_status(
                metric_worker_status=MetricWorkerStatus.WARNING,
                message=f"Exiting with {queue_size + in_flight} abandoned metric(s).",
                description=f"Set {CONFIDENT_METRIC_LOGGING_FLUSH}=1 as an environment variable to flush remaining metrics to Confident AI.",
            )

    def _ensure_worker_thread_running(self):
        """Ensure the background worker thread is running."""
        with self._thread_lock:
            if (
                self._worker_thread is None
                or not self._worker_thread.is_alive()
            ):
                self._worker_thread = threading.Thread(
                    target=self._process_metric_queue,
                    daemon=self._daemon,
                )
                self._worker_thread.start()

    def _print_metric_data_status(
        self,
        metric_worker_status: MetricWorkerStatus,
        message: str,
        description: Optional[str] = None,
    ):
        """Print metric data worker status messages."""
        if getattr(get_settings(), CONFIDENT_METRIC_LOGGING_VERBOSE, False):
            console = Console()
            message_prefix = "[dim][Confident AI Metric Data Log][/dim]"
            if metric_worker_status == MetricWorkerStatus.SUCCESS:
                message = f"[green]{message}[/green]"
            elif metric_worker_status == MetricWorkerStatus.FAILURE:
                message = f"[red]{message}[/red]"
            elif metric_worker_status == MetricWorkerStatus.WARNING:
                message = f"[yellow]{message}[/yellow]"

            if bool(CONFIDENT_METRIC_LOGGING_VERBOSE):
                if description:
                    message += f": {description}"

                console.print(
                    message_prefix,
                    message,
                    f"\nTo disable dev logging, set {CONFIDENT_METRIC_LOGGING_VERBOSE}=0 as an environment variable.",
                )

    def _process_metric_queue(self):
        """Worker thread function that processes the metric queue."""
        import threading

        main_thr = threading.main_thread()

        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Buffer for payloads that need to be sent after main exits
        remaining_metric_request_bodies: List[Dict[str, Any]] = []

        async def _a_send_metric(metric_data: ApiMetricData):
            nonlocal remaining_metric_request_bodies
            try:
                # Build API object & payload
                try:
                    body = metric_data.model_dump(
                        by_alias=True,
                        exclude_none=True,
                    )
                except AttributeError:
                    # Pydantic version below 2.0
                    body = metric_data.dict(by_alias=True, exclude_none=True)

                # If the main thread is still alive, send now
                if main_thr.is_alive():
                    api = Api()
                    _, _ = await api.a_send_request(
                        method=HttpMethods.POST,
                        endpoint=Endpoints.METRIC_DATA_ENDPOINT,
                        body=body,
                    )
                    queue_size = self._metric_queue.qsize()
                    in_flight = len(self._in_flight_tasks)
                    status = f"({queue_size} metric{'s' if queue_size!=1 else ''} remaining in queue, {in_flight} in flight)"
                    self._print_metric_data_status(
                        metric_worker_status=MetricWorkerStatus.SUCCESS,
                        message=f"Successfully posted metric data {status}",
                    )
                elif self._flush_enabled:
                    # Main thread gone → to be flushed
                    remaining_metric_request_bodies.append(body)

            except Exception as e:
                queue_size = self._metric_queue.qsize()
                in_flight = len(self._in_flight_tasks)
                status = f"({queue_size} metric{'s' if queue_size!=1 else ''} remaining in queue, {in_flight} in flight)"
                self._print_metric_data_status(
                    metric_worker_status=MetricWorkerStatus.FAILURE,
                    message=f"Error posting metric data {status}",
                    description=str(e),
                )
            finally:
                task = asyncio.current_task()
                if task:
                    self._in_flight_tasks.discard(task)

        async def async_worker():
            # Continue while user code is running or work remains
            while (
                main_thr.is_alive()
                or not self._metric_queue.empty()
                or self._in_flight_tasks
            ):
                try:
                    metric_data = self._metric_queue.get(
                        block=True, timeout=1.0
                    )

                    # Rate-limit
                    now = perf_counter()
                    elapsed = now - self._last_post_time
                    if elapsed < self._min_interval:
                        await asyncio.sleep(self._min_interval - elapsed)
                    self._last_post_time = perf_counter()

                    # Schedule async send
                    task = asyncio.create_task(_a_send_metric(metric_data))
                    self._in_flight_tasks.add(task)
                    self._metric_queue.task_done()

                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                except Exception as e:
                    self._print_metric_data_status(
                        message="Error in metric worker",
                        metric_worker_status=MetricWorkerStatus.FAILURE,
                        description=str(e),
                    )
                    await asyncio.sleep(1.0)

        try:
            loop.run_until_complete(async_worker())
        finally:
            # Drain any pending tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._flush_metrics(remaining_metric_request_bodies)
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _flush_metrics(
        self, remaining_metric_request_bodies: List[Dict[str, Any]]
    ):
        """Flush remaining metrics synchronously."""
        if not remaining_metric_request_bodies:
            return

        self._print_metric_data_status(
            MetricWorkerStatus.WARNING,
            message=f"Flushing {len(remaining_metric_request_bodies)} remaining metric(s)",
        )

        for body in remaining_metric_request_bodies:
            try:
                api = Api()
                _, link = api.send_request(
                    method=HttpMethods.POST,
                    endpoint=Endpoints.METRIC_DATA_ENDPOINT,
                    body=body,
                )
                qs = self._metric_queue.qsize()
                self._print_metric_data_status(
                    metric_worker_status=MetricWorkerStatus.SUCCESS,
                    message=f"Successfully posted metric data ({qs} metrics remaining in queue, 1 in flight)",
                    description=link,
                )
            except Exception as e:
                qs = self._metric_queue.qsize()
                self._print_metric_data_status(
                    metric_worker_status=MetricWorkerStatus.FAILURE,
                    message="Error flushing remaining metric(s)",
                    description=str(e),
                )


# Global metric manager instance
metric_data_manager = MetricDataManager()
