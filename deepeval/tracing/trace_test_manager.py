import asyncio
import logging
import traceback
from time import monotonic
from typing import Optional, Dict, Any

from deepeval.config.settings import get_settings
from deepeval.errors import DeepEvalError

logger = logging.getLogger(__name__)


class TraceTestingManager:
    test_name: Optional[str] = None
    _test_dict: Optional[Dict[str, Any]] = None

    @property
    def test_dict(self) -> Optional[Dict[str, Any]]:
        return self._test_dict

    @test_dict.setter
    def test_dict(self, value: Optional[Dict[str, Any]]) -> None:
        # Accept None to clear or dict only.
        # Anything else is a bug we want to find fast.
        if value is None or isinstance(value, dict):
            self._test_dict = value
            return

        # Bad type: log an error and if DEEPEVAL_LOG_STACK_TRACES
        # then capture stack to pinpoint where it came from
        if get_settings().DEEPEVAL_LOG_STACK_TRACES:
            stack = "".join(traceback.format_stack(limit=25))
            logger.error(
                "TraceTestingManager.test_dict expected dict|None, got %s. "
                "Offender stack:\n%s",
                type(value).__name__,
                stack,
            )
        else:
            logger.error(
                "TraceTestingManager.test_dict expected dict|None, got %s. ",
                type(value).__name__,
            )

        raise DeepEvalError(
            f"TraceTestingManager.test_dict expected dict|None, got {type(value).__name__}"
        )

    async def wait_for_test_dict(
        self,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        *,
        raise_on_timeout: bool = True,
    ) -> Dict[str, Any]:
        settings = get_settings()
        if timeout is None:
            timeout = max(
                10.0,
                float(settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS)
                + float(settings.DEEPEVAL_TASK_GATHER_BUFFER_SECONDS),
            )

        deadline = monotonic() + timeout
        while self.test_dict is None and monotonic() < deadline:
            await asyncio.sleep(poll_interval)

        if self.test_dict is None:
            if settings.DEEPEVAL_LOG_STACK_TRACES:
                stack = "".join(traceback.format_stack(limit=25))
                logger.error(
                    "TraceTestingManager.wait_for_test_dict timed out after %.2fs "
                    "(test_name=%r). Offender stack follows:\n%s",
                    timeout,
                    self.test_name,
                    stack,
                )
            else:
                logger.error(
                    "TraceTestingManager.wait_for_test_dict timed out after %.2fs "
                    "(test_name=%r).",
                    timeout,
                    self.test_name,
                )

            if raise_on_timeout:
                raise DeepEvalError(
                    f"Timed out waiting for trace dict (test_name={self.test_name!r}, timeout={timeout}s)."
                )
            return {}

        return self.test_dict

    def is_active(self) -> bool:
        """Return True iff a test case is currently active."""
        return self.test_name is not None

    def set_test_dict(self, test_dict: Optional[Dict[str, Any]] = None) -> None:
        """Publish (or clear with None) the captured trace dict for the test harness."""
        self.test_dict = test_dict  # only assign to property for validation

    def set_case(self, name: str) -> None:
        """Enter test mode for a specific expected JSON path/name."""
        self.test_name = name
        self.set_test_dict(None)

    def clear_payload(self) -> None:
        """Clear only the published dict, keep test mode active."""
        self.set_test_dict(None)

    def disable(self) -> None:
        """Exit test mode entirely (clear both name and payload)."""
        self.test_name = None
        self.set_test_dict(None)


trace_testing_manager = TraceTestingManager()
