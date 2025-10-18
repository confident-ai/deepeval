from typing import Optional, Dict, Any
import asyncio
from time import monotonic


class TraceTestingManager:
    test_name: Optional[str] = None
    test_dict: Optional[Dict[str, Any]] = None

    async def wait_for_test_dict(
        self, timeout: float = 10.0, poll_interval: float = 0.05
    ) -> Dict[str, Any]:
        deadline = monotonic() + timeout
        while self.test_dict is None and monotonic() < deadline:
            await asyncio.sleep(poll_interval)
        return self.test_dict or {}


trace_testing_manager = TraceTestingManager()
