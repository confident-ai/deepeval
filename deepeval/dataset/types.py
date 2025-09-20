import asyncio

from typing import Any
from deepeval.dataset.utils import coerce_to_task


class EvaluationTasks:

    def __init__(self):
        self._tasks: list[asyncio.Future] = []

    def append(self, obj: Any):
        self._tasks.append(coerce_to_task(obj))

    def get_tasks(self) -> list[asyncio.Future]:
        return list(self._tasks)

    def num_tasks(self):
        return len(self._tasks)

    def clear_tasks(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks.clear()
