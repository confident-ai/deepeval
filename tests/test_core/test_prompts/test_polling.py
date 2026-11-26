import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deepeval.prompt import Prompt


@pytest.mark.parametrize(
    ("version", "label", "hash_value", "branch"),
    [
        (None, None, None, "production"),
        ("1", None, None, None),
        (None, "release", None, None),
        (None, None, "commit-hash", None),
    ],
)
def test_polling_uses_the_registered_interval(
    version, label, hash_value, branch
):
    async def run():
        prompt = Prompt(alias="example")
        with patch(
            "deepeval.prompt.prompt.asyncio.sleep",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ) as sleep:
            await prompt.create_polling_task(
                version, label, hash_value, branch, refresh=60
            )
            tasks = [
                task
                for group in prompt._polling_tasks.values()
                for task in group.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(
                isinstance(result, asyncio.CancelledError) for result in results
            )
            sleep.assert_awaited_once_with(60)

    asyncio.run(run())


def test_distinct_branches_have_independent_refresh_tasks():
    async def run():
        prompt = Prompt(alias="example")
        with patch.object(prompt, "poll", new_callable=AsyncMock) as poll:
            await prompt.create_polling_task(
                None, None, None, "production", refresh=60
            )
            await prompt.create_polling_task(
                None, None, None, "staging", refresh=30
            )
            tasks = [
                task
                for group in prompt._polling_tasks.values()
                for task in group.values()
            ]
            await asyncio.gather(*tasks)
            assert poll.await_count == 2
            assert prompt._refresh_map["branch"] == {
                "production": 60,
                "staging": 30,
            }

    asyncio.run(run())
