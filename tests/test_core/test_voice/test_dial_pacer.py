import asyncio
import time

import pytest

from deepeval.voice.connectors.utils import DialPacer


async def _dial_times(pacer: DialPacer, calls: int, interval_s: float):
    times = []

    async def dial():
        await pacer.wait(interval_s)
        times.append(time.monotonic())

    await asyncio.gather(*(dial() for _ in range(calls)))
    return sorted(times)


@pytest.mark.asyncio
async def test_parallel_dials_are_spaced_by_the_interval():
    times = await _dial_times(DialPacer(), calls=3, interval_s=0.05)

    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert all(gap >= 0.045 for gap in gaps)


@pytest.mark.asyncio
async def test_the_first_dial_goes_out_immediately():
    started = time.monotonic()

    await DialPacer().wait(5.0)

    assert time.monotonic() - started < 0.05


def test_a_pacer_works_across_separate_event_loops():
    pacer = DialPacer()

    asyncio.run(pacer.wait(0.01))
    asyncio.run(pacer.wait(0.01))
