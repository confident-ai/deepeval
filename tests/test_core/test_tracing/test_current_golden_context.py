import asyncio
import pytest


from deepeval.contextvars import (
    set_current_golden,
    get_current_golden,
    reset_current_golden,
)


class GoldenStub:
    def __init__(self, name, expected_output=None):
        self.name = name
        self.expected_output = expected_output


@pytest.mark.asyncio
async def test_current_golden_is_task_local_and_resets():
    # set in the outer task
    token_outer = set_current_golden(
        GoldenStub("outer", expected_output="E_OUTER")
    )
    try:
        assert get_current_golden().name == "outer"

        # spawn a nested task that sets a different golden
        async def child():
            tok_inner = set_current_golden(
                GoldenStub("inner", expected_output="E_INNER")
            )
            try:
                # inside child: see inner
                g = get_current_golden()
                assert g is not None and g.name == "inner"
                await asyncio.sleep(
                    0
                )  # yield to event loop to ensure context stability
                # still inner after await
                assert get_current_golden().name == "inner"
            finally:
                reset_current_golden(tok_inner)
                # after reset in child: child sees parent's value - outer
                assert get_current_golden().name == "outer"

        await child()

        # Back in parent: still outer, unaffected by child
        assert get_current_golden().name == "outer"
    finally:
        reset_current_golden(token_outer)
        assert get_current_golden() is None


@pytest.mark.asyncio
async def test_task_creation_captures_context():
    # contextVars are captured at task creation time.
    token = set_current_golden(GoldenStub("captured", expected_output="E_CAP"))
    try:

        async def probe():
            # The context should be the one captured when task was created
            g = get_current_golden()
            assert g is not None and g.name == "captured"

        t = asyncio.create_task(probe())
        # mutate after creating the task, this should not affect the already created task
        reset_current_golden(token)
        assert get_current_golden() is None

        await t
    finally:
        # ensure clean end state even if assertions above change in the future
        try:
            reset_current_golden(token)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_each_task_captures_value_at_creation_time():
    t0 = set_current_golden(GoldenStub("G1"))
    try:

        async def read_name():
            return get_current_golden().name

        # task1 captures G1
        task1 = asyncio.create_task(read_name())

        # switch to G2, then create task2
        reset_current_golden(t0)
        t1 = set_current_golden(GoldenStub("G2"))
        try:
            task2 = asyncio.create_task(read_name())
            n1, n2 = await asyncio.gather(task1, task2)
            assert n1 == "G1"
            assert n2 == "G2"
        finally:
            reset_current_golden(t1)
    finally:
        try:
            reset_current_golden(t0)
        except Exception:
            pass


def test_set_none_restores_previous_on_reset():
    t0 = set_current_golden(GoldenStub("prev"))
    try:
        t1 = set_current_golden(None)
        try:
            assert get_current_golden() is None
        finally:
            reset_current_golden(t1)
        assert get_current_golden().name == "prev"
    finally:
        reset_current_golden(t0)
        assert get_current_golden() is None


@pytest.mark.asyncio
async def test_gather_sees_per_task_snapshots():
    async def run_with(name):
        token = set_current_golden(GoldenStub(name))
        try:
            await asyncio.sleep(0)
            return get_current_golden().name
        finally:
            reset_current_golden(token)

    n1, n2 = await asyncio.gather(run_with("A"), run_with("B"))
    assert {n1, n2} == {"A", "B"}
