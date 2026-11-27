"""Regression tests for the ``Synthesizer(async_mode=True)`` event-loop handling.

Background
----------
The synchronous ``generate_*`` wrappers used to drive their ``async`` pipeline
with::

    loop = get_or_create_event_loop()
    loop.run_until_complete(self.a_generate_...(...))

``get_or_create_event_loop`` calls ``nest_asyncio.apply()`` whenever it is
invoked while an event loop is *already running* (Jupyter/IPython, an async web
server, or any ``async def``). ``nest_asyncio`` monkeypatches the running loop so
that ``run_until_complete`` can be called re-entrantly on it. That global patch
of asyncio's scheduler is a well-known source of scheduling deadlocks on
Python 3.12 and is entirely avoidable.

The fix routes every synchronous wrapper through :func:`deepeval.utils.run_async`,
which uses a fresh ``asyncio.run`` loop from a sync context and a dedicated
worker thread (with its own fresh loop) when a loop is already running — so it
never re-enters the caller's loop and never applies ``nest_asyncio``.

These tests assert:
  * ``async_mode=True`` completes from a plain synchronous caller, and
  * ``async_mode=True`` completes when called from *within* a running event loop
    **without** monkeypatching asyncio via ``nest_asyncio.apply()``.

The second test fails on the pre-fix code (``nest_asyncio.apply`` is called) and
passes after the fix. Both run under a hard timeout so a regression surfaces as a
failure rather than a hung test run.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from unittest.mock import patch

import pytest

import deepeval.utils as deepeval_utils
from deepeval.dataset import Golden
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import (
    EvolutionConfig,
    FiltrationConfig,
    StylingConfig,
)
from deepeval.synthesizer.schema import (
    InputFeedback,
    PromptStyling,
    Response,
    RewrittenInput,
    SyntheticData,
    SyntheticDataList,
)

_TIMEOUT_SECONDS = 60


class _FakeLLM(DeepEvalBaseLLM):
    """A minimal custom (non-native) model that returns canned structured
    objects, so the full synthesizer pipeline runs without any network I/O."""

    def __init__(self):
        self.name = "fake-async-model"

    def load_model(self, *args, **kwargs):
        return self

    def get_model_name(self, *args, **kwargs):
        return self.name

    def _answer(self, schema):
        if schema is None:
            return "canned response"
        if schema is SyntheticDataList:
            return SyntheticDataList(
                data=[
                    SyntheticData(input="synthetic input", used_source_files=[])
                ]
            )
        if schema is SyntheticData:
            return SyntheticData(input="styled input", used_source_files=[])
        if schema is InputFeedback:
            return InputFeedback(score=1.0, feedback="great")
        if schema is RewrittenInput:
            return RewrittenInput(rewritten_input="rewritten")
        if schema is Response:
            return Response(response="expected output")
        if schema is PromptStyling:
            return PromptStyling(scenario="s", task="t", input_format="f")
        return schema()

    def generate(self, prompt, schema=None, *args, **kwargs):
        return self._answer(schema)

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        # Force interleaving between concurrent tasks.
        await asyncio.sleep(0)
        return self._answer(schema)


def _make_synthesizer(max_concurrent=2):
    model = _FakeLLM()
    return Synthesizer(
        model=model,
        styling_config=StylingConfig(
            input_format="question",
            expected_output_format="answer",
            task="qa",
            scenario="user asks a question",
        ),
        evolution_config=EvolutionConfig(
            evolutions={Evolution.CONSTRAINED: 1.0}, num_evolutions=1
        ),
        filtration_config=FiltrationConfig(
            synthetic_input_quality_threshold=0.7,
            max_quality_retries=1,
            critic_model=model,
        ),
        async_mode=True,
        max_concurrent=max_concurrent,
        cost_tracking=False,
    )


def _build_goldens(n):
    return [
        Golden(
            input=f"question {i}",
            expected_output=f"answer {i}",
            context=[f"supporting quote {i}"],
            retrieval_context=[f"supporting quote {i}"],
        )
        for i in range(n)
    ]


def _run_with_timeout(fn, timeout=_TIMEOUT_SECONDS):
    """Run ``fn`` on a worker thread, failing (rather than hanging) on deadlock."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            pytest.fail(
                f"generate_goldens_from_goldens(async_mode=True) hung for "
                f">{timeout}s (event-loop deadlock)"
            )


def test_generate_goldens_async_mode_completes_from_plain_sync_caller():
    """The default async path completes from an ordinary synchronous caller."""
    synth = _make_synthesizer()
    goldens = _build_goldens(4)

    result = _run_with_timeout(
        lambda: synth.generate_goldens_from_goldens(
            goldens=goldens,
            max_goldens_per_golden=1,
            include_expected_output=True,
        )
    )

    assert len(result) == 4
    assert all(isinstance(g, Golden) for g in result)


def test_generate_goldens_async_mode_completes_inside_running_loop_without_nest_asyncio():
    """Calling the sync wrapper from within a running event loop must complete
    and must NOT globally monkeypatch asyncio via ``nest_asyncio.apply()``.

    Pre-fix, ``get_or_create_event_loop`` calls ``nest_asyncio.apply()`` on the
    running loop and re-enters it with ``run_until_complete``; this assertion
    fails there. After the fix the pipeline runs on an isolated worker-thread
    loop and ``nest_asyncio.apply`` is never called.
    """
    synth = _make_synthesizer()
    goldens = _build_goldens(4)

    async def _driver():
        # We are inside a running event loop here; calling the *synchronous*
        # wrapper exercises run_async's "loop already running" branch.
        return synth.generate_goldens_from_goldens(
            goldens=goldens,
            max_goldens_per_golden=1,
            include_expected_output=True,
        )

    real_apply = deepeval_utils.nest_asyncio.apply
    with patch.object(
        deepeval_utils.nest_asyncio, "apply", wraps=real_apply
    ) as apply_spy:
        result = _run_with_timeout(lambda: asyncio.run(_driver()))

    apply_spy.assert_not_called()
    assert len(result) == 4
    assert all(isinstance(g, Golden) for g in result)
