"""Tests for the prompt_builder feature in Synthesizer.

prompt_builder is an optional callable that, when provided,
replaces SynthesizerTemplate.generate_synthetic_inputs at the
prompt-generation call sites.

Two un-tested behaviors:
1. Default path (prompt_builder=None): SynthesizerTemplate.generate_synthetic_inputs
   is called.
2. Custom path: the callable is invoked with the correct keyword arguments and
   SynthesizerTemplate.generate_synthetic_inputs is NOT called.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepeval.synthesizer.schema import SyntheticData

# ---------------------------------------------------------------------------
# Helpers – lightweight stand-ins so we never hit real LLMs
# ---------------------------------------------------------------------------

_INIT_MODEL_PATHS = [
    "deepeval.synthesizer.synthesizer.initialize_model",
    "deepeval.synthesizer.config.initialize_model",
]


def _make_synthesizer(**overrides):
    """Build a Synthesizer with a fake model, no real LLM calls."""
    from deepeval.synthesizer.synthesizer import Synthesizer

    fake_model = MagicMock()
    fake_model.get_model_name.return_value = "fake-model"
    fake_model.generate.return_value = ("fake", 0.0)
    fake_model.a_generate = AsyncMock(return_value=("fake", 0.0))

    defaults = dict(model=fake_model, async_mode=False)
    defaults.update(overrides)

    with patch(_INIT_MODEL_PATHS[0], return_value=(fake_model, True)), patch(
        _INIT_MODEL_PATHS[1], return_value=(fake_model, True)
    ):
        return Synthesizer(**defaults)


def _run_sync_generate(synth, context=None, max_goldens_per_context=1):
    """Call generate_goldens_from_contexts with progress-bar machinery stubbed."""
    if context is None:
        context = ["ctx sentence"]

    with patch(
        "deepeval.synthesizer.synthesizer.synthesizer_progress_context"
    ) as mock_ctx:
        progress_mock = MagicMock()
        mock_ctx.return_value.__enter__ = MagicMock(
            return_value=(progress_mock, 0)
        )
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "deepeval.synthesizer.synthesizer.add_pbar", return_value=0
        ), patch("deepeval.synthesizer.synthesizer.update_pbar"), patch(
            "deepeval.synthesizer.synthesizer.remove_pbars"
        ):
            return synth.generate_goldens_from_contexts(
                contexts=[context],
                include_expected_output=False,
                max_goldens_per_context=max_goldens_per_context,
                _send_data=False,
            )


def _stub_sync_internals(synth):
    """Attach lightweight stubs for all I/O helpers used by the sync loop."""
    dummy_input = SyntheticData(input="q1")
    synth._generate_inputs = MagicMock(return_value=[dummy_input])
    synth._rewrite_inputs = MagicMock(return_value=([dummy_input], [1.0]))
    synth._evolve_input = MagicMock(return_value=("evolved", ["Reasoning"]))
    return dummy_input


async def _run_async_generate(synth, context=None, max_goldens_per_context=1):
    """Call _a_generate_from_context directly with progress-bar stubs."""
    if context is None:
        context = ["ctx sentence"]

    goldens = []
    semaphore = asyncio.Semaphore(10)

    with patch(
        "deepeval.synthesizer.synthesizer.add_pbar", return_value=0
    ), patch("deepeval.synthesizer.synthesizer.update_pbar"), patch(
        "deepeval.synthesizer.synthesizer.remove_pbars"
    ):
        await synth._a_generate_from_context(
            semaphore=semaphore,
            context=context,
            goldens=goldens,
            include_expected_output=False,
            max_goldens_per_context=max_goldens_per_context,
            source_files=None,
            context_index=0,
            progress=None,
            pbar_id=None,
        )

    return goldens


def _stub_async_internals(synth):
    """Attach lightweight async stubs for all I/O helpers used by the async loop."""
    dummy_input = SyntheticData(input="q1")
    synth._a_generate_inputs = AsyncMock(return_value=[dummy_input])
    synth._a_rewrite_inputs = AsyncMock(return_value=([dummy_input], [1.0]))
    synth._a_evolve_input = AsyncMock(return_value=("evolved", ["Reasoning"]))
    return dummy_input


# ---------------------------------------------------------------------------
# Instantiation smoke tests
# ---------------------------------------------------------------------------


def test_prompt_builder_stored_on_instance():
    """prompt_builder passed at construction must be accessible as an instance attribute."""
    builder = lambda **kw: "custom prompt"
    synth = _make_synthesizer(prompt_builder=builder)
    assert synth.prompt_builder is builder


def test_prompt_builder_defaults_to_none():
    """When prompt_builder is omitted, the attribute must be None (default path active)."""
    synth = _make_synthesizer()
    assert synth.prompt_builder is None


# ---------------------------------------------------------------------------
# Sync path — default uses SynthesizerTemplate
# ---------------------------------------------------------------------------


def test_sync_default_calls_synthesizer_template():
    """Without a custom prompt_builder, the sync loop must delegate prompt
    construction to SynthesizerTemplate.generate_synthetic_inputs."""
    synth = _make_synthesizer()
    _stub_sync_internals(synth)

    with patch(
        "deepeval.synthesizer.synthesizer.SynthesizerTemplate.generate_synthetic_inputs",
        return_value="<default prompt>",
    ) as mock_template:
        goldens = _run_sync_generate(synth)

    mock_template.assert_called_once()
    assert len(goldens) == 1


# ---------------------------------------------------------------------------
# Sync path — custom builder is called with correct kwargs
# ---------------------------------------------------------------------------


def test_sync_custom_prompt_builder_called_with_correct_kwargs():
    """When prompt_builder is set, the sync loop must call it with the
    expected keyword arguments and must NOT fall through to
    SynthesizerTemplate.generate_synthetic_inputs."""
    custom_builder = MagicMock(return_value="<my prompt>")
    synth = _make_synthesizer(prompt_builder=custom_builder)
    _stub_sync_internals(synth)

    context = ["ctx sentence"]

    with patch(
        "deepeval.synthesizer.synthesizer.SynthesizerTemplate.generate_synthetic_inputs",
    ) as mock_template:
        goldens = _run_sync_generate(synth, context=context)

    custom_builder.assert_called_once_with(
        context=context,
        max_goldens_per_context=1,
        scenario=synth.styling_config.scenario,
        task=synth.styling_config.task,
        input_format=synth.styling_config.input_format,
    )
    mock_template.assert_not_called()
    assert len(goldens) == 1


# ---------------------------------------------------------------------------
# Async path — mirrors sync tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_default_calls_synthesizer_template():
    """Without a custom prompt_builder, the async loop must delegate prompt
    construction to SynthesizerTemplate.generate_synthetic_inputs."""
    synth = _make_synthesizer()
    _stub_async_internals(synth)

    with patch(
        "deepeval.synthesizer.synthesizer.SynthesizerTemplate.generate_synthetic_inputs",
        return_value="<default prompt>",
    ) as mock_template:
        goldens = await _run_async_generate(synth)

    mock_template.assert_called_once()
    assert len(goldens) == 1


@pytest.mark.asyncio
async def test_async_custom_prompt_builder_called_with_correct_kwargs():
    """When prompt_builder is set, the async loop must call it with the
    expected keyword arguments and must NOT fall through to
    SynthesizerTemplate.generate_synthetic_inputs."""
    custom_builder = MagicMock(return_value="<my prompt>")
    synth = _make_synthesizer(prompt_builder=custom_builder)
    _stub_async_internals(synth)

    context = ["ctx sentence"]

    with patch(
        "deepeval.synthesizer.synthesizer.SynthesizerTemplate.generate_synthetic_inputs",
    ) as mock_template:
        goldens = await _run_async_generate(synth, context=context)

    custom_builder.assert_called_once_with(
        context=context,
        max_goldens_per_context=1,
        scenario=synth.styling_config.scenario,
        task=synth.styling_config.task,
        input_format=synth.styling_config.input_format,
    )
    mock_template.assert_not_called()
    assert len(goldens) == 1
