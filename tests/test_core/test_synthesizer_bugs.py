"""Tests for three synthesizer bugs:

Bug 1: _a_generate_text_to_sql_from_context crashes with AttributeError
       when include_expected_output=False (expected_output is None).
Bug 2: generate_goldens_from_scratch sync path assigns every golden the
       evolutions metadata from the *last* loop iteration.
Bug 3: _rewrite_inputs / _a_rewrite_inputs raises UnboundLocalError
       when max_quality_retries=0.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepeval.synthesizer.config import FiltrationConfig, StylingConfig
from deepeval.synthesizer.schema import (
    SQLData,
    SyntheticData,
)


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

    defaults = dict(
        model=fake_model,
        async_mode=False,
    )
    defaults.update(overrides)

    # Patch initialize_model everywhere it is imported
    with patch(_INIT_MODEL_PATHS[0], return_value=(fake_model, True)), patch(
        _INIT_MODEL_PATHS[1], return_value=(fake_model, True)
    ):
        synth = Synthesizer(**defaults)
    return synth


# ===================================================================
# Bug 1: text_to_sql – AttributeError on None expected_output
# ===================================================================


class TestTextToSqlNoneExpectedOutput:
    """When include_expected_output=False the golden's expected_output
    must be None, not crash with AttributeError: 'NoneType' has no
    attribute 'sql'."""

    @pytest.mark.asyncio
    async def test_no_crash_when_expected_output_disabled(self):
        synth = _make_synthesizer()

        # Mock _a_generate_inputs to control what synthetic inputs come back
        synthetic_inputs = [SyntheticData(input="show all users")]
        synth._a_generate_inputs = AsyncMock(return_value=synthetic_inputs)

        goldens = []
        context = ["CREATE TABLE users (id INT, name TEXT)"]

        await synth._a_generate_text_to_sql_from_context(
            context=context,
            goldens=goldens,
            include_expected_output=False,
            max_goldens_per_context=1,
            progress_bar=None,
        )

        assert len(goldens) == 1
        assert goldens[0].expected_output is None

    @pytest.mark.asyncio
    async def test_expected_output_populated_when_enabled(self):
        synth = _make_synthesizer()

        sql_data = SQLData(sql="SELECT * FROM users")
        synth._a_generate_schema = AsyncMock(return_value=sql_data)
        synth._a_generate_inputs = AsyncMock(
            return_value=[SyntheticData(input="show all users")]
        )

        goldens = []
        context = ["CREATE TABLE users (id INT, name TEXT)"]

        await synth._a_generate_text_to_sql_from_context(
            context=context,
            goldens=goldens,
            include_expected_output=True,
            max_goldens_per_context=1,
            progress_bar=None,
        )

        assert len(goldens) == 1
        assert goldens[0].expected_output == "SELECT * FROM users"


# ===================================================================
# Bug 2: generate_goldens_from_scratch – wrong evolutions metadata
# ===================================================================


class TestFromScratchEvolutionsMetadata:
    """Each golden must preserve its own evolutions list, not the
    last iteration's."""

    def test_each_golden_has_own_evolutions(self):
        synth = _make_synthesizer(
            styling_config=StylingConfig(
                scenario="test scenario",
                task="test task",
                input_format="question",
            ),
        )

        # Two inputs that will evolve differently
        inputs = [
            SyntheticData(input="input_A"),
            SyntheticData(input="input_B"),
        ]
        synth._generate_inputs = MagicMock(return_value=inputs)

        # _evolve_input returns (evolved_prompt, evolutions_used)
        # Make each call return a different evolutions list
        call_count = 0

        def fake_evolve_input(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("evolved_A", ["Reasoning"])
            else:
                return ("evolved_B", ["Comparative", "Hypothetical"])

        synth._evolve_input = MagicMock(side_effect=fake_evolve_input)

        with patch(
            "deepeval.synthesizer.synthesizer.synthesizer_progress_context"
        ) as mock_ctx:
            progress_mock = MagicMock()
            mock_ctx.return_value.__enter__ = MagicMock(
                return_value=(progress_mock, 0)
            )
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            with patch(
                "deepeval.synthesizer.synthesizer.add_pbar",
                return_value=0,
            ), patch("deepeval.synthesizer.synthesizer.update_pbar"), patch(
                "deepeval.synthesizer.synthesizer.remove_pbars"
            ):
                goldens = synth.generate_goldens_from_scratch(
                    num_goldens=2,
                    _send_data=False,
                )

        assert len(goldens) == 2
        # Bug 2 regression: before the fix, both goldens had the
        # *last* iteration's evolutions (["Comparative", "Hypothetical"])
        assert goldens[0].additional_metadata["evolutions"] == ["Reasoning"]
        assert goldens[1].additional_metadata["evolutions"] == [
            "Comparative",
            "Hypothetical",
        ]


# ===================================================================
# Bug 3: _rewrite_inputs / _a_rewrite_inputs – UnboundLocalError
# ===================================================================


class TestRewriteInputsZeroRetries:
    """When max_quality_retries=0, score must default instead of
    raising UnboundLocalError."""

    def test_sync_no_crash_with_zero_retries(self):
        synth = _make_synthesizer()

        with patch(
            _INIT_MODEL_PATHS[1],
            return_value=(MagicMock(), True),
        ):
            synth.filtration_config = FiltrationConfig(max_quality_retries=0)

        inputs = [
            SyntheticData(input="question 1"),
            SyntheticData(input="question 2"),
        ]
        context = ["some context"]

        filtered, scores = synth._rewrite_inputs(context, inputs)

        assert len(filtered) == 2
        assert len(scores) == 2
        # Scores default to 0.0 when the retry loop never executes
        assert all(s == 0.0 for s in scores)
        # Inputs pass through unchanged
        assert filtered[0].input == "question 1"
        assert filtered[1].input == "question 2"

    @pytest.mark.asyncio
    async def test_async_no_crash_with_zero_retries(self):
        synth = _make_synthesizer()

        with patch(
            _INIT_MODEL_PATHS[1],
            return_value=(MagicMock(), True),
        ):
            synth.filtration_config = FiltrationConfig(max_quality_retries=0)

        inputs = [
            SyntheticData(input="async question 1"),
            SyntheticData(input="async question 2"),
        ]
        context = ["some context"]

        filtered, scores = await synth._a_rewrite_inputs(context, inputs)

        assert len(filtered) == 2
        assert len(scores) == 2
        assert all(s == 0.0 for s in scores)
        assert filtered[0].input == "async question 1"
        assert filtered[1].input == "async question 2"
