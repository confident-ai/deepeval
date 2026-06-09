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


class TestCrossFileSourceLabeling:
    def test_format_context_with_sources_labels_each_chunk(self):
        synth = _make_synthesizer()

        context = ["chunk A text", "chunk B text"]
        chunk_source_files = ["file_a.txt", "file_b.txt"]

        formatted = synth._format_context_with_sources(
            context=context,
            chunk_source_files=chunk_source_files,
        )

        assert formatted == [
            "[SOURCE: file_a.txt] chunk A text",
            "[SOURCE: file_b.txt] chunk B text",
        ]

    def test_format_context_single_file_is_not_labeled(self):
        """Single-file contexts (the default path) must be left untouched."""
        synth = _make_synthesizer()

        context = ["chunk A text", "chunk B text"]
        chunk_source_files = ["file_a.txt", "file_a.txt"]

        formatted = synth._format_context_with_sources(
            context=context,
            chunk_source_files=chunk_source_files,
        )

        assert formatted == context

    def test_format_context_no_sources_returns_context(self):
        synth = _make_synthesizer()
        context = ["only chunk"]

        assert synth._format_context_with_sources(context, None) == context
        # Mismatched lengths fall back to the raw context.
        assert (
            synth._format_context_with_sources(context, ["a", "b"]) == context
        )


class TestBuildContextsWithSources:
    def test_build_contexts_attaches_source_per_chunk(self):
        synth = _make_synthesizer()

        contexts = [["c1", "c2"], ["c3"]]
        source_files = ["doc1.txt", "doc2.txt"]
        scores = [0.9, 0.7]

        built = synth._build_contexts_with_sources(
            contexts=contexts,
            source_files=source_files,
            context_scores=scores,
        )

        assert len(built) == 2
        assert built[0].source_files == ["doc1.txt"]
        assert built[0].chunk_source_files == ["doc1.txt", "doc1.txt"]
        assert built[0].score == 0.9
        assert built[1].chunk_source_files == ["doc2.txt"]

    def test_build_contexts_handles_missing_sources_and_scores(self):
        synth = _make_synthesizer()

        built = synth._build_contexts_with_sources(
            contexts=[["c1"]],
            source_files=[],
            context_scores=None,
        )

        assert built[0].source_files == []
        assert built[0].chunk_source_files == []
        assert built[0].score is None


class TestMergeCrossFileContexts:
    @staticmethod
    def _fake_embedder():
        # Deterministic, orthogonal embeddings so similarity never drives the
        # outcome — selection then depends only on the disjoint-source rule.
        embedder = MagicMock()
        embedder.embed_texts.side_effect = lambda texts: [
            [1.0] * len(texts) for _ in texts
        ]
        return embedder

    def _build(self, synth, source_files):
        return synth._build_contexts_with_sources(
            contexts=[[f"chunk from {s}"] for s in source_files],
            source_files=source_files,
            context_scores=[0.5] * len(source_files),
        )

    def test_merge_partitions_without_duplicating_contexts(self):
        synth = _make_synthesizer()
        built = self._build(synth, ["a.txt", "b.txt", "c.txt", "d.txt"])

        merged = synth._merge_cross_file_contexts(
            built,
            self._fake_embedder(),
            target_files_per_context=2,
        )

        # Each original file appears in exactly one merged context.
        seen = [s for m in merged for s in m.source_files]
        assert sorted(seen) == ["a.txt", "b.txt", "c.txt", "d.txt"]
        assert len(seen) == len(set(seen))  # no duplication
        # target=2 -> groups of 2 distinct files.
        assert all(len(m.source_files) == 2 for m in merged)

    def test_merge_keeps_chunk_labels_aligned(self):
        synth = _make_synthesizer()
        built = self._build(synth, ["a.txt", "b.txt"])

        merged = synth._merge_cross_file_contexts(
            built, self._fake_embedder(), target_files_per_context=2
        )

        assert len(merged) == 1
        group = merged[0]
        assert len(group.chunk_source_files) == len(group.context)
        assert set(group.chunk_source_files) == {"a.txt", "b.txt"}

    def test_merge_respects_max_files_per_context(self):
        synth = _make_synthesizer()
        built = self._build(
            synth, ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
        )

        merged = synth._merge_cross_file_contexts(
            built,
            self._fake_embedder(),
            target_files_per_context=10,  # asks for more than the cap
            max_files_per_context=2,
        )

        assert all(len(m.source_files) <= 2 for m in merged)

    def test_merge_rejects_target_below_two(self):
        synth = _make_synthesizer()
        built = self._build(synth, ["a.txt", "b.txt"])

        with pytest.raises(ValueError):
            synth._merge_cross_file_contexts(
                built, self._fake_embedder(), target_files_per_context=1
            )

    def test_merge_rejects_max_files_below_two(self):
        synth = _make_synthesizer()
        built = self._build(synth, ["a.txt", "b.txt"])

        with pytest.raises(ValueError):
            synth._merge_cross_file_contexts(
                built, self._fake_embedder(), max_files_per_context=1
            )

    def test_merge_noop_for_single_context(self):
        synth = _make_synthesizer()
        built = self._build(synth, ["a.txt"])

        merged = synth._merge_cross_file_contexts(built, self._fake_embedder())

        assert merged == built

    def test_merge_skips_embedding_when_one_distinct_file(self):
        """Multiple same-file contexts -> nothing to merge, no embed cost."""
        synth = _make_synthesizer()
        built = synth._build_contexts_with_sources(
            contexts=[["c1"], ["c2"], ["c3"]],
            source_files=["a.txt", "a.txt", "a.txt"],
            context_scores=[0.5, 0.5, 0.5],
        )
        embedder = self._fake_embedder()

        merged = synth._merge_cross_file_contexts(built, embedder)

        assert merged == built
        embedder.embed_texts.assert_not_called()

    def test_merge_passes_through_unlabeled_contexts(self):
        """A source-less context must not corrupt chunk-label alignment."""
        synth = _make_synthesizer()
        built = synth._build_contexts_with_sources(
            contexts=[["a chunk"], ["b chunk"], ["unlabeled chunk"]],
            source_files=["a.txt", "b.txt"],  # 3rd context has no source
            context_scores=[0.5, 0.5, 0.5],
        )
        assert built[2].source_files == []

        merged = synth._merge_cross_file_contexts(
            built, self._fake_embedder(), target_files_per_context=2
        )

        # Every merged context keeps chunk labels aligned (or has none).
        for m in merged:
            assert not m.chunk_source_files or len(m.chunk_source_files) == len(
                m.context
            )
        # The unlabeled chunk survives untouched in exactly one context.
        assert any("unlabeled chunk" in m.context for m in merged)
        # a.txt and b.txt still merged together despite the stray context.
        assert any(set(m.source_files) == {"a.txt", "b.txt"} for m in merged)

    @pytest.mark.asyncio
    async def test_async_merge_uses_async_embedder(self):
        synth = _make_synthesizer()
        built = self._build(synth, ["a.txt", "b.txt"])

        embedder = MagicMock()
        embedder.a_embed_texts = AsyncMock(
            side_effect=lambda texts: [[1.0] for _ in texts]
        )

        merged = await synth._a_merge_cross_file_contexts(
            built, embedder, target_files_per_context=2
        )

        embedder.a_embed_texts.assert_awaited_once()
        assert len(merged) == 1
        assert set(merged[0].source_files) == {"a.txt", "b.txt"}


class TestCosineSimilarity:
    def test_identical_vectors(self):
        synth = _make_synthesizer()
        assert synth._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        synth = _make_synthesizer()
        assert synth._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_is_safe(self):
        synth = _make_synthesizer()
        assert synth._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestSyntheticInputsTemplateGating:
    def test_single_source_file_omits_cross_file_section(self):
        from deepeval.synthesizer.templates.template import SynthesizerTemplate

        prompt = SynthesizerTemplate.generate_synthetic_inputs(
            context=["[SOURCE: a.txt] chunk"],
            max_goldens_per_context=2,
            scenario=None,
            task=None,
            input_format=None,
            available_source_files=["a.txt"],
        )

        assert "used_source_files" not in prompt
        assert "cross-file" not in prompt

    def test_multi_source_files_inject_section_and_example(self):
        from deepeval.synthesizer.templates.template import SynthesizerTemplate

        prompt = SynthesizerTemplate.generate_synthetic_inputs(
            context=["[SOURCE: a.txt] c1", "[SOURCE: b.txt] c2"],
            max_goldens_per_context=2,
            scenario=None,
            task=None,
            input_format=None,
            available_source_files=["a.txt", "b.txt"],
            target_files_per_context=2,
        )

        assert "used_source_files" in prompt
        # The few-shot example must demonstrate the new key.
        assert '"used_source_files": ["a.txt", "b.txt"]' in prompt
        assert "at least 2 different source files" in prompt


class TestSyntheticScenariosTemplateGating:
    def test_single_source_file_omits_cross_file_section(self):
        from deepeval.synthesizer.templates.template import SynthesizerTemplate

        prompt = SynthesizerTemplate.generate_synthetic_scenarios(
            context=["[SOURCE: a.txt] chunk"],
            max_goldens_per_context=2,
            scenario_context=None,
            conversational_task=None,
            participant_roles=None,
            available_source_files=["a.txt"],
        )

        assert "used_source_files" not in prompt
        assert "cross-file" not in prompt

    def test_multi_source_files_inject_section_and_example(self):
        from deepeval.synthesizer.templates.template import SynthesizerTemplate

        prompt = SynthesizerTemplate.generate_synthetic_scenarios(
            context=["[SOURCE: a.txt] c1", "[SOURCE: b.txt] c2"],
            max_goldens_per_context=2,
            scenario_context=None,
            conversational_task=None,
            participant_roles=None,
            available_source_files=["a.txt", "b.txt"],
            target_files_per_context=2,
        )

        assert "used_source_files" in prompt
        # The few-shot example must demonstrate the new key.
        assert '"used_source_files": ["a.txt", "b.txt"]' in prompt
        assert "at least 2 different source files" in prompt


class TestRewriteScenariosPreservesSources:
    def _passing_feedback(self):
        feedback = MagicMock()
        feedback.feedback = "great"
        feedback.score = 1.0  # above default 0.5 threshold -> no rewrite
        return feedback

    def test_sync_rewrite_keeps_used_source_files(self):
        from deepeval.synthesizer.schema import ConversationalScenario

        synth = _make_synthesizer()
        synth._generate_schema = MagicMock(
            return_value=self._passing_feedback()
        )

        scenarios = [
            ConversationalScenario(
                scenario="a scenario",
                used_source_files=["a.txt", "b.txt"],
            )
        ]
        filtered, scores = synth._rewrite_scenarios(["ctx"], scenarios)

        assert filtered[0].used_source_files == ["a.txt", "b.txt"]
        assert scores == [1.0]

    @pytest.mark.asyncio
    async def test_async_rewrite_keeps_used_source_files(self):
        from deepeval.synthesizer.schema import ConversationalScenario

        synth = _make_synthesizer()
        synth._a_generate_schema = AsyncMock(
            return_value=self._passing_feedback()
        )

        scenarios = [
            ConversationalScenario(
                scenario="a scenario",
                used_source_files=["a.txt", "b.txt"],
            )
        ]
        filtered, scores = await synth._a_rewrite_scenarios(["ctx"], scenarios)

        assert filtered[0].used_source_files == ["a.txt", "b.txt"]
        assert scores == [1.0]
