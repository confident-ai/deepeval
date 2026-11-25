"""
Regression tests for per-golden evolutions metadata on ConversationalGolden.

`generate_conversational_goldens_from_scratch` (sync) used to reuse the
`evolutions_used` loop variable from the LAST evolved scenario for every
golden it synthesized, while the async counterpart
(`a_generate_conversational_goldens_from_scratch`) correctly carried each
scenario's own evolutions through `(evolved_scenario, evolutions)` tuples.

These tests stub out every LLM touchpoint (scenario generation and
evolution) and assert each returned ConversationalGolden carries the
evolutions that were actually used to evolve its own scenario.
"""

from typing import List

import pytest

from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import ConversationalStylingConfig
from deepeval.synthesizer.schema import ConversationalScenario
import deepeval.synthesizer.synthesizer as synth_mod
from deepeval.dataset import ConversationalGolden
from tests.test_core.stubs import (
    DummyModel,
    DummyEvolutionConfig,
    stub_synthesizer_progress_context,
)

NUM_SCENARIOS = 3


def _make_conversational_synthesizer(*, async_mode: bool) -> Synthesizer:
    # Build an instance without running __init__ so no model, embedder,
    # or telemetry setup is needed. These are the only attributes the
    # from-scratch conversational paths touch.
    s = Synthesizer.__new__(Synthesizer)
    s.async_mode = async_mode
    s.max_concurrent = 2
    s.model = DummyModel()
    s.evolution_config = DummyEvolutionConfig()
    s.synthetic_conversational_goldens = []
    s.synthesis_cost = 0
    s.cost_tracking = False
    s.using_native_model = False
    s.conversational_styling_config = ConversationalStylingConfig(
        scenario_context="Customer service interactions",
        conversational_task="Resolve customer complaints",
        participant_roles="Agent and customer",
    )
    return s


def _stub_common_pipeline(monkeypatch, synthesizer: Synthesizer):
    # Avoid Rich Progress setup in tests, and neuter add_pbar since the
    # DummyProgress stub does not implement add_task.
    monkeypatch.setattr(
        synth_mod,
        "synthesizer_progress_context",
        stub_synthesizer_progress_context,
    )
    monkeypatch.setattr(synth_mod, "add_pbar", lambda *args, **kwargs: None)

    scenarios = [
        ConversationalScenario(scenario=f"scenario-{i}")
        for i in range(NUM_SCENARIOS)
    ]
    monkeypatch.setattr(synthesizer, "_generate_scenarios", lambda prompt: scenarios)


def _assert_per_golden_metadata(goldens: List[ConversationalGolden]):
    assert len(goldens) == NUM_SCENARIOS
    for i, golden in enumerate(goldens):
        assert isinstance(golden, ConversationalGolden)
        assert golden.scenario == f"evolved-scenario-{i}"
        # Each golden must carry the evolutions used for ITS OWN scenario,
        # not the ones from the last iteration of the evolve loop.
        assert golden.additional_metadata == {
            "evolutions": [f"evolution-used-by-scenario-{i}"]
        }


def test_sync_generate_conversational_goldens_from_scratch_keeps_per_golden_evolutions(
    monkeypatch,
):
    s = _make_conversational_synthesizer(async_mode=False)
    _stub_common_pipeline(monkeypatch, s)

    def fake_evolve_scenario(scenario, **kwargs):
        return f"evolved-{scenario}", [f"evolution-used-by-{scenario}"]

    monkeypatch.setattr(s, "_evolve_scenario", fake_evolve_scenario)

    goldens = s.generate_conversational_goldens_from_scratch(
        num_goldens=NUM_SCENARIOS,
        _send_data=False,
    )

    _assert_per_golden_metadata(goldens)
    # The fix must not regress internal bookkeeping.
    assert s.synthetic_conversational_goldens == goldens


@pytest.mark.asyncio
async def test_async_generate_conversational_goldens_from_scratch_keeps_per_golden_evolutions(
    monkeypatch,
):
    s = _make_conversational_synthesizer(async_mode=True)
    _stub_common_pipeline(monkeypatch, s)

    async def fake_a_evolve_scenario(scenario, **kwargs):
        return f"evolved-{scenario}", [f"evolution-used-by-{scenario}"]

    monkeypatch.setattr(s, "_a_evolve_scenario", fake_a_evolve_scenario)

    goldens = await s.a_generate_conversational_goldens_from_scratch(
        num_goldens=NUM_SCENARIOS,
    )

    # The async path is the reference behavior the sync path is aligned to.
    _assert_per_golden_metadata(goldens)
    assert s.synthetic_conversational_goldens == goldens
