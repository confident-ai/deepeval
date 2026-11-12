"""The Voice Skills benchmark, run against a stub agent with no network."""

import json

import pytest

from deepeval.benchmarks import VoiceSkills
from deepeval.benchmarks.voice_skills.journey import build_graph
from deepeval.test_case import Audio
from deepeval.voice import CallbackVoiceConnector, VoiceConfig
from deepeval.voice.connectors.types import ConnectorTurn
from tests.test_core.test_voice.test_voice_simulation import (
    _StaticVoiceModel,
    _StubSTT,
    _StubTTS,
    _tone,
)

JOURNEY = {
    "id": "01-cancel",
    "skills": ["cancellation_handling"],
    "golden": {
        "scenario": "Cancel a membership",
        "expected_outcome": "Cancelled.",
    },
    "caller": {"description": "Polite and firm.", "accent": "Jamaican"},
    "line": {"quality": "landline"},
    "background": None,
    "agent_setup": {},
    "graph": {
        "start": "open",
        "nodes": [
            {
                "id": "open",
                "say": "I want to cancel.",
                "audio": "01-cancel/open.wav",
            },
            {
                "id": "number",
                "say": "Six six zero three.",
                "when": "The agent asks for the number",
            },
            {
                "id": "name",
                "say": "Winston Clarke.",
                "when": "The agent asks for the name",
            },
            {
                "id": "done",
                "say": "Thanks.",
                "terminal": True,
                "ending": {"name": "cancelled", "result": "expected"},
            },
            {
                "id": "hung_up",
                "say": "Forget it.",
                "terminal": True,
                "ending": {"name": "hung_up", "result": "failure"},
            },
        ],
        "groups": [
            {"id": "ident", "nodes": ["number", "name"], "order": "any"}
        ],
        "edges": [
            {"from": "open", "to": "ident"},
            {
                "from": "ident",
                "to": "done",
                "when": "The agent confirms the cancellation",
            },
            {
                "from": "ident",
                "to": "hung_up",
                "when": "The agent makes a third offer",
            },
        ],
        "reactions": [
            {
                "id": "reask",
                "when": "The agent asks again",
                "say": "I told you.",
                "audio": "01-cancel/reaction-reask.wav",
                "max_visits": 2,
                "after_max_visits": "hung_up",
            },
        ],
        "catch_all": {
            "say": "Sorry, say that again?",
            "audio": "catch-all.wav",
            "off_script_after": 2,
        },
    },
    "expected_tools": [],
    "forbidden_tools": [],
    "metrics": [
        {
            "metric": "TurnTakingNaturalnessMetric",
            "sub_scores": [
                {"skill": "reply_speed"},
                {"skill": "premature_response"},
            ],
        },
    ],
}


class _FirstEdge(_StaticVoiceModel):
    """Always follows the first way out of a node."""

    def generate(self, prompt: str, schema=None):
        if schema is not None and schema.__name__ == "EdgeChoice":
            return schema(index=1, reason="static")
        return super().generate(prompt, schema)


@pytest.fixture
def dataset(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "01-cancel.json").write_text(json.dumps(JOURNEY))
    return str(tmp_path)


def test_a_journey_file_becomes_a_graph_with_groups_reactions_and_endings(
    dataset,
):
    root = build_graph(JOURNEY, dataset)

    assert root.name == "open"
    assert sorted(child.name for child, _ in root.edges) == ["name", "number"]
    assert [reaction.name for reaction, _, _ in root.reactions] == ["reask"]
    assert root.fallback.max_visits == 2
    endings = {
        child.name: child.fails
        for member, _ in root.edges
        for child, _ in member.edges
        if child.terminal
    }
    assert endings == {"done": False, "hung_up": True}


def test_unknown_journey_ids_are_rejected(dataset):
    with pytest.raises(ValueError):
        VoiceSkills(dataset_path=dataset, journeys=["99-missing"])


def test_evaluate_runs_each_journey_and_scores_it_by_skill(dataset):
    async def agent(_audio: Audio) -> ConnectorTurn:
        return ConnectorTurn(audio=_tone(0.4), transcript="the agent speaks")

    benchmark = VoiceSkills(
        dataset_path=dataset,
        n_runs=2,
        simulator_model=_FirstEdge(),
        programmatic_only=True,
    )
    result = benchmark.evaluate(
        VoiceConfig(
            connector=CallbackVoiceConnector(agent, turn_detection="eager"),
            tts_model=_StubTTS(),
            stt_model=_StubSTT(),
            output_dir=None,
        )
    )

    calls = benchmark.predictions
    assert sorted(calls["run"].unique()) == [1, 2]
    assert set(calls["ending"]) == {"cancelled"}
    assert result.skill_scores["task_completion"] == 1.0
    assert result.skill_scores["efficiency"] == 1.0
    assert result.skill_scores["reply_speed"] == 1.0
    # The stub agent answers with a harsh test tone, which audio integrity
    # fails, and one failed skill fails the journey.
    assert result.skill_scores["audio_integrity"] == 0.0
    assert result.journey_scores == {"01-cancel": False}
    assert result.overall_accuracy == 0.0
