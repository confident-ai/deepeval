"""Turns a Voice Skills journey file into what a simulation needs: the
caller graph, the persona it is played through and the metrics that grade
the call."""

import glob
import json
import os
from typing import Any, Dict, List, Optional, Union

from deepeval.dataset import ConversationalGolden, Persona
from deepeval.dataset.golden import BackgroundNoiseSettings, LineSettings
from deepeval.metrics import (
    AudioIntegrityMetric,
    BaseConversationalMetric,
    ConversationalGEval,
    ConversationalToolCorrectnessMetric,
    GoalAccuracyMetric,
    KnowledgeRetentionMetric,
    SimulationEfficiencyMetric,
    SimulationPathMetric,
    TranscriptionAccuracyMetric,
    TurnTakingNaturalnessMetric,
    VoiceStyleMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.simulator import SimulationGroup, SimulationNode
from deepeval.simulator import events as voice_events
from deepeval.test_case import Audio, MultiTurnParams, ToolCall, Turn

# Line settings the journey files may carry that `LineSettings` has no word
# for.
LINE_QUALITIES = {"landline": "phone"}
ALWAYS_SKILLS = ["task_completion", "efficiency"]

Journey = Dict[str, Any]


def load_journeys(dataset_path: str) -> List[Journey]:
    paths = sorted(
        glob.glob(os.path.join(dataset_path, "scenarios", "[0-9]*.json"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No journeys found under {os.path.join(dataset_path, 'scenarios')}."
        )
    journeys = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            journeys.append(json.load(file))
    return journeys


def _recording(dataset_path: str, relative: Optional[str]) -> Optional[Audio]:
    if not relative:
        return None
    path = os.path.join(dataset_path, "audio", relative)
    return Audio(url=path) if os.path.exists(path) else None


def _events(
    dataset_path: str, records: List[Dict], caller_voice: str = "female"
) -> List[voice_events.VoiceEvent]:
    """The voice events a node file lists. Kinds the simulator cannot place
    on a node yet, such as a scripted cut-in, are left out."""
    built = []
    for record in records:
        kind = getattr(voice_events, record["type"], None)
        if kind is None:
            continue
        fields = {
            key: value
            for key, value in record.items()
            if key in kind.__dataclass_fields__ and key != "audio"
        }
        if "audio" in kind.__dataclass_fields__:
            fields["audio"] = _recording(dataset_path, record.get("audio"))
        if kind is voice_events.NonSpeechSound:
            # One recorded sound per kind and voice, shared by every journey.
            sound = f'sounds/{record.get("kind", "cough")}-{caller_voice}.wav'
            fields["audio"] = _recording(dataset_path, sound)
        built.append(kind(**fields))
    return built


def _says(dataset_path: str, text: str, audio: Optional[str]):
    def action() -> Turn:
        return Turn(
            role="user", content=text, audio=_recording(dataset_path, audio)
        )

    return action


REPEAT_WHEN = (
    "The agent asks the caller to repeat or confirm what they just said, or "
    "says it did not catch it or could not find it"
)


def _repeats(dataset_path: str, recordings: Dict[str, Optional[str]]):
    """Says the caller's last line again, the way a person asked to would."""

    def action(last_user_turn: Optional[Turn] = None) -> Turn:
        text = last_user_turn.content if last_user_turn else ""
        return Turn(
            role="user",
            content=text,
            audio=_recording(dataset_path, recordings.get(text)),
        )

    return action


def build_graph(journey: Journey, dataset_path: str) -> SimulationNode:
    graph = journey["graph"]
    nodes: Dict[str, SimulationNode] = {}
    whens: Dict[str, str] = {}
    for node in graph["nodes"]:
        ending = node.get("ending")
        nodes[node["id"]] = SimulationNode(
            action=_says(dataset_path, node["say"], node.get("audio")),
            name=node["id"],
            max_visits=node.get("max_visits"),
            ending=ending["name"] if ending else None,
            fails=bool(node.get("fails_call"))
            or (ending is not None and ending["result"] == "failure"),
            events=_events(
                dataset_path,
                node.get("events", []),
                journey["caller"].get("voice", "female"),
            ),
        )
        whens[node["id"]] = node.get("when", "")
    groups = {
        group["id"]: SimulationGroup(
            [(nodes[member], whens[member]) for member in group["nodes"]]
        )
        for group in graph.get("groups", [])
    }

    for edge in graph["edges"]:
        source: Union[SimulationNode, SimulationGroup] = (
            groups.get(edge["from"]) or nodes[edge["from"]]
        )
        if edge["to"] in groups:
            source.add_group(groups[edge["to"]])
        else:
            source.add_node(nodes[edge["to"]], when=edge["when"])

    root = nodes[graph["start"]]
    for reaction in graph.get("reactions", []):
        root.add_reaction(
            SimulationNode(
                action=_says(
                    dataset_path, reaction["say"], reaction.get("audio")
                ),
                name=reaction["id"],
                max_visits=reaction.get("max_visits"),
            ),
            when=reaction["when"],
            then=nodes[reaction["after_max_visits"]],
        )
    fallback = graph.get("catch_all")
    asked_again = {reaction["id"] for reaction in graph.get("reactions", [])}
    if fallback and "reask" not in asked_again:
        # A journey with no line for being asked again still has a caller
        # who would repeat themselves twice before giving up.
        recordings = {
            line["say"]: line.get("audio")
            for line in graph["nodes"] + graph.get("reactions", [])
        }
        root.add_reaction(
            SimulationNode(
                action=_repeats(dataset_path, recordings),
                name="repeat",
                max_visits=2,
            ),
            when=REPEAT_WHEN,
            then=SimulationNode(
                action=_says(
                    dataset_path, fallback["say"], fallback.get("audio")
                ),
                name="gave_up_repeating",
                terminal=True,
                ending="off_script",
                fails=True,
            ),
        )
    if fallback:
        root.off_script(
            SimulationNode(
                action=_says(
                    dataset_path, fallback["say"], fallback.get("audio")
                ),
                name="off_script",
                max_visits=fallback.get("off_script_after"),
            )
        )
    return root


def build_golden(journey: Journey, dataset_path: str) -> ConversationalGolden:
    caller = journey["caller"]
    line = dict(journey.get("line") or {})
    line["quality"] = LINE_QUALITIES.get(
        line.get("quality"), line.get("quality", "phone")
    )
    background = journey.get("background") or {}
    noise_path = os.path.join(
        dataset_path, "backgrounds", background.get("audio", "")
    )
    return ConversationalGolden(
        name=journey["id"],
        scenario=journey["golden"]["scenario"],
        expected_outcome=journey["golden"]["expected_outcome"],
        persona=Persona(
            name=journey["id"],
            characteristics=f'{caller["description"]} Accent: {caller["accent"]}.',
            line=LineSettings(**line),
            background_noise=(
                BackgroundNoiseSettings(
                    audio=noise_path, volume=background["volume"]
                )
                if background.get("audio") and os.path.exists(noise_path)
                else None
            ),
            # The agent answers the phone, so it speaks first.
            speaks_first=False,
        ),
    )


def _tool(call: Dict) -> ToolCall:
    return ToolCall(
        name=call["name"], input_parameters=call["input_parameters"]
    )


def build_metrics(
    journey: Journey,
    judge: Optional[Union[str, DeepEvalBaseLLM]] = None,
    audio_judge: Optional[Union[str, DeepEvalBaseLLM]] = None,
    programmatic_only: bool = False,
) -> Dict[str, BaseConversationalMetric]:
    """The metrics for one journey, keyed by the skill each one grades."""
    setup = journey.get("agent_setup") or {}
    metrics: Dict[str, BaseConversationalMetric] = {
        "task_completion": SimulationPathMetric(),
        "efficiency": SimulationEfficiencyMetric(
            expected_tool_calls=len(journey["expected_tools"]), threshold=0.5
        ),
        "audio_integrity": AudioIntegrityMetric(),
    }
    if journey["expected_tools"] or journey["forbidden_tools"]:
        metrics["entity_extraction"] = ConversationalToolCorrectnessMetric(
            expected_tools=[_tool(call) for call in journey["expected_tools"]],
            forbidden_tools=[
                _tool(call) for call in journey["forbidden_tools"]
            ],
            should_consider_ordering="tool_order" in setup,
        )
    for spec in journey["metrics"]:
        if spec["metric"] == "TurnTakingNaturalnessMetric":
            for sub in spec["sub_scores"]:
                metrics[sub["skill"]] = TurnTakingNaturalnessMetric(
                    sub_scores=[sub["skill"]]
                )
    if programmatic_only:
        return metrics

    metrics["goal_accuracy"] = GoalAccuracyMetric(model=judge)
    metrics["transcription_accuracy"] = TranscriptionAccuracyMetric(model=judge)
    metrics["redundant_reprompts"] = KnowledgeRetentionMetric(model=judge)
    for spec in journey["metrics"]:
        if spec["metric"] == "ConversationalGEval":
            metrics[spec["skill"]] = ConversationalGEval(
                name=spec["name"],
                criteria=spec["criteria"],
                evaluation_params=[
                    MultiTurnParams.CONTENT,
                    MultiTurnParams.SCENARIO,
                    MultiTurnParams.EXPECTED_OUTCOME,
                ],
                model=judge,
            )
    if "style_instructions" in journey["skills"] and setup.get("voice_style"):
        metrics["style_instructions"] = VoiceStyleMetric(
            style=setup["voice_style"], model=audio_judge
        )
    return metrics
