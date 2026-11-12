"""Turns a Voice Skills journey file into what a simulation needs: the
caller graph, the persona it is played through and the metrics that grade
the call."""

import glob
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

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


def _says(
    dataset_path: str,
    text: str,
    audio: Optional[str],
    metadata: Optional[Dict] = None,
):
    def action() -> Turn:
        return Turn(
            role="user",
            content=text,
            audio=_recording(dataset_path, audio),
            metadata=metadata,
        )

    return action


class ShortAnswer(BaseModel):
    reply: Literal[
        "fine",
        "go_ahead",
        "okay",
        "no",
        "unsure",
        "carry_on",
        "done",
        "handed_over",
        "again",
    ]


SHORT_ANSWER_PROMPT = """You are playing a caller on the phone to a company. The agent has just said something your script has no line for, so you can only give one of five short answers.

What the call is about: {scenario}
Who you are and how you behave: {summary}
What you want to be true by the end: {outcome}

The call so far:
{history}

The agent just said:
<<<
{reply}
>>>

Pick the answer the caller would give:
- "fine": the agent proposes or asks to confirm something that agrees with what the caller wants. Said as "Yes, that's fine."
- "go_ahead": the agent offers to do something the caller wants, or asks permission to carry on. Said as "Yes, go ahead."
- "okay": the agent gives an instruction or a piece of information and waits to be acknowledged. Said as "Okay."
- "no": what the agent proposes, assumes or asks to confirm goes against what the caller wants, or is not true. Said as "No."
- "handed_over": the agent says it is transferring the caller, putting them through to someone, or ending the call. Said as "Okay."
- "done": the agent is wrapping up or asks if there is anything else, and everything the caller rang about has been dealt with. Never pick this while something the caller wants is still outstanding. Said as "No, that's all, thanks. Bye."
- "unsure": the agent asks for something the caller would not know or has not been given. Said as "I'm not sure, sorry."
- "carry_on": the agent is stalling, chatting, or asking about something that does not matter to this call. Said as "Can we just carry on?"
- "again": what the agent said was cut off or made no sense. Said as "Sorry, can you say that again?"

Reply with JSON only: {{"reply": "<one of fine, no, unsure, carry_on, again>"}}"""


def _short_answer(dataset_path: str, answers: Dict, journey: Journey):
    """The caller's reply to something the script did not see coming: one of
    a few short recorded answers, picked to fit what the caller wants."""

    async def action(
        simulator=None, last_assistant_turn=None, turns=None
    ) -> Turn:
        key = "again"
        said = last_assistant_turn.content if last_assistant_turn else ""
        if simulator is not None and said:
            choice = await simulator.a_generate_schema(
                SHORT_ANSWER_PROMPT.format(
                    scenario=journey["golden"]["scenario"],
                    summary=journey["summary"],
                    outcome=journey["golden"]["expected_outcome"],
                    history="\n".join(
                        f'{"Caller" if turn.role == "user" else "Agent"}: '
                        f"{turn.content}"
                        for turn in (turns or [])[-12:]
                    ),
                    reply=said,
                ),
                ShortAnswer,
            )
            key = getattr(choice, "reply", None) or "again"
        # Handing over is answered the way any "Okay." is.
        answer = answers.get(key) or answers.get(
            "okay" if key == "handed_over" else "again"
        )
        return Turn(
            role="user",
            content=answer["say"],
            audio=_recording(dataset_path, answer.get("audio")),
            metadata={"short_answer": key},
        )

    return action


REPEAT_WHEN = (
    "The agent says it did not hear or understand what the caller just said, "
    "or asks them to say it again"
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


LOOK_AHEAD = 4
# Skills whose caller behaviour, cutting in on the agent or saying "mm-hmm"
# while it talks, the simulated caller cannot perform yet. They are left out
# of grading until it can.
NOT_YET_GRADED = {"overlapping_speech", "backchannel_vs_barge_in"}
# Tools that read a record rather than change anything.
LOOKUPS = ("lookup", "look_up", "get", "find", "verify", "check", "search", "list", "track", "trace", "fetch", "retrieve", "locate", "identify", "validate", "authenticate")


_ASKS = re.compile(r"\basks\b.*", re.IGNORECASE)


def _is_answer(when: str) -> bool:
    """A line the caller only says because the agent asked for it."""
    return bool(_ASKS.search(when))


def _the_question(when: str) -> str:
    """The asking part of an edge condition, such as "The agent asks for a
    phone number" out of "The agent confirms the name and asks for a phone
    number". Whatever else the agent did, it is the question the caller
    answers."""
    found = _ASKS.search(when)
    return "The agent " + found.group(0) if found else when


def _answer_out_of_order(
    graph: Dict,
    nodes: Dict[str, SimulationNode],
    groups: List[SimulationGroup],
    dataset_path: str,
) -> None:
    """Let the caller answer a question the script expected later.

    A real caller asked for their date of birth before their reference gives
    it. Two things make that work. A detail the caller only gives when asked
    can be given at any point, once, and the call carries on from where it
    was. And a line can lead past such details to what follows them, since
    an agent that already has them will not ask again. A line where the
    caller asks for something or changes their mind is never skipped, and
    corrections, failures and endings stay where the script put them: they
    have to be earned.
    """

    def ordinary(node: SimulationNode) -> bool:
        return (
            not node.terminal and not node.fails and node.max_visits is None
        )

    incoming: Dict[int, List[str]] = {}
    for edge in graph["edges"]:
        for target in [edge["to"]]:
            if target in nodes:
                incoming.setdefault(id(nodes[target]), []).append(
                    edge.get("when", "")
                )
    grouped = {id(member) for group in groups for member, _ in group.members}
    answers = {
        id(node)
        for node in nodes.values()
        if ordinary(node)
        and id(node) not in grouped
        and incoming.get(id(node))
        and all(_is_answer(when) for when in incoming[id(node)])
    }
    scripted = {id(node): list(node.edges) for node in nodes.values()}
    details = [pair for group in groups for pair in group.members]

    for node in nodes.values():
        if node.terminal:
            continue
        reached = {id(child) for child, _ in node.edges} | {id(node)}
        ahead, frontier = [], [node]
        for _ in range(LOOK_AHEAD):
            # Only a detail the agent may already have can be passed over.
            frontier = [
                child
                for current in frontier
                for child, _ in scripted[id(current)]
                if id(child) in answers
            ]
            for current in frontier:
                ahead.extend(scripted[id(current)])
        for child, when in details + ahead:
            if id(child) in reached or not ordinary(child):
                continue
            reached.add(id(child))
            node.add_node(child, when)

    root = nodes[graph["start"]]
    for line in graph["nodes"]:
        node = nodes[line["id"]]
        if id(node) not in answers:
            continue
        out_of_turn = SimulationNode(
            action=_says(
                dataset_path, line.get("text", line["say"]), line.get("audio")
            ),
            name=line["id"],
            max_visits=1,
        )
        # Said once, here or in its place in the script, never both.
        out_of_turn.same_line_as = node
        node.same_line_as = out_of_turn
        root.add_reaction(
            out_of_turn,
            when=" or ".join(
                dict.fromkeys(_the_question(when) for when in incoming[id(node)])
            ),
        )


_VALUE = re.compile(
    r"\d|\b(zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|double|dot|at)\b|\b[A-Z]\b",
    re.IGNORECASE,
)
_READ_BACK = re.compile(r"\bread(s|ing)?\b.*\bback\b|\brepeats?\b", re.I)
_WRONG = re.compile(r"\b(different|other than|wrong|instead)\b", re.I)


def _confirm_read_backs(
    lines: List[Dict],
    nodes: Dict[str, SimulationNode],
    dataset_path: str,
    confirm: Dict,
) -> None:
    """Let the caller say yes to an agent that reads a detail back correctly.

    Without it, a careful agent that checks what it heard is treated as one
    asking for the detail again. The yes is not counted as a caller turn when
    efficiency is worked out, so checking costs the agent nothing.

    Only a line that carries a value (a number, a code, a spelling, a name)
    gets one. Where the script already has the caller's answer to a correct
    read-back, the yes stands in for it, so the call goes on from there
    rather than from before the read-back.
    """
    for line in lines:
        node = nodes[line["id"]]
        text = line.get("text", line["say"])
        if node.terminal or not _VALUE.search(text):
            continue
        yes = SimulationNode(
            action=_says(
                dataset_path,
                confirm["say"],
                confirm.get("audio"),
                metadata={"confirmation": True},
            ),
            name="confirms",
        )
        after = next(
            (
                child
                for child, when in node.edges
                if _READ_BACK.search(when)
                and not _WRONG.search(when)
                and not child.terminal
                and not child.fails
                and child.max_visits is None
            ),
            None,
        )
        if after is not None:
            yes.edges = list(after.edges)
            if getattr(after, "same_line_as", None) is None:
                yes.same_line_as = after
                after.same_line_as = yes
        else:
            yes.edges = list(node.edges)
        node.add_node(
            yes,
            when=(
                "The agent repeats back what the caller just said "
                f'("{text}") without getting any of it wrong, and '
                "asks whether that is right"
            ),
        )


def build_graph(journey: Journey, dataset_path: str) -> SimulationNode:
    graph = journey["graph"]
    nodes: Dict[str, SimulationNode] = {}
    whens: Dict[str, str] = {}
    for node in graph["nodes"]:
        ending = node.get("ending")
        nodes[node["id"]] = SimulationNode(
            action=_says(
                dataset_path, node.get("text", node["say"]), node.get("audio")
            ),
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

    _answer_out_of_order(graph, nodes, list(groups.values()), dataset_path)
    if graph.get("confirm"):
        _confirm_read_backs(
            graph["nodes"], nodes, dataset_path, graph["confirm"]
        )

    root = nodes[graph["start"]]
    for reaction in graph.get("reactions", []):
        root.add_reaction(
            SimulationNode(
                action=_says(
                    dataset_path,
                    reaction.get("text", reaction["say"]),
                    reaction.get("audio"),
                ),
                name=reaction["id"],
                max_visits=reaction.get("max_visits"),
            ),
            when=(
                reaction["when"]
                if reaction["id"] != "reask"
                # The line repeats particular details, so it only fits when
                # it is one of those the agent asks for again.
                else "The agent asks again for a detail the caller already "
                "gave, and it is one of the details in this reply: "
                f'"{reaction.get("text", reaction["say"])}"'
            ),
            then=nodes[reaction["after_max_visits"]],
        )
    fallback = graph.get("catch_all")
    asked_again = {reaction["id"] for reaction in graph.get("reactions", [])}
    if fallback and "reask" not in asked_again:
        # A journey with no line for being asked again still has a caller
        # who would repeat themselves twice before giving up.
        recordings = {
            line.get("text", line["say"]): line.get("audio")
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
        answers = graph.get("short_answers")
        root.off_script(
            SimulationNode(
                action=(
                    _short_answer(dataset_path, answers, journey)
                    if answers
                    else _says(
                        dataset_path, fallback["say"], fallback.get("audio")
                    )
                ),
                name="off_script",
                max_visits=fallback.get("off_script_after"),
            )
        )
    return root


# Short answers that close a call: the caller saying they have nothing else,
# or the agent handing them over or hanging up.
CLOSING = {"done": "wrapped_up", "handed_over": "handed_over"}


def settle_ending(test_case: Any) -> None:
    """A call closed by the caller saying goodbye, or by the agent handing
    over, ends there however many short answers it took. Whether it ended
    well is then down to the agent having done everything it had to, which
    `closed_without_the_work` checks."""
    metadata = test_case.metadata or {}
    said = [turn for turn in test_case.turns if turn.role == "user"]
    key = (said[-1].metadata or {}).get("short_answer") if said else None
    if key not in CLOSING:
        return
    if metadata.get("Simulation ending") not in (None, "off_script"):
        return
    metadata["Simulation ending"] = CLOSING[key]
    metadata["Simulation failed"] = bool(metadata.get("Failed nodes"))
    metadata["Closed by a short answer"] = True
    test_case.metadata = metadata


def closed_without_the_work(test_case: Any, tools_score: Any) -> bool:
    """A call the caller closed with a short answer only counts as finished
    when every expected tool call was made."""
    closed = (test_case.metadata or {}).get("Closed by a short answer")
    return bool(closed) and tools_score is not None and tools_score < 1


def stop_when_caller_is_done(turns: List[Turn]) -> Any:
    """Ends the call once the caller has said goodbye and been answered."""
    from deepeval.simulator.controller import end, proceed

    said = [turn for turn in turns if turn.role == "user"]
    if (
        said
        and turns[-1].role == "assistant"
        and (said[-1].metadata or {}).get("short_answer") in CLOSING
    ):
        return end("The call was closed.")
    return proceed()


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
            lookup_tools=[
                call["name"]
                for call in journey["expected_tools"]
                if call["name"].startswith(LOOKUPS)
            ],
        )
    for spec in journey["metrics"]:
        if spec["metric"] == "TurnTakingNaturalnessMetric":
            for sub in spec["sub_scores"]:
                if sub["skill"] in NOT_YET_GRADED:
                    continue
                metrics[sub["skill"]] = TurnTakingNaturalnessMetric(
                    sub_scores=[sub["skill"]]
                )
    if programmatic_only:
        return metrics

    # Whether the caller got what they rang for is the ending the call
    # reached. A judge reading the transcript turn by turn marks even a
    # perfect call down, so it is not used.
    metrics["transcription_accuracy"] = TranscriptionAccuracyMetric(model=judge)
    metrics["redundant_reprompts"] = KnowledgeRetentionMetric(model=judge)
    for spec in journey["metrics"]:
        if (
            spec["metric"] == "ConversationalGEval"
            and spec["skill"] not in NOT_YET_GRADED
        ):
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
            style=setup["voice_style"],
            # The judge has to hear the agent, so it defaults to a model that
            # takes audio.
            model=audio_judge or "gpt-audio-1.5",
        )
    return metrics
