import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepeval.dataset import (
    BackgroundNoiseSettings,
    ConversationalGolden,
    InterruptionBehavior,
    Persona,
)
from deepeval.metrics import (
    AgentResponsivenessMetric,
    TurnTakingNaturalnessMetric,
    VoiceNaturalnessMetric,
)
from deepeval.models.llms import OpenAIModel
from deepeval.models.stt import OpenAISTTModel
from deepeval.models.tts import OpenAITTSModel
from deepeval.simulator import ConversationSimulator
from deepeval.voice import CallbackVoiceConnector, VoiceConfig

from defect_agents import build_clip_library, make_agent

QA_DIR = os.path.dirname(os.path.abspath(__file__))

SCENARIO = (
    "The caller ordered a birthday present ten days ago, it has not arrived, "
    "and they want to know where it is and when it will arrive."
)
EXPECTED_OUTCOME = (
    "The agent locates the order, explains the delay, and gives an arrival "
    "date."
)
CHARACTERISTICS = (
    "A busy customer chasing a late order. Speaks in short direct sentences."
)


@dataclass
class Cell:
    name: str
    agent: str
    frequency: Optional[str] = None
    overlap: Optional[str] = None
    turn_detection: str = "balanced"
    noise: Optional[Tuple[str, float]] = None
    hold_timeout: Optional[float] = None
    max_user_simulations: int = 3
    expect_any_interrupted: Optional[bool] = None
    expect_max_spoken_replies: Optional[int] = None
    expect_wall_under_s: Optional[float] = None
    expect_min_spoken_replies: Optional[int] = None
    expect_sequential: Optional[bool] = None


CELLS = [
    Cell(
        name="control_off",
        agent="control",
        expect_any_interrupted=False,
        expect_min_spoken_replies=2,
        expect_sequential=True,
    ),
    Cell(
        name="rambler_frequent_insist",
        agent="rambler",
        frequency="frequent",
        overlap="insist",
        turn_detection="patient",
        expect_any_interrupted=True,
    ),
    Cell(
        name="rambler_frequent_yield",
        agent="rambler",
        frequency="frequent",
        overlap="yield",
        turn_detection="patient",
    ),
    Cell(
        name="rambler_normal_adaptive",
        agent="rambler",
        frequency="normal",
        overlap="adaptive",
        turn_detection="patient",
    ),
    Cell(
        name="corpse_hangup",
        agent="corpse",
        hold_timeout=20.0,
        max_user_simulations=4,
        expect_max_spoken_replies=2,
        expect_wall_under_s=240.0,
        expect_sequential=True,
    ),
    Cell(
        name="thinker_eager",
        agent="thinker",
        frequency="rare",
        overlap="yield",
        turn_detection="eager",
    ),
    Cell(
        name="thinker_patient",
        agent="thinker",
        frequency="rare",
        overlap="yield",
        turn_detection="patient",
    ),
    Cell(
        name="noise_cafe_light",
        agent="control",
        noise=("cafe.wav", 0.15),
        expect_min_spoken_replies=2,
        expect_sequential=True,
    ),
    Cell(
        name="noise_cafe_heavy",
        agent="control",
        noise=("cafe.wav", 0.5),
        expect_min_spoken_replies=1,
        expect_sequential=True,
    ),
    Cell(
        name="noise_white_heavy",
        agent="control",
        noise=("white.wav", 0.5),
        expect_min_spoken_replies=1,
        expect_sequential=True,
    ),
]


def build_persona(cell: Cell) -> Persona:
    behavior = None
    if cell.frequency is not None:
        behavior = InterruptionBehavior(
            frequency=cell.frequency, overlap=cell.overlap or "adaptive"
        )
    noise = None
    if cell.noise is not None:
        filename, volume = cell.noise
        noise = BackgroundNoiseSettings(
            audio=os.path.join(QA_DIR, "noise", filename), volume=volume
        )
    return Persona(
        characteristics=CHARACTERISTICS,
        interruption_behavior=behavior,
        hold_timeout=cell.hold_timeout,
        background_noise=noise,
    )


def timeline_overlap_s(test_case) -> Optional[float]:
    spans = []
    for turn in test_case.turns:
        audio = turn.audio
        if audio is None or audio.start_time is None or not audio.duration:
            continue
        spans.append((audio.start_time, audio.start_time + audio.duration))
    if len(spans) < 2:
        return None
    spans.sort()
    overlap = 0.0
    running_end = spans[0][1]
    for start, end in spans[1:]:
        if start < running_end:
            overlap += min(end, running_end) - start
        running_end = max(running_end, end)
    return round(overlap, 2)


def run_cell(cell: Cell, library: dict, out_root: str, repeat: int) -> dict:
    out_dir = os.path.join(out_root, f"{cell.name}_r{repeat}")
    simulator = ConversationSimulator(
        simulator_model=OpenAIModel(model="gpt-4o-mini"),
        voice_config=VoiceConfig(
            connector=lambda: CallbackVoiceConnector(
                make_agent(library, cell.agent),
                turn_detection=cell.turn_detection,
            ),
            tts_model=OpenAITTSModel(voice="coral"),
            stt_model=OpenAISTTModel(model="whisper-1"),
            output_dir=out_dir,
            combine_audio_files=True,
        ),
    )
    golden = ConversationalGolden(
        name=cell.name,
        scenario=SCENARIO,
        expected_outcome=EXPECTED_OUTCOME,
        persona=build_persona(cell),
    )

    started = time.perf_counter()
    test_case = simulator.simulate(
        [golden], max_user_simulations=cell.max_user_simulations
    )[0]
    wall_s = time.perf_counter() - started

    assistant_turns = [t for t in test_case.turns if t.role == "assistant"]
    spoken_replies = [
        t for t in assistant_turns if (t.content or "").strip()
    ]
    facts = {
        "wall_s": round(wall_s, 1),
        "turns": len(test_case.turns),
        "assistant_turns": len(assistant_turns),
        "spoken_replies": len(spoken_replies),
        "interrupted_turns": [
            i
            for i, t in enumerate(test_case.turns)
            if bool(getattr(t, "interrupted", None))
        ],
        "first_reply_audio_s": (
            round(assistant_turns[0].audio.duration, 2)
            if assistant_turns and assistant_turns[0].audio
            and assistant_turns[0].audio.duration
            else None
        ),
        "latencies_ms": [
            round(t.latency_ms)
            for t in assistant_turns
            if t.latency_ms is not None
        ],
        "timeline_overlap_s": timeline_overlap_s(test_case),
    }

    metrics = {}
    for metric_class in (
        TurnTakingNaturalnessMetric,
        AgentResponsivenessMetric,
        VoiceNaturalnessMetric,
    ):
        metric = metric_class()
        try:
            metric.measure(test_case, _show_indicator=False)
            metrics[metric.__name__] = {
                "score": (
                    round(metric.score, 3) if metric.score is not None else None
                ),
                "skipped": bool(metric.skipped),
                "reason": metric.reason,
            }
        except Exception as error:
            metrics[metric_class.__name__] = {"error": str(error)}

    checks = []
    interrupted_any = len(facts["interrupted_turns"]) > 0
    if cell.expect_any_interrupted is not None:
        checks.append(
            {
                "check": "interruptions "
                + ("expected" if cell.expect_any_interrupted else "forbidden"),
                "ok": interrupted_any == cell.expect_any_interrupted,
                "observed": facts["interrupted_turns"],
            }
        )
    if cell.expect_max_spoken_replies is not None:
        checks.append(
            {
                "check": f"spoken replies <= {cell.expect_max_spoken_replies}",
                "ok": facts["spoken_replies"] <= cell.expect_max_spoken_replies,
                "observed": facts["spoken_replies"],
            }
        )
    if cell.expect_min_spoken_replies is not None:
        checks.append(
            {
                "check": f"spoken replies >= {cell.expect_min_spoken_replies}",
                "ok": facts["spoken_replies"] >= cell.expect_min_spoken_replies,
                "observed": facts["spoken_replies"],
            }
        )
    if cell.expect_sequential:
        overlap = facts["timeline_overlap_s"]
        checks.append(
            {
                "check": "timeline has no phantom overlap (< 1s)",
                "ok": overlap is not None and overlap < 1.0,
                "observed": overlap,
            }
        )
    if cell.expect_wall_under_s is not None:
        checks.append(
            {
                "check": f"finished under {cell.expect_wall_under_s}s",
                "ok": wall_s < cell.expect_wall_under_s,
                "observed": facts["wall_s"],
            }
        )

    return {
        "cell": cell.name,
        "repeat": repeat,
        "config": asdict(cell),
        "facts": facts,
        "metrics": metrics,
        "checks": checks,
        "recordings": out_dir,
    }


def cross_cell_checks(records: list) -> list:
    checks = []
    by_name = {}
    for record in records:
        by_name.setdefault(record["cell"], []).append(record)
    eager = by_name.get("thinker_eager")
    patient = by_name.get("thinker_patient")
    if eager and patient:
        eager_s = eager[0]["facts"]["first_reply_audio_s"]
        patient_s = patient[0]["facts"]["first_reply_audio_s"]
        if eager_s is not None and patient_s is not None:
            checks.append(
                {
                    "check": "callback signals turn completion, so the "
                    "thinker's pause survives both eager and patient "
                    "(full clip, equal lengths)",
                    "ok": eager_s > 9.0
                    and patient_s > 9.0
                    and abs(eager_s - patient_s) < 0.5,
                    "observed": {"eager_s": eager_s, "patient_s": patient_s},
                }
            )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default="")
    arguments = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the matrix.")
    if not os.path.exists(os.path.join(QA_DIR, "noise", "cafe.wav")):
        raise RuntimeError("Run voice_qa/make_noise.py first.")

    selected = [
        cell
        for cell in CELLS
        if not arguments.cells or cell.name in arguments.cells.split(",")
    ]
    out_root = arguments.out or os.path.join(
        QA_DIR, "results", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_root, exist_ok=True)

    library = asyncio.run(build_clip_library())

    records = []
    for repeat in range(1, arguments.repeats + 1):
        for cell in selected:
            print(f"=== {cell.name} (repeat {repeat}) ===")
            try:
                record = run_cell(cell, library, out_root, repeat)
            except Exception as error:
                record = {
                    "cell": cell.name,
                    "repeat": repeat,
                    "error": str(error),
                }
            records.append(record)
            for check in record.get("checks", []):
                print(
                    f"  [{'PASS' if check['ok'] else 'FAIL'}] "
                    f"{check['check']} -> {check['observed']}"
                )

    summary = {
        "records": records,
        "cross_cell_checks": cross_cell_checks(records),
    }
    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    failures = [
        check
        for record in records
        for check in record.get("checks", [])
        if not check["ok"]
    ] + [check for check in summary["cross_cell_checks"] if not check["ok"]]
    print(f"\nSummary written to {summary_path}")
    print(f"Cells run: {len(records)}, failed checks: {len(failures)}")
    for check in failures:
        print(f"  FAIL: {check['check']} -> {check['observed']}")
    print("Now listen to the WAVs under", out_root)


if __name__ == "__main__":
    main()
