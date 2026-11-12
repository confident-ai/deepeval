from dataclasses import replace
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.benchmarks.voice_skills.journey import (
    Journey,
    build_golden,
    build_graph,
    build_metrics,
    closed_without_the_work,
    load_journeys,
    settle_ending,
    stop_when_caller_is_done,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.simulator import ConversationSimulator
from deepeval.simulator.controller import proceed
from deepeval.telemetry import capture_benchmark_run
from deepeval.test_case import ConversationalTestCase
from deepeval.voice import VoiceConfig
from deepeval.voice.connectors import BaseVoiceConnector


class VoiceSkillsResult(BaseModel):
    overall_accuracy: float
    journey_scores: Dict[str, bool]
    skill_scores: Dict[str, float]


class VoiceSkills(DeepEvalBaseBenchmark):
    """Recorded callers ring a voice agent about one real task each.

    Every caller is a graph: a mistake by the agent sends the call down a
    harder path, and repeated mistakes end it. A journey passes when every
    metric that applies to it passes on every run.

    Args:
        dataset_path: The folder holding `scenarios/` and `audio/`.
        journeys: The ids of the journeys to run. Defaulted to all of them.
        n_runs: Calls made per journey. Defaulted to 3.
        simulator_model: Routes the caller through the graph.
        judge_model: Grades the skills that need judgement.
        audio_judge_model: Grades voice style; has to take audio.
        programmatic_only: Leave out every metric that uses an LLM.
        max_user_simulations: The most caller turns in one call.
    """

    def __init__(
        self,
        dataset_path: str,
        journeys: Optional[List[str]] = None,
        n_runs: int = 3,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        judge_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        audio_judge_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        programmatic_only: bool = False,
        max_user_simulations: int = 30,
    ):
        if n_runs < 1:
            raise ValueError("`n_runs` must be at least 1.")
        # The journeys are files on disk, so the Hugging Face `datasets`
        # package the base class loads is not needed.
        self.tasks = []
        self.dataset = None
        self.dataset_path = dataset_path
        self.journey_ids = journeys
        self.n_runs = n_runs
        self.simulator_model = simulator_model
        self.judge_model = judge_model
        self.audio_judge_model = audio_judge_model
        self.programmatic_only = programmatic_only
        self.max_user_simulations = max_user_simulations
        self.journeys: List[Journey] = self.load_benchmark_dataset()
        self.predictions = None
        self.test_cases: List[ConversationalTestCase] = []
        self.journey_scores: Optional[Dict[str, bool]] = None
        self.skill_scores: Optional[Dict[str, float]] = None
        self.overall_score: Optional[float] = None

    def load_benchmark_dataset(self) -> List[Journey]:
        journeys = load_journeys(self.dataset_path)
        if self.journey_ids is None:
            return journeys
        known = {journey["id"] for journey in journeys}
        unknown = set(self.journey_ids) - known
        if unknown:
            raise ValueError(f"Unknown journeys: {', '.join(sorted(unknown))}.")
        return [
            journey for journey in journeys if journey["id"] in self.journey_ids
        ]

    def evaluate(
        self,
        voice_config: Optional[VoiceConfig] = None,
        agent: Optional[Callable[[Journey], BaseVoiceConnector]] = None,
    ) -> VoiceSkillsResult:
        """Run every journey against one agent.

        Pass `voice_config` for an agent that is already deployed, or `agent`
        to build one per journey, which is what a journey's own policy,
        records and tools need.
        """
        import pandas as pd

        if voice_config is None and agent is None:
            raise ValueError("Pass `voice_config` or `agent`.")

        with capture_benchmark_run("VoiceSkills", len(self.journeys)):
            rows = []
            for journey in self.journeys:
                for run in range(1, self.n_runs + 1):
                    config = voice_config
                    if agent is not None:
                        config = (
                            replace(voice_config, connector=agent(journey))
                            if voice_config is not None
                            else VoiceConfig(connector=agent(journey))
                        )
                    rows += self._call(journey, run, config)

            self.predictions = pd.DataFrame(rows)
            graded = self.predictions[self.predictions["passed"].notna()]
            self.journey_scores = {
                journey_id: bool(group["passed"].all())
                for journey_id, group in graded.groupby("journey")
            }
            self.skill_scores = {
                skill: float(group["passed"].mean())
                for skill, group in graded.groupby("skill")
            }
            self.overall_score = (
                sum(self.journey_scores.values()) / len(self.journey_scores)
                if self.journey_scores
                else 0.0
            )
            print(f"Overall Voice Skills pass rate: {self.overall_score}")
            return VoiceSkillsResult(
                overall_accuracy=self.overall_score,
                journey_scores=self.journey_scores,
                skill_scores=self.skill_scores,
            )

    def _call(
        self, journey: Journey, run: int, voice_config: VoiceConfig
    ) -> List[Dict]:
        """Make one call and grade it, one row per skill."""
        simulator = ConversationSimulator(
            voice_config=voice_config,
            simulation_graph=build_graph(journey, self.dataset_path),
            # The graph ends the call, or the caller saying goodbye does; no
            # judge reads the transcript to decide.
            stopping_controller=stop_when_caller_is_done,
            simulator_model=self.simulator_model,
        )
        test_case = simulator.simulate(
            [build_golden(journey, self.dataset_path)],
            max_user_simulations=self.max_user_simulations,
        )[0]
        metadata = test_case.metadata or {}
        settle_ending(test_case)
        metadata = test_case.metadata or {}
        scripted = (journey.get("shape") or {}).get("shortest_path")
        if scripted and "Shortest path" in metadata:
            # Measured on the script as written, not on the shortcuts a
            # caller can take when asked things out of order.
            metadata["Shortest path"] = scripted
        usage = getattr(voice_config.connector, "usage", None)
        if usage:
            metadata["Agent usage"] = dict(usage)
            test_case.metadata = metadata
        self.test_cases.append(test_case)

        rows = []
        metrics = build_metrics(
            journey,
            self.judge_model,
            self.audio_judge_model,
            self.programmatic_only,
        )
        for skill, metric in metrics.items():
            row = {
                "journey": journey["id"],
                "run": run,
                "skill": skill,
                "metric": metric.__name__,
                "ending": metadata.get("Simulation ending"),
                "score": None,
                "passed": None,
                "reason": None,
            }
            try:
                metric.measure(test_case, _show_indicator=False)
                row["score"] = metric.score
                row["reason"] = metric.reason
                # A skipped metric says nothing about the skill either way.
                if metric.score is not None:
                    row["passed"] = bool(metric.is_successful())
            except Exception as error:
                row["reason"] = f"Could not be graded: {error}"
            rows.append(row)
        tools = next(
            (row["score"] for row in rows if row["skill"] == "entity_extraction"),
            None,
        )
        if closed_without_the_work(test_case, tools):
            for row in rows:
                if row["skill"] == "task_completion":
                    row.update(
                        score=0.0,
                        passed=False,
                        reason="The call was closed before every expected "
                        "tool call was made.",
                    )
        return rows
