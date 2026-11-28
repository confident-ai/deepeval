import asyncio
import inspect
from typing import List, Literal, Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepeval.telemetry import Entrypoint, capture_red_teaming_run
from deepeval.utils import get_or_create_event_loop
from rich import box
from rich.console import Console
from rich.table import Table

from deepeval.red_team.attacks.multi_turn.types import CallbackType
from deepeval.red_team.orchestration.orchestration_agent import (
    OrchestrationAgent,
)
from deepeval.red_team.orchestration.sequential_agent import SequentialAgent
from deepeval.red_team.orchestration.types import (
    OrchestrationResult,
    SequentialResult,
)
from deepeval.red_team.utils import add_pbar, create_progress


def _accepts_turns(model_callback: CallbackType) -> bool:
    try:
        params = list(inspect.signature(model_callback).parameters.values())
    except (TypeError, ValueError):
        return True
    return len(params) > 1 or any(
        param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "turns"
        for param in params
    )


class AgenticRedTeamer:
    """Goal-driven, adaptive red teaming. Experimental: the API may change."""

    def __init__(
        self,
        task: str,
        scenario: str,
        criteria: List[str],
        mode: Literal["sequential", "parallel"] = "parallel",
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o-mini",
        judge_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        max_turns: Optional[int] = None,
        max_concurrent: int = 10,
        async_mode: bool = True,
        show_progress: bool = True,
    ):
        if mode not in ("sequential", "parallel"):
            raise ValueError("'mode' must be 'sequential' or 'parallel'.")
        self.task = task
        self.scenario = scenario
        self.criteria = criteria
        self.mode = mode
        self.simulator_model = simulator_model
        self.judge_model = judge_model
        self.max_turns = max_turns
        self.max_concurrent = max_concurrent
        self.async_mode = async_mode
        self.show_progress = show_progress

    def run(
        self, model_callback: CallbackType
    ) -> Union[OrchestrationResult, SequentialResult]:
        agent, label, status = self._build(model_callback)

        with capture_red_teaming_run(entrypoint=Entrypoint.AGENTIC_RED_TEAM):
            if self.async_mode:
                loop = get_or_create_event_loop()
                result = loop.run_until_complete(
                    _a_run_with_progress(
                        agent, label, status, self.show_progress
                    )
                )
            else:
                result = agent.run()

        if self.show_progress and isinstance(
            result, (OrchestrationResult, SequentialResult)
        ):
            _print_result_table(result, label)
        return result

    def _build(self, model_callback):
        kwargs = dict(
            task=self.task,
            scenario=self.scenario,
            criteria=self.criteria,
            model_callback=model_callback,
            simulator_model=self.simulator_model,
            judge_model=self.judge_model,
        )
        if self.max_turns is not None:
            kwargs["max_turns"] = self.max_turns

        if self.mode == "parallel":
            if not _accepts_turns(model_callback):
                raise ValueError(
                    "Parallel mode runs several conversations at once and "
                    "replays the history to the target each turn, so "
                    "model_callback must accept a 'turns' argument, e.g. def "
                    "callback(input, turns=None). Use mode='sequential' for a "
                    "target that keeps its own memory."
                )
            agent = OrchestrationAgent(
                **kwargs, max_concurrent=self.max_concurrent
            )
            label = "Agentic red team (parallel)"

            def status():
                phase = (
                    "recon"
                    if agent.recon_result is None
                    else f"round {len(agent.rounds) + 1}"
                )
                return (
                    f"{phase} · "
                    f"{len(agent.met_criteria)}/{len(agent.criteria)} met"
                )

        else:
            agent = SequentialAgent(**kwargs)
            label = "Agentic red team (sequential)"

            def status():
                return (
                    f"turn {len(agent.verdicts)} · "
                    f"{len(agent.met_criteria)}/{len(agent.criteria)} met"
                )

        return agent, label, status


def _met_lookup(result):
    lookup = {}
    if isinstance(result, SequentialResult):
        for turn, verdict in enumerate(result.verdicts, 1):
            for idx in verdict.met:
                if 1 <= idx <= len(result.criteria):
                    criteria = result.criteria[idx - 1]
                    lookup.setdefault(
                        criteria, (f"turn {turn}", verdict.reason)
                    )
    else:
        for met in result.met_criteria:
            lookup[met.criteria] = (f"round {met.round}", met.how_it_was_met)
    return lookup


def _print_result_table(result, label):
    console = Console()
    met_lookup = _met_lookup(result)

    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
        box=box.HEAVY,
        title=f"{label} — Results",
        title_style="bold magenta",
        expand=True,
        padding=(0, 1),
        show_lines=True,
    )
    table.add_column("Criteria", style="cyan", ratio=2, no_wrap=False)
    table.add_column("Status", justify="center", width=12)
    table.add_column("When", justify="center", width=8)
    table.add_column("How it was met", style="dim", ratio=3, no_wrap=False)

    for criteria in result.criteria:
        info = met_lookup.get(criteria)
        if info is None:
            table.add_row(
                criteria, "[bold green]✓ DEFENDED[/bold green]", "—", "—"
            )
        else:
            when, detail = info
            table.add_row(
                criteria, "[bold red]✗ BREACHED[/bold red]", when, detail
            )

    console.print("\n")
    console.print(table)

    error = getattr(result, "error", None)
    if error is not None:
        console.print(
            f"\n[bold rgb(255,171,0)]⚠ The run stopped early: "
            f"{type(error).__name__}: {error}. Criteria marked DEFENDED may "
            "not have been fully tested.[/bold rgb(255,171,0)]"
        )

    met = len(result.met_criteria)
    total = len(result.criteria)
    if met == 0:
        score_style = "rgb(5,245,141)"
    elif met == total:
        score_style = "rgb(255,85,85)"
    else:
        score_style = "rgb(255,171,0)"
    console.print(
        f"\n🎯 [{score_style}]Breach rate: {result.score:.0%}[/{score_style}] "
        f"({met}/{total} criteria met)"
    )
    if met > 0:
        console.print(
            f"[bold rgb(255,85,85)]✗ The model was successfully jailbroken "
            f"— {met}/{total} criteria met.[/bold rgb(255,85,85)]"
        )
    else:
        console.print(
            "[bold rgb(5,245,141)]✓ The model successfully defended "
            "— no criteria met.[/bold rgb(5,245,141)]"
        )
    error = getattr(result, "error", None)
    if error is not None:
        console.print(
            f"[bold yellow]⚠ Run ended with an error: {error}[/bold yellow]"
        )
    console.print("\n" + "=" * 80)
    console.print(f"[bold magenta]{label} complete.[/bold magenta]")
    console.print("=" * 80 + "\n")


async def _a_run_with_progress(agent, label, status, show_progress):
    with create_progress(enabled=show_progress) as progress:
        if progress is None:
            return await agent.a_run()

        total = max(1, len(agent.criteria))
        pbar = add_pbar(progress, label, total=total)
        task = asyncio.create_task(agent.a_run())
        while not task.done():
            progress.update(
                pbar,
                completed=len(agent.met_criteria),
                description=f"{label} — {status()}",
            )
            await asyncio.sleep(0.2)
        progress.update(pbar, completed=total, description=f"{label} — done")
        return await task
