"""The context managers call sites use.

Each opens a scope, and -- for evaluations -- captures on the way out so the
totals and the outcome are known. The trade-off is that a hard crash mid-run
loses the event, where the old code sent a bare one up front.
"""

import uuid
from contextlib import contextmanager
from dataclasses import replace
from typing import (
    Any,
    Container,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Set,
)

from deepeval.telemetry.client import (
    capture,
    register_integration,
    telemetry_opt_out,
)
from deepeval.telemetry.context import (
    RunAccumulator,
    pop_run,
    push_run,
    record_guardrail,
)
from deepeval.telemetry.events import Entrypoint, Event, Feature
from deepeval.telemetry.identity import (
    get_feature_status,
    get_last_feature,
    set_last_feature,
)
from deepeval.telemetry.properties import (
    CUSTOM_PROVIDER,
    UNKNOWN_CLI_COMMAND,
    EventProperties,
    LoginMethod,
    LoginOutcome,
    LoginPromptSurface,
    Outcome,
)
from deepeval.tracing.integrations import Integration

_ENTRYPOINT_FEATURE = {
    Entrypoint.EVALUATE: Feature.EVALUATION,
    Entrypoint.EVALS_ITERATOR: Feature.COMPONENT_EVALUATION,
    Entrypoint.PYTEST: Feature.EVALUATION,
    Entrypoint.COMPARE: Feature.EVALUATION,
    Entrypoint.STANDALONE: Feature.EVALUATION,
}


@contextmanager
def capture_evaluation_run(
    entrypoint: Entrypoint,
    run_id: Optional[str] = None,
    skip_if_empty: bool = False,
) -> Iterator[RunAccumulator]:
    """Open a run scope and emit exactly one `Evaluation` on the way out.

    `run_id` identifies the logical run. It is generated per scope unless a
    caller supplies one, which is how the pytest workers of a single
    `-n` session -- separate processes, each with their own counters -- report
    as one run rather than as several.

    `skip_if_empty` suppresses the event when the scope saw no deepeval work,
    for scopes that wrap something broader than an evaluation.
    """
    if telemetry_opt_out():
        yield RunAccumulator(entrypoint=entrypoint)
        return

    feature = _ENTRYPOINT_FEATURE.get(entrypoint, Feature.EVALUATION)
    feature_status = get_feature_status(feature)
    accumulator, token = push_run(entrypoint, run_id or str(uuid.uuid4()))

    outcome = Outcome.COMPLETED
    error_type: Optional[str] = None
    try:
        yield accumulator
    except KeyboardInterrupt:
        outcome = Outcome.INTERRUPTED
        raise
    except BaseException as error:
        outcome = Outcome.ERRORED
        # Class name only. Messages routinely carry prompts, keys, org names.
        error_type = type(error).__name__
        raise
    finally:
        pop_run(token)
        try:
            # Never `return` from this block: that would discard an exception
            # on its way out of the user's evaluation.
            if accumulator.has_activity() or not skip_if_empty:
                set_last_feature(feature)
                capture(
                    Event.EVALUATION,
                    replace(
                        accumulator.snapshot(),
                        outcome=outcome,
                        error_type=error_type,
                        feature_name=feature,
                        feature_status=feature_status,
                    ),
                )
        except Exception:
            pass


@contextmanager
def capture_synthesizer_run(
    method: str,
    max_generations: Optional[int],
    num_evolutions: int,
    evolutions: Dict[Any, Any],
) -> Iterator[None]:
    if telemetry_opt_out():
        yield
        return
    feature_status = get_feature_status(Feature.SYNTHESIZER)
    set_last_feature(Feature.SYNTHESIZER)
    capture(
        Event.SYNTHESIZER,
        EventProperties(
            feature_name=Feature.SYNTHESIZER,
            feature_status=feature_status,
            synth_method=method,
            synth_max_generations=max_generations,
            synth_num_evolutions=num_evolutions,
            synth_evolutions=sorted(
                getattr(evolution, "value", str(evolution))
                for evolution in (evolutions or {})
            ),
        ),
    )
    yield


@contextmanager
def capture_conversation_simulator_run(
    num_conversations: int,
) -> Iterator[None]:
    if telemetry_opt_out():
        yield
        return
    feature_status = get_feature_status(Feature.CONVERSATION_SIMULATOR)
    set_last_feature(Feature.CONVERSATION_SIMULATOR)
    capture(
        Event.CONVERSATION_SIMULATOR,
        EventProperties(
            feature_name=Feature.CONVERSATION_SIMULATOR,
            feature_status=feature_status,
            num_conversations=num_conversations,
        ),
    )
    yield


@contextmanager
def capture_benchmark_run(benchmark: str, num_tasks: Any) -> Iterator[None]:
    if telemetry_opt_out():
        yield
        return
    feature_status = get_feature_status(Feature.BENCHMARK)
    set_last_feature(Feature.BENCHMARK)
    capture(
        Event.BENCHMARK,
        EventProperties(
            feature_name=Feature.BENCHMARK,
            feature_status=feature_status,
            benchmark_name=benchmark,
            benchmark_num_tasks=(
                num_tasks if isinstance(num_tasks, int) else None
            ),
        ),
    )
    yield


# Classes under `deepeval.` whose instances carry user-written names: a
# vulnerability defined inline, or a framework pulled from Confident AI.
_USER_DEFINED_RED_TEAM_CLASSES = frozenset(
    {"CustomVulnerability", "RedTeamingFramework"}
)


def _red_team_component_name(component: Any) -> str:
    """A bounded name for a vulnerability, attack, framework or guard.

    Same invariant as `describe_judge`: only classes defined under `deepeval.`
    are named, anything else collapses to `custom`.
    """
    try:
        cls = type(component)
        if not (cls.__module__ or "").startswith("deepeval."):
            return CUSTOM_PROVIDER
        if cls.__name__ in _USER_DEFINED_RED_TEAM_CLASSES:
            return CUSTOM_PROVIDER
        return cls.__name__
    except Exception:
        return CUSTOM_PROVIDER


@contextmanager
def capture_red_teaming_run(
    vulnerabilities: Optional[Iterable[Any]] = None,
    attacks: Optional[Iterable[Any]] = None,
    framework: Optional[Any] = None,
    entrypoint: Entrypoint = Entrypoint.RED_TEAM,
) -> Iterator[RunAccumulator]:
    """Open a run scope and emit exactly one `Red Teaming` on the way out.

    The same shape as `capture_evaluation_run`: red teaming metrics measured
    inside the scope count towards this run rather than the standalone path,
    and test cases are recorded with `record_test_case`. Takes the objects
    rather than their names so user-written names never reach the payload.
    """
    if telemetry_opt_out():
        yield RunAccumulator(entrypoint=entrypoint)
        return

    feature_status = get_feature_status(Feature.REDTEAMING)
    vulnerability_names = sorted(
        {_red_team_component_name(v) for v in vulnerabilities or []}
    )
    attack_names = sorted({_red_team_component_name(a) for a in attacks or []})
    framework_name = (
        _red_team_component_name(framework) if framework is not None else None
    )
    accumulator, token = push_run(entrypoint, str(uuid.uuid4()))

    outcome = Outcome.COMPLETED
    error_type: Optional[str] = None
    try:
        yield accumulator
    except KeyboardInterrupt:
        outcome = Outcome.INTERRUPTED
        raise
    except BaseException as error:
        outcome = Outcome.ERRORED
        # Class name only. Messages routinely carry prompts, keys, org names.
        error_type = type(error).__name__
        raise
    finally:
        pop_run(token)
        try:
            # Never `return` from this block: that would discard an exception
            # on its way out of the user's red teaming run.
            set_last_feature(Feature.REDTEAMING)
            capture(
                Event.RED_TEAMING,
                replace(
                    accumulator.snapshot(),
                    outcome=outcome,
                    error_type=error_type,
                    feature_name=Feature.REDTEAMING,
                    feature_status=feature_status,
                    rt_vulnerabilities=vulnerability_names,
                    rt_attacks=attack_names,
                    rt_framework=framework_name,
                ),
            )
        except Exception:
            pass


def record_guard(guard: Any, async_mode: bool) -> None:
    """One guard check, the guardrail equivalent of a metric's `start_measure`.

    Batched into `Guardrails` events by the same threshold, interval and
    process-exit policy as standalone metrics.
    """
    record_guardrail(
        _red_team_component_name(guard),
        async_mode=async_mode,
        model=getattr(guard, "model", None),
    )


def record_tracing_integration(integration: Integration) -> None:
    """Fires once per process, not once per handler construction."""
    first_time = register_integration(integration.value)
    if telemetry_opt_out() or not first_time:
        return

    feature_status = get_feature_status(Feature.TRACING_INTEGRATION)
    set_last_feature(Feature.TRACING_INTEGRATION)
    capture(
        Event.INTEGRATION_INSTALLED,
        EventProperties(
            feature_name=Feature.TRACING_INTEGRATION,
            feature_status=feature_status,
            integration=integration.value,
        ),
    )


@contextmanager
def capture_tracing_integration(integration: Integration) -> Iterator[None]:
    record_tracing_integration(integration)
    yield


def capture_cli_command(
    command: Optional[str], known_commands: Container[str]
) -> None:
    """Command name only, never arguments or flag values.

    `known_commands` is the dispatch table of the Click group that just routed
    this invocation, so the set of possible values is defined by the CLI
    itself. Registering a command is the only step; there is no second list
    here to keep in sync.
    """
    if telemetry_opt_out():
        return
    name = (
        command
        if command and command in known_commands
        else UNKNOWN_CLI_COMMAND
    )
    capture(Event.CLI_COMMAND, EventProperties(cli_command=name))


def capture_login_prompt_shown(surface: LoginPromptSurface) -> None:
    if telemetry_opt_out():
        return
    capture(Event.LOGIN_PROMPT_SHOWN, EventProperties(prompt_surface=surface))


_surfaces_seen: Set[LoginPromptSurface] = set()


def capture_login_prompt_shown_once(surface: LoginPromptSurface) -> None:
    """For surfaces that re-render continuously, such as a TUI pane.

    Counting every repaint would measure scrolling, not exposure.
    """
    if surface in _surfaces_seen:
        return
    _surfaces_seen.add(surface)
    capture_login_prompt_shown(surface)


def record_login_completed(surface: LoginPromptSurface) -> None:
    """A login that happened outside `deepeval login`, e.g. inline in `view`."""
    if telemetry_opt_out():
        return
    capture(
        Event.LOGIN,
        EventProperties(
            login_outcome=LoginOutcome.COMPLETED,
            prompt_surface=surface,
            last_feature=get_last_feature(),
        ),
    )


class LoginSpan:
    """Lets the login flow report how it actually ended, and by which path."""

    def __init__(self) -> None:
        self.outcome = LoginOutcome.ABANDONED
        self.method = LoginMethod.UNKNOWN

    def set_outcome(self, outcome: LoginOutcome) -> None:
        self.outcome = outcome

    def set_method(self, method: LoginMethod) -> None:
        self.method = method


@contextmanager
def capture_login_event() -> Iterator[LoginSpan]:
    """Captures on exit with the real outcome.

    The old version entered before any user interaction and hardcoded
    `completed: True`, so an abandoned login was indistinguishable from a
    successful one.
    """
    span = LoginSpan()
    if telemetry_opt_out():
        yield span
        return

    last_feature = get_last_feature()
    try:
        yield span
    except BaseException:
        if span.outcome is LoginOutcome.ABANDONED:
            span.outcome = LoginOutcome.FAILED
        raise
    finally:
        try:
            capture(
                Event.LOGIN,
                EventProperties(
                    login_outcome=span.outcome,
                    login_method=span.method,
                    last_feature=last_feature,
                ),
            )
        except Exception:
            pass
