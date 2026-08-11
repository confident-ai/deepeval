"""Property keys, closed value sets, and the typed payload.

Keys use lowercase dotted namespaces. That is a convention rather than a
dependency -- it happens to match OpenTelemetry's attribute style, so a future
bridge is mechanical, and it groups related properties in any vendor's UI.
"""

from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, List, Optional, Union

from deepeval.telemetry.events import Entrypoint, Feature

PropValue = Union[str, int, float, bool, List[str]]


class Prop(str, Enum):
    # telemetry meta
    SCHEMA_VERSION = "telemetry.schema_version"
    SDK_LANGUAGE = "sdk.language"
    SDK_VERSION = "deepeval.version"
    RUNTIME = "runtime.kind"
    # identity
    USER_STATUS = "user.status"
    USER_ID = "user.unique_id"
    LOGGED_IN = "user.logged_in"
    # evaluation
    ENTRYPOINT = "eval.entrypoint"
    RUN_ID = "eval.run_id"
    OUTCOME = "eval.outcome"
    ERROR_TYPE = "eval.error_type"
    TEST_CASE_COUNT = "eval.test_case_count"
    GOLDEN_COUNT = "eval.golden_count"
    TURN_KIND = "eval.turn_kind"
    METRIC_RUNS = "eval.metric_runs"
    METRICS = "eval.metrics"
    METRICS_COUNT = "eval.metrics_count"
    ASYNC_MODE = "eval.async_mode"
    IN_COMPONENT = "eval.in_component"
    FLUSH_REASON = "eval.flush_reason"
    # judge
    PROVIDER = "judge.provider"
    MODEL = "judge.model"
    # tracing
    TRACING_ENABLED = "tracing.enabled"
    TRACED = "tracing.traced"
    TRACE_COUNT = "tracing.trace_count"
    INTEGRATION = "tracing.integration"
    INTEGRATIONS = "tracing.integrations"
    INTEGRATIONS_COUNT = "tracing.integrations_count"
    INTEGRATIONS_PRIMARY = "tracing.integrations_primary"
    # feature adoption
    FEATURE_NAME = "feature.name"
    FEATURE_STATUS = "feature.status"
    LAST_FEATURE = "feature.last"
    # cli and login funnel
    CLI_COMMAND = "cli.command"
    PROMPT_SURFACE = "login.surface"
    LOGIN_OUTCOME = "login.outcome"
    LOGIN_METHOD = "login.method"
    # synthesizer
    SYNTH_METHOD = "synthesizer.method"
    SYNTH_MAX_GENERATIONS = "synthesizer.max_generations"
    SYNTH_NUM_EVOLUTIONS = "synthesizer.num_evolutions"
    SYNTH_EVOLUTIONS = "synthesizer.evolutions"
    # benchmark
    BENCHMARK_NAME = "benchmark.name"
    BENCHMARK_NUM_TASKS = "benchmark.num_tasks"
    # conversation simulator
    NUM_CONVERSATIONS = "simulator.num_conversations"


class Language(str, Enum):
    """Which SDK emitted the event. An absent value means Python."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"


class Runtime(str, Enum):
    CI_GITHUB = "ci_github"
    CI_GITLAB = "ci_gitlab"
    CI_OTHER = "ci_other"
    CONTAINER = "container"
    NOTEBOOK = "notebook"
    INTERACTIVE = "interactive"
    SCRIPT = "script"


class UserStatus(str, Enum):
    NEW = "new"
    OLD = "old"


class Outcome(str, Enum):
    COMPLETED = "completed"
    ERRORED = "errored"
    INTERRUPTED = "interrupted"


class TurnKind(str, Enum):
    """Whether a run evaluated single-turn or multi-turn items.

    `MIXED` is not reachable from one `evaluate()` call, which rejects a mixed
    list, but a `deepeval test run` session accumulates across many
    `assert_test` calls that are each validated on their own.
    """

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    MIXED = "mixed"


class FlushReason(str, Enum):
    """Why a standalone metric batch was sent.

    Needed because a threshold or interval flush is a partial session, so
    `eval.metric_runs` must be summed rather than counted across them.
    """

    THRESHOLD = "threshold"
    INTERVAL = "interval"
    PROCESS_EXIT = "process_exit"
    MANUAL = "manual"


# `judge.provider` is the judge model's class name, read off the class itself
# rather than restated in an enum here. Bounding it needs an invariant, not a
# list: only classes defined under `deepeval.` are emitted, and anything else
# collapses to `custom`. See `judge.py`.
CUSTOM_PROVIDER = "custom"

# Emitted instead of a model name whenever the name is not in the in-repo model
# registries, so user-defined model names cannot leak.
UNKNOWN_MODEL = "other"

# `cli.command` is validated against the Click group that dispatched it, so an
# unrecognised name can only mean the callback ran outside normal dispatch.
UNKNOWN_CLI_COMMAND = "unknown"


class LoginPromptSurface(str, Enum):
    POST_EVAL = "post_eval"
    POST_ARENA = "post_arena"
    CLI_VIEW = "cli_view"
    CLI_VIEW_NO_RUN = "cli_view_no_run"
    INSPECT_TUI = "inspect_tui"


class LoginOutcome(str, Enum):
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "failed"


class LoginMethod(str, Enum):
    """Only the browser flow returns an email, so it is the only path that can
    attach an identity. The others are still worth telling apart."""

    BROWSER = "browser"
    PASTE = "paste"
    API_KEY_FLAG = "api_key_flag"
    UNKNOWN = "unknown"


class TelemetryKey(str, Enum):
    """Keys of the on-disk identity file.

    Replaces the f-string-built `DEEPEVAL_{feature}_STATUS` pattern; feature
    status now lives under a single key holding a `feature=status` list.
    """

    ID = "DEEPEVAL_ID"
    STATUS = "DEEPEVAL_STATUS"
    LAST_FEATURE = "DEEPEVAL_LAST_FEATURE"
    SEEN_FEATURES = "DEEPEVAL_SEEN_FEATURES"
    LOGGED_IN_WITH = "LOGGED_IN_WITH"


@dataclass(frozen=True)
class EventProperties:
    """A typed payload. `to_dict` is the only enum-to-str conversion point."""

    # telemetry meta
    schema_version: Optional[int] = None
    sdk_language: Optional[Language] = None
    sdk_version: Optional[str] = None
    runtime: Optional[Runtime] = None
    # identity
    user_status: Optional[UserStatus] = None
    user_id: Optional[str] = None
    logged_in: Optional[bool] = None
    # evaluation
    entrypoint: Optional[Entrypoint] = None
    run_id: Optional[str] = None
    outcome: Optional[Outcome] = None
    error_type: Optional[str] = None
    test_case_count: Optional[int] = None
    golden_count: Optional[int] = None
    turn_kind: Optional[TurnKind] = None
    metric_runs: Optional[int] = None
    metrics: Optional[List[str]] = None
    metrics_count: Optional[int] = None
    async_mode: Optional[bool] = None
    in_component: Optional[bool] = None
    flush_reason: Optional[FlushReason] = None
    # judge
    provider: Optional[str] = None
    model: Optional[str] = None
    # tracing
    tracing_enabled: Optional[bool] = None
    traced: Optional[bool] = None
    trace_count: Optional[int] = None
    integration: Optional[str] = None
    integrations: Optional[List[str]] = None
    integrations_count: Optional[int] = None
    integrations_primary: Optional[str] = None
    # feature adoption
    feature_name: Optional[Feature] = None
    feature_status: Optional[UserStatus] = None
    last_feature: Optional[Feature] = None
    # cli and login funnel
    cli_command: Optional[str] = None
    prompt_surface: Optional[LoginPromptSurface] = None
    login_outcome: Optional[LoginOutcome] = None
    login_method: Optional[LoginMethod] = None
    # synthesizer
    synth_method: Optional[str] = None
    synth_max_generations: Optional[int] = None
    synth_num_evolutions: Optional[int] = None
    synth_evolutions: Optional[List[str]] = None
    # benchmark
    benchmark_name: Optional[str] = None
    benchmark_num_tasks: Optional[int] = None
    # conversation simulator
    num_conversations: Optional[int] = None

    def to_dict(self) -> Dict[str, PropValue]:
        payload: Dict[str, PropValue] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            payload[_FIELD_TO_PROP[field.name].value] = (
                value.value if isinstance(value, Enum) else value
            )
        return payload

    def merged_with(self, other: "EventProperties") -> Dict[str, PropValue]:
        """Base properties overlaid with event-specific ones."""
        return {**self.to_dict(), **other.to_dict()}


_FIELD_TO_PROP: Dict[str, Prop] = {
    "schema_version": Prop.SCHEMA_VERSION,
    "sdk_language": Prop.SDK_LANGUAGE,
    "sdk_version": Prop.SDK_VERSION,
    "runtime": Prop.RUNTIME,
    "user_status": Prop.USER_STATUS,
    "user_id": Prop.USER_ID,
    "logged_in": Prop.LOGGED_IN,
    "entrypoint": Prop.ENTRYPOINT,
    "run_id": Prop.RUN_ID,
    "outcome": Prop.OUTCOME,
    "error_type": Prop.ERROR_TYPE,
    "test_case_count": Prop.TEST_CASE_COUNT,
    "golden_count": Prop.GOLDEN_COUNT,
    "turn_kind": Prop.TURN_KIND,
    "metric_runs": Prop.METRIC_RUNS,
    "metrics": Prop.METRICS,
    "metrics_count": Prop.METRICS_COUNT,
    "async_mode": Prop.ASYNC_MODE,
    "in_component": Prop.IN_COMPONENT,
    "flush_reason": Prop.FLUSH_REASON,
    "provider": Prop.PROVIDER,
    "model": Prop.MODEL,
    "tracing_enabled": Prop.TRACING_ENABLED,
    "traced": Prop.TRACED,
    "trace_count": Prop.TRACE_COUNT,
    "integration": Prop.INTEGRATION,
    "integrations": Prop.INTEGRATIONS,
    "integrations_count": Prop.INTEGRATIONS_COUNT,
    "integrations_primary": Prop.INTEGRATIONS_PRIMARY,
    "feature_name": Prop.FEATURE_NAME,
    "feature_status": Prop.FEATURE_STATUS,
    "last_feature": Prop.LAST_FEATURE,
    "cli_command": Prop.CLI_COMMAND,
    "prompt_surface": Prop.PROMPT_SURFACE,
    "login_outcome": Prop.LOGIN_OUTCOME,
    "login_method": Prop.LOGIN_METHOD,
    "synth_method": Prop.SYNTH_METHOD,
    "synth_max_generations": Prop.SYNTH_MAX_GENERATIONS,
    "synth_num_evolutions": Prop.SYNTH_NUM_EVOLUTIONS,
    "synth_evolutions": Prop.SYNTH_EVOLUTIONS,
    "benchmark_name": Prop.BENCHMARK_NAME,
    "benchmark_num_tasks": Prop.BENCHMARK_NUM_TASKS,
    "num_conversations": Prop.NUM_CONVERSATIONS,
}
