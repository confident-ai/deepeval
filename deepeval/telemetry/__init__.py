"""deepeval telemetry.

Events model what the user did, not which code path fired. One `Evaluation`
event carries an `entrypoint` dimension rather than five event names, counters
are accumulated in memory and flushed once per run, and every property key and
value comes from an enum.

Opt out with `DEEPEVAL_TELEMETRY_OPT_OUT=1`.
"""

import os

from deepeval.constants import HIDDEN_DIR, KEY_FILE
from deepeval.telemetry.api import (
    LoginSpan,
    capture_benchmark_run,
    capture_cli_command,
    capture_conversation_simulator_run,
    capture_evaluation_run,
    capture_login_event,
    capture_login_prompt_shown,
    capture_login_prompt_shown_once,
    capture_synthesizer_run,
    capture_tracing_integration,
    record_login_completed,
    record_tracing_integration,
)
from deepeval.telemetry.client import (
    NoopBackend,
    PostHogBackend,
    TelemetryBackend,
    base_properties,
    blocked_by_firewall,
    capture,
    flush,
    get_backend,
    installed_integrations,
    register_integration,
    set_backend,
    telemetry_opt_out,
)
from deepeval.telemetry.context import (
    RunAccumulator,
    current_run,
    flush_standalone_metrics,
    record_golden,
    record_metric,
    record_test_case,
    turn_kind_of,
)
from deepeval.telemetry.events import (
    TELEMETRY_RUN_ID_ENV_VAR,
    TELEMETRY_SCHEMA_VERSION,
    Entrypoint,
    Event,
    Feature,
)
from deepeval.telemetry.identity import (
    TELEMETRY_DATA_FILE,
    Identity,
    get_feature_status,
    get_identity,
    get_last_feature,
    get_logged_in_with,
    get_status,
    get_unique_id,
    read_telemetry_file,
    set_last_feature,
    set_logged_in_with,
    telemetry_path,
    write_telemetry_file,
)
from deepeval.telemetry.judge import describe_judge
from deepeval.telemetry.properties import (
    EventProperties,
    FlushReason,
    LoginMethod,
    LoginOutcome,
    LoginPromptSurface,
    Outcome,
    Prop,
    Runtime,
    TelemetryKey,
    TurnKind,
    UserStatus,
)
from deepeval.telemetry.runtime import detect_runtime


def _migrate_project_files() -> None:
    """Move stray project files into the hidden dir. Predates this package."""
    if telemetry_opt_out():
        return
    try:
        if os.path.exists(KEY_FILE) and not os.path.isdir(HIDDEN_DIR):
            temp_name = ".deepeval_temp"
            os.rename(KEY_FILE, temp_name)
            os.makedirs(HIDDEN_DIR, exist_ok=True)
            os.rename(temp_name, os.path.join(HIDDEN_DIR, KEY_FILE))

        os.makedirs(HIDDEN_DIR, exist_ok=True)

        for name in (".deepeval-cache.json", ".temp_test_run_data.json"):
            if os.path.exists(name):
                os.rename(name, os.path.join(HIDDEN_DIR, name))
    except OSError:
        pass


_migrate_project_files()


__all__ = [
    "TELEMETRY_DATA_FILE",
    "TELEMETRY_RUN_ID_ENV_VAR",
    "TELEMETRY_SCHEMA_VERSION",
    "Entrypoint",
    "Event",
    "EventProperties",
    "Feature",
    "FlushReason",
    "Identity",
    "LoginMethod",
    "LoginOutcome",
    "LoginPromptSurface",
    "LoginSpan",
    "NoopBackend",
    "Outcome",
    "PostHogBackend",
    "Prop",
    "Runtime",
    "RunAccumulator",
    "TelemetryBackend",
    "TelemetryKey",
    "UserStatus",
    "base_properties",
    "blocked_by_firewall",
    "capture",
    "capture_benchmark_run",
    "capture_cli_command",
    "capture_conversation_simulator_run",
    "capture_evaluation_run",
    "capture_login_event",
    "capture_login_prompt_shown",
    "capture_login_prompt_shown_once",
    "capture_synthesizer_run",
    "capture_tracing_integration",
    "current_run",
    "describe_judge",
    "detect_runtime",
    "flush",
    "flush_standalone_metrics",
    "get_backend",
    "get_feature_status",
    "get_identity",
    "get_last_feature",
    "get_logged_in_with",
    "get_status",
    "get_unique_id",
    "installed_integrations",
    "read_telemetry_file",
    "record_golden",
    "record_login_completed",
    "record_metric",
    "record_test_case",
    "turn_kind_of",
    "TurnKind",
    "record_tracing_integration",
    "register_integration",
    "set_backend",
    "set_last_feature",
    "set_logged_in_with",
    "telemetry_opt_out",
    "telemetry_path",
    "write_telemetry_file",
]
