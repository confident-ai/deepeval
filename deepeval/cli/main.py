"""
DeepEval CLI: Model Provider Configuration Commands

General behavior for all `set-*` / `unset-*` commands:

- Non-secret settings (model name, endpoint, deployment, toggles) are always
  persisted in the hidden `.deepeval/.deepeval` JSON store.
- Secrets (API keys) are **never** written to the JSON store.
- If `--save=dotenv[:path]` is passed, both secrets and non-secrets are
  written to the specified dotenv file (default: `.env.local`).
  Dotenv files should be git-ignored.
- If `--save` is not passed, only the JSON store is updated.
- When unsetting a provider, only that provider’s keys are removed.
  If another provider’s credentials remain (e.g. `OPENAI_API_KEY`), it
  may still be selected as the default.
"""

import os
import webbrowser
import threading
import random
import string
import socket
import typer
import importlib.metadata
from typing import List, Optional
from rich import print
from rich.markup import escape
from rich.console import Console
from rich.table import Table
from enum import Enum
from pathlib import Path
from pydantic import SecretStr
from pydantic_core import PydanticUndefined
from deepeval.key_handler import (
    EmbeddingKeyValues,
    ModelKeyValues,
)
from deepeval.telemetry import capture_login_event, capture_view_event
from deepeval.config.settings import get_settings
from deepeval.utils import delete_file_if_exists, open_browser
from deepeval.test_run.test_run import (
    LATEST_TEST_RUN_FILE_PATH,
    global_test_run_manager,
)
from deepeval.cli.test import app as test_app
from deepeval.cli.server import start_server
from deepeval.cli.utils import (
    coerce_blank_to_none,
    is_optional,
    load_service_account_key_file,
    parse_and_validate,
    render_login_message,
    resolve_field_names,
    upload_and_open_link,
    PROD,
)
from deepeval.confident.api import (
    is_confident,
)

app = typer.Typer(name="deepeval", no_args_is_help=True)
app.add_typer(test_app, name="test")


class Regions(Enum):
    US = "US"
    EU = "EU"


def version_callback(value: Optional[bool] = None) -> None:
    if not value:
        return
    try:
        version = importlib.metadata.version("deepeval")
    except importlib.metadata.PackageNotFoundError:
        from deepeval import __version__ as version  # type: ignore
    typer.echo(version)  # or: typer.echo(f"deepeval {v}")
    raise typer.Exit()


def generate_pairing_code():
    """Generate a random pairing code."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))  # Bind to port 0 to get an available port
        return s.getsockname()[1]


def is_openai_configured() -> bool:
    s = get_settings()
    v = s.OPENAI_API_KEY
    if isinstance(v, SecretStr):
        try:
            if v.get_secret_value().strip():
                return True
        except Exception:
            pass
    elif v and str(v).strip():
        return True
    env = os.getenv("OPENAI_API_KEY")
    return bool(env and env.strip())


def _handle_save_result(
    *,
    handled: bool,
    path: Optional[str],
    updates: dict,
    save: Optional[str],
    quiet: bool,
    success_msg: Optional[str] = None,
    updated_msg: str = "Saved environment variables to {path} (ensure it's git-ignored).",
    no_changes_msg: str = "No changes to save in {path}.",
    tip_msg: Optional[str] = None,
) -> bool:
    if not handled and save is not None:
        raise typer.BadParameter(
            "Unsupported --save option. Use --save=dotenv[:path].",
            param_hint="--save",
        )

    if quiet:
        return False

    if path and updates:
        print(updated_msg.format(path=path))
    elif path:
        print(no_changes_msg.format(path=path))
    elif tip_msg:
        print(tip_msg)

    if success_msg:
        print(success_msg)

    return True


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the DeepEval version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    pass


@app.command(name="set-confident-region")
def set_confident_region_command(
    region: Regions = typer.Argument(
        ..., help="The data region to use (US or EU)"
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """Set the Confident AI data region."""
    # Add flag emojis based on region
    flag = "🇺🇸" if region == Regions.US else "🇪🇺"

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.CONFIDENT_REGION = region.value

    handled, path, updates = edit_ctx.result

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using the {flag}  {region.value} data region for Confident AI."
        ),
    )


@app.command(
    help=(
        "Login will prompt you for your Confident AI API key (input hidden). "
        "Get it from https://app.confident-ai.com. "
        "Required to log events to the server. "
        "The API key will be saved in your environment variables, typically in .env.local, unless a different path is provided with --save."
    )
)
def login(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Where to persist settings. Format: dotenv[:path]. Defaults to .env.local. If omitted, login still writes to .env.local.",
    ),
):
    api_key = coerce_blank_to_none(
        typer.prompt("🔐 Enter your API Key", hide_input=True)
    )

    with capture_login_event() as span:
        completed = False
        try:
            # Resolve the key from CLI flag or interactive flow
            if api_key is not None:
                key = api_key
            else:
                render_login_message()

                # Start the pairing server
                port = find_available_port()
                pairing_code = generate_pairing_code()
                pairing_thread = threading.Thread(
                    target=start_server,
                    args=(pairing_code, port, PROD),
                    daemon=True,
                )
                pairing_thread.start()

                # Open web url
                login_url = f"{PROD}/pair?code={pairing_code}&port={port}"
                webbrowser.open(login_url)
                print(
                    f"(open this link if your browser did not open: [link={PROD}]{PROD}[/link])"
                )

                # Manual fallback if still empty
                while True:
                    api_key = coerce_blank_to_none(
                        typer.prompt("🔐 Enter your API Key", hide_input=True)
                    )
                    if api_key:
                        break
                    else:
                        print("❌ API Key cannot be empty. Please try again.\n")
                key = api_key

            settings = get_settings()
            save = save or settings.DEEPEVAL_DEFAULT_SAVE or "dotenv:.env.local"
            with settings.edit(save=save) as edit_ctx:
                settings.API_KEY = key
                settings.CONFIDENT_API_KEY = key

            handled, path, updated = edit_ctx.result

            if updated:
                if not handled and save is not None:
                    # invalid --save format (unsupported)
                    print(
                        "Unsupported --save option. Use --save=dotenv[:path]."
                    )
                elif path:
                    # persisted to a file
                    print(
                        f"Saved environment variables to {path} (ensure it's git-ignored)."
                    )

            completed = True
            print(
                "\n🎉🥳 Congratulations! You've successfully logged in! :raising_hands:"
            )
            print(
                "You're now using DeepEval with [rgb(106,0,255)]Confident AI[/rgb(106,0,255)]. "
                "Follow our quickstart tutorial here: "
                "[bold][link=https://www.confident-ai.com/docs/llm-evaluation/quickstart]"
                "https://www.confident-ai.com/docs/llm-evaluation/quickstart[/link][/bold]"
            )
        except Exception as e:
            completed = False
            print(f"Login failed: {e}")
        finally:
            if getattr(span, "set_attribute", None):
                span.set_attribute("completed", completed)


@app.command()
def logout(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Where to remove the saved key from. Use format dotenv[:path]. If omitted, uses DEEPEVAL_DEFAULT_SAVE or .env.local. The JSON keystore is always cleared.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Log out of Confident AI.

    Behavior:
    - Always clears the Confident API key from the JSON keystore and process env.
    - Also removes credentials from a dotenv file; defaults to DEEPEVAL_DEFAULT_SAVE if set, otherwise.env.local.
      Override the target with --save=dotenv[:path].
    """
    settings = get_settings()
    save = save or settings.DEEPEVAL_DEFAULT_SAVE or "dotenv:.env.local"
    with settings.edit(save=save) as edit_ctx:
        settings.API_KEY = None
        settings.CONFIDENT_API_KEY = None

    handled, path, updated = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updated,
        save=save,
        quiet=quiet,
        updated_msg="Removed Confident AI key(s) from {path}.",
        tip_msg=None,
    ):
        print("\n🎉🥳 You've successfully logged out! :raising_hands: ")

    delete_file_if_exists(LATEST_TEST_RUN_FILE_PATH)


@app.command()
def view():
    with capture_view_event() as span:
        if is_confident():
            last_test_run_link = (
                global_test_run_manager.get_latest_test_run_link()
            )
            if last_test_run_link:
                print(f"🔗 View test run: {last_test_run_link}")
                open_browser(last_test_run_link)
            else:
                upload_and_open_link(_span=span)
        else:
            upload_and_open_link(_span=span)


@app.command(
    name="settings",
    help=(
        "Power-user command to set/unset any DeepEval Settings field. "
        "Uses Pydantic type validation. Supports partial, case-insensitive matching for --unset and --list."
    ),
)
def update_settings(
    set_: Optional[List[str]] = typer.Option(
        None,
        "-u",
        "--set",
        help="Set a setting (repeatable). Format: KEY=VALUE",
    ),
    unset: Optional[List[str]] = typer.Option(
        None,
        "-U",
        "--unset",
        help=(
            "Unset setting(s) by name or partial match (repeatable, case-insensitive). "
            "If a filter matches multiple keys, all are unset."
        ),
    ),
    list_: bool = typer.Option(
        False,
        "-l",
        "--list",
        help="List available settings. You can optionally pass a FILTER argument, such as `-l verbose`.",
    ),
    filters: Optional[List[str]] = typer.Argument(
        None,
        help="Optional filter(s) for --list (case-insensitive substring match). You can pass multiple terms.",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist settings to dotenv. Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    def _format_setting_value(val: object) -> str:
        if isinstance(val, SecretStr):
            secret = val.get_secret_value()
            return "********" if secret and secret.strip() else ""
        if val is None:
            return ""
        s = str(val)
        return s if len(s) <= 120 else (s[:117] + "…")

    def _print_settings_list(filter_terms: Optional[List[str]]) -> None:
        needles = []
        for term in filter_terms or []:
            t = term.strip().lower().replace("-", "_")
            if t:
                needles.append(t)

        table = Table(title="Settings")
        table.add_column("Name", style="bold")
        table.add_column("Value", overflow="fold")
        table.add_column("Description", overflow="fold")

        shown = 0
        for name in sorted(fields.keys()):
            hay = name.lower().replace("-", "_")
            if needles and not any(n in hay for n in needles):
                continue

            field_info = fields[name]
            desc = field_info.description or ""
            current_val = getattr(settings, name, None)
            table.add_row(name, _format_setting_value(current_val), desc)
            shown += 1

        if shown == 0:
            raise typer.BadParameter(f"No settings matched: {filter_terms!r}")

        Console().print(table)

    settings = get_settings()
    fields = type(settings).model_fields

    if filters is not None and not list_:
        raise typer.BadParameter("FILTER can only be used with --list / -l.")

    if list_:
        if set_ or unset:
            raise typer.BadParameter(
                "--list cannot be combined with --set/--unset."
            )
        _print_settings_list(filters)
        return

    # Build an assignment plan: name -> value (None means "unset")
    plan: dict[str, object] = {}

    # --unset (filters)
    if unset:
        matched_any = False
        for f in unset:
            matches = resolve_field_names(settings, f)
            if not matches:
                continue
            matched_any = True
            for name in matches:
                field_info = fields[name]
                ann = field_info.annotation

                # "unset" semantics:
                # - Optional -> None
                # - else -> reset to default if it exists
                if is_optional(ann):
                    plan[name] = None
                elif field_info.default is not PydanticUndefined:
                    plan[name] = field_info.default
                else:
                    raise typer.BadParameter(
                        f"Cannot unset required setting {name} (no default, not Optional)."
                    )

        if unset and not matched_any:
            raise typer.BadParameter(f"No settings matched: {unset!r}")

    # --set KEY=VALUE
    if set_:
        for item in set_:
            key, sep, raw = item.partition("=")
            if not sep:
                raise typer.BadParameter(
                    f"--set must be KEY=VALUE (got {item!r})"
                )

            matches = resolve_field_names(settings, key)
            if not matches:
                raise typer.BadParameter(f"Unknown setting: {key!r}")
            if len(matches) > 1:
                raise typer.BadParameter(
                    f"Ambiguous setting {key!r}; matches: {', '.join(matches)}"
                )

            name = matches[0]
            field_info = fields[name]
            plan[name] = parse_and_validate(name, field_info, raw)

    if not plan:
        # nothing requested
        return

    with settings.edit(save=save) as edit_ctx:
        for name, val in plan.items():
            setattr(settings, name, val)

    handled, path, updates = edit_ctx.result

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=":wrench: Settings updated." if updates else None,
    )


@app.command(
    name="set-debug",
    help=(
        "Configure verbosity flags (global LOG_LEVEL, verbose mode), retry logger levels, "
        "gRPC logging, and Confident trace toggles. Use the --save option to persist settings "
        "to a dotenv file (default: .env.local)."
    ),
)
def set_debug(
    # Core verbosity
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Global LOG_LEVEL (DEBUG|INFO|WARNING|ERROR|CRITICAL|NOTSET).",
    ),
    verbose: Optional[bool] = typer.Option(
        None, "--verbose/--no-verbose", help="Toggle DEEPEVAL_VERBOSE_MODE."
    ),
    debug_async: Optional[bool] = typer.Option(
        None,
        "--debug-async/--no-debug-async",
        help="Toggle DEEPEVAL_DEBUG_ASYNC.",
    ),
    log_stack_traces: Optional[bool] = typer.Option(
        None,
        "--log-stack-traces/--no-log-stack-traces",
        help="Toggle DEEPEVAL_LOG_STACK_TRACES.",
    ),
    # Retry logging dials
    retry_before_level: Optional[str] = typer.Option(
        None,
        "--retry-before-level",
        help="Log level before a retry attempt (DEBUG|INFO|WARNING|ERROR|CRITICAL|NOTSET or numeric).",
    ),
    retry_after_level: Optional[str] = typer.Option(
        None,
        "--retry-after-level",
        help="Log level after a retry attempt (DEBUG|INFO|WARNING|ERROR|CRITICAL|NOTSET or numeric).",
    ),
    # gRPC visibility
    grpc: Optional[bool] = typer.Option(
        None, "--grpc/--no-grpc", help="Toggle DEEPEVAL_GRPC_LOGGING."
    ),
    grpc_verbosity: Optional[str] = typer.Option(
        None,
        "--grpc-verbosity",
        help="Set GRPC_VERBOSITY (DEBUG|INFO|ERROR|NONE).",
    ),
    grpc_trace: Optional[str] = typer.Option(
        None,
        "--grpc-trace",
        help=(
            "Set GRPC_TRACE to comma-separated tracer names or glob patterns "
            "(e.g. 'tcp,http,secure_endpoint', '*' for all, 'list_tracers' to print available)."
        ),
    ),
    # Confident tracing
    trace_verbose: Optional[bool] = typer.Option(
        None,
        "--trace-verbose/--no-trace-verbose",
        help="Enable / disable CONFIDENT_TRACE_VERBOSE.",
    ),
    trace_env: Optional[str] = typer.Option(
        None,
        "--trace-env",
        help='Set CONFIDENT_TRACE_ENVIRONMENT ("development", "staging", "production", etc).',
    ),
    trace_flush: Optional[bool] = typer.Option(
        None,
        "--trace-flush/--no-trace-flush",
        help="Enable / disable  CONFIDENT_TRACE_FLUSH.",
    ),
    trace_sample_rate: Optional[float] = typer.Option(
        None,
        "--trace-sample-rate",
        help="Set CONFIDENT_TRACE_SAMPLE_RATE.",
    ),
    metric_logging_verbose: Optional[bool] = typer.Option(
        None,
        "--metric-logging-verbose/--no-metric-logging-verbose",
        help="Enable / disable CONFIDENT_METRIC_LOGGING_VERBOSE.",
    ),
    metric_logging_flush: Optional[bool] = typer.Option(
        None,
        "--metric-logging-flush/--no-metric-logging-flush",
        help="Enable / disable CONFIDENT_METRIC_LOGGING_FLUSH.",
    ),
    metric_logging_sample_rate: Optional[float] = typer.Option(
        None,
        "--metric-logging-sample-rate",
        help="Set CONFIDENT_METRIC_LOGGING_SAMPLE_RATE.",
    ),
    metric_logging_enabled: Optional[bool] = typer.Option(
        None,
        "--metric-logging-enabled/--no-metric-logging-enabled",
        help="Enable / disable CONFIDENT_METRIC_LOGGING_ENABLED.",
    ),
    # Persistence
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Configure debug and logging behaviors for DeepEval.

    Use verbosity flags to set the global log level, retry logging behavior, gRPC logging,
    Confident AI tracing, and more. This command applies changes immediately but can also
    persist settings to a dotenv file with --save.
    """
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        # Core verbosity
        if log_level is not None:
            settings.LOG_LEVEL = log_level
        if verbose is not None:
            settings.DEEPEVAL_VERBOSE_MODE = verbose
        if debug_async is not None:
            settings.DEEPEVAL_DEBUG_ASYNC = debug_async
        if log_stack_traces is not None:
            settings.DEEPEVAL_LOG_STACK_TRACES = log_stack_traces

        # Retry logging
        if retry_before_level is not None:
            settings.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL = retry_before_level
        if retry_after_level is not None:
            settings.DEEPEVAL_RETRY_AFTER_LOG_LEVEL = retry_after_level

        # gRPC
        if grpc is not None:
            settings.DEEPEVAL_GRPC_LOGGING = grpc
        if grpc_verbosity is not None:
            settings.GRPC_VERBOSITY = grpc_verbosity
        if grpc_trace is not None:
            settings.GRPC_TRACE = grpc_trace

        # Confident tracing
        if trace_verbose is not None:
            settings.CONFIDENT_TRACE_VERBOSE = trace_verbose
        if trace_env is not None:
            settings.CONFIDENT_TRACE_ENVIRONMENT = trace_env
        if trace_flush is not None:
            settings.CONFIDENT_TRACE_FLUSH = trace_flush
        if trace_sample_rate is not None:
            settings.CONFIDENT_TRACE_SAMPLE_RATE = trace_sample_rate

        # Confident metrics
        if metric_logging_verbose is not None:
            settings.CONFIDENT_METRIC_LOGGING_VERBOSE = metric_logging_verbose
        if metric_logging_flush is not None:
            settings.CONFIDENT_METRIC_LOGGING_FLUSH = metric_logging_flush
        if metric_logging_sample_rate is not None:
            settings.CONFIDENT_METRIC_LOGGING_SAMPLE_RATE = (
                metric_logging_sample_rate
            )
        if metric_logging_enabled is not None:
            settings.CONFIDENT_METRIC_LOGGING_ENABLED = metric_logging_enabled

    handled, path, updates = edit_ctx.result

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=":loud_sound: Debug options updated." if updates else None,
    )


@app.command(
    name="unset-debug",
    help=(
        "Restore default behavior by removing debug-related overrides. "
        "Use --save to also remove these keys from a dotenv file (default: .env.local)."
    ),
)
def unset_debug(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the debug-related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        # Core verbosity
        settings.LOG_LEVEL = None
        settings.DEEPEVAL_VERBOSE_MODE = None
        settings.DEEPEVAL_DEBUG_ASYNC = None
        settings.DEEPEVAL_LOG_STACK_TRACES = None

        # Retry logging dials
        settings.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL = None
        settings.DEEPEVAL_RETRY_AFTER_LOG_LEVEL = None

        # gRPC visibility
        settings.DEEPEVAL_GRPC_LOGGING = None
        settings.GRPC_VERBOSITY = None
        settings.GRPC_TRACE = None

        # Confident tracing
        settings.CONFIDENT_TRACE_VERBOSE = None
        settings.CONFIDENT_TRACE_ENVIRONMENT = None
        settings.CONFIDENT_TRACE_FLUSH = None
        settings.CONFIDENT_TRACE_SAMPLE_RATE = None

        # Confident metrics
        settings.CONFIDENT_METRIC_LOGGING_VERBOSE = None
        settings.CONFIDENT_METRIC_LOGGING_FLUSH = None
        settings.CONFIDENT_METRIC_LOGGING_SAMPLE_RATE = None
        settings.CONFIDENT_METRIC_LOGGING_ENABLED = None

    handled, path, updates = edit_ctx.result

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=":mute: Debug options unset." if updates else None,
        tip_msg=None,
    )


#############################################
# OpenAI Integration ########################
#############################################


@app.command(name="set-openai")
def set_openai_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider (e.g., `gpt-4.1`).",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for OPENAI_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Configure OpenAI as the active LLM provider.

    What this does:
    - Sets the active provider flag to `USE_OPENAI_MODEL`.
    - Persists the selected model name and any cost overrides in the JSON store.
    - secrets are never written to `.deepeval/.deepeval` (JSON).

    Pricing rules:
    - If `model` is a known OpenAI model, you may omit costs (built‑in pricing is used).
    - If `model` is custom/unsupported, you must provide both
      `--cost-per-input-token` and `--cost-per-output-token`.

    Secrets & saving:

    - If you run with --prompt-api-key, DeepEval will set OPENAI_API_KEY for this session.
    - If --save=dotenv[:path] is used (or DEEPEVAL_DEFAULT_SAVE is set), the key will be written to that dotenv file (plaintext).

    Secrets are never written to .deepeval/.deepeval (legacy JSON store).

    Args:
        --model: OpenAI model name, such as `gpt-4o-mini`.
        --prompt-api-key: Prompt interactively for OPENAI_API_KEY (input hidden). Avoids putting secrets on the command line (shell history/process args). Not suitable for CI.
        --cost-per-input-token: USD per input token (optional for known models).
        --cost-per-output-token: USD per output token (optional for known models).
        --save: Persist config (and supported secrets) to a dotenv file; format `dotenv[:path]`.
        --quiet: Suppress printing to the terminal.

    Example:
        deepeval set-openai \\
          --model gpt-4o-mini \\
          --cost-per-input-token 0.0005 \\
          --cost-per-output-token 0.0015 \\
          --save dotenv:.env.local
    """
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("OpenAI API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_OPENAI_MODEL)
        if model is not None:
            settings.OPENAI_MODEL_NAME = model
        if api_key is not None:
            settings.OPENAI_API_KEY = api_key
        if cost_per_input_token is not None:
            settings.OPENAI_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.OPENAI_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.OPENAI_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "OpenAI model name is not set. Pass --model (or set OPENAI_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using OpenAI's `{escape(effective_model)}` "
            "for all evals that require an LLM."
        ),
    )


@app.command(name="unset-openai")
def unset_openai_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the OpenAI related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove OPENAI_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Unset OpenAI as the active provider.

    Behavior:
    - Removes OpenAI keys (model, costs, toggle) from the JSON store.
    - If `--save` is provided, removes those keys from the specified dotenv file.
    - After unsetting, if `OPENAI_API_KEY` is still set in the environment,
      OpenAI may still be usable by default. Otherwise, no active provider is configured.

    Args:
        --save: Remove OpenAI keys from the given dotenv file as well.
        --clear-secrets: Removes OPENAI_API_KEY from the dotenv store
        --quiet: Suppress printing to the terminal

    Example:
        deepeval unset-openai --save dotenv:.env.local
    """

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.OPENAI_MODEL_NAME = None
        settings.OPENAI_COST_PER_INPUT_TOKEN = None
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = None
        settings.USE_OPENAI_MODEL = None
        if clear_secrets:
            settings.OPENAI_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed OpenAI environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "OpenAI has been unset. No active provider is configured. "
                "Set one with the CLI, or add credentials to .env[.local]."
            )


#############################################
# Azure Integration ########################
#############################################


@app.command(name="set-azure-openai")
def set_azure_openai_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider (e.g., `gpt-4.1`).",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for AZURE_OPENAI_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    api_version: Optional[str] = typer.Option(
        None,
        "-v",
        "--api-version",
        help="Azure OpenAI API version (passed to the Azure OpenAI client).",
    ),
    model_version: Optional[str] = typer.Option(
        None, "-V", "--model-version", help="Azure model version"
    ),
    deployment_name: Optional[str] = typer.Option(
        None, "-d", "--deployment-name", help="Azure OpenAI deployment name"
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Azure OpenAI API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)
    api_version = coerce_blank_to_none(api_version)
    deployment_name = coerce_blank_to_none(deployment_name)
    model_version = coerce_blank_to_none(model_version)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_AZURE_OPENAI)
        if model is not None:
            settings.AZURE_MODEL_NAME = model
        if api_key is not None:
            settings.AZURE_OPENAI_API_KEY = api_key
        if base_url is not None:
            settings.AZURE_OPENAI_ENDPOINT = base_url
        if api_version is not None:
            settings.OPENAI_API_VERSION = api_version
        if deployment_name is not None:
            settings.AZURE_DEPLOYMENT_NAME = deployment_name
        if model_version is not None:
            settings.AZURE_MODEL_VERSION = model_version

    handled, path, updates = edit_ctx.result

    effective_model = settings.AZURE_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Azure OpenAI model name is not set. Pass --model (or set AZURE_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Azure OpenAI's `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-azure-openai")
def unset_azure_openai_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Azure OpenAI–related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove AZURE_OPENAI_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.AZURE_OPENAI_ENDPOINT = None
        settings.OPENAI_API_VERSION = None
        settings.AZURE_DEPLOYMENT_NAME = None
        settings.AZURE_MODEL_NAME = None
        settings.AZURE_MODEL_VERSION = None
        settings.USE_AZURE_OPENAI = None
        if clear_secrets:
            settings.AZURE_OPENAI_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Azure OpenAI environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "Azure OpenAI has been unset. No active provider is configured. Set one with the CLI, or add credentials to .env[.local]."
            )


@app.command(name="set-azure-openai-embedding")
def set_azure_openai_embedding_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider (e.g., `gpt-4.1`).",
    ),
    deployment_name: Optional[str] = typer.Option(
        None,
        "-d",
        "--deployment-name",
        help="Azure embedding deployment name",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    model = coerce_blank_to_none(model)
    deployment_name = coerce_blank_to_none(deployment_name)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(
            EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING
        )
        if model is not None:
            settings.AZURE_EMBEDDING_MODEL_NAME = model
        if deployment_name is not None:
            settings.AZURE_EMBEDDING_DEPLOYMENT_NAME = deployment_name

    handled, path, updates = edit_ctx.result

    effective_model = settings.AZURE_EMBEDDING_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Azure OpenAI embedding model name is not set. Pass --model (or set AZURE_EMBEDDING_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Azure OpenAI embedding model `{escape(effective_model)}` for all evals that require text embeddings."
        ),
    )


@app.command(name="unset-azure-openai-embedding")
def unset_azure_openai_embedding_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Azure OpenAI embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.AZURE_EMBEDDING_MODEL_NAME = None
        settings.AZURE_EMBEDDING_DEPLOYMENT_NAME = None
        settings.USE_AZURE_OPENAI_EMBEDDING = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Azure OpenAI embedding environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "Azure OpenAI embedding has been unset. No active provider is configured. Set one with the CLI, or add credentials to .env[.local]."
            )


#############################################
# Anthropic Model Integration ###############
#############################################


@app.command(name="set-anthropic")
def set_anthropic_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for ANTHROPIC_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Anthropic API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_ANTHROPIC_MODEL)
        if api_key is not None:
            settings.ANTHROPIC_API_KEY = api_key
        if model is not None:
            settings.ANTHROPIC_MODEL_NAME = model
        if cost_per_input_token is not None:
            settings.ANTHROPIC_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.ANTHROPIC_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Anthropic model name is not set. Pass --model (or set ANTHROPIC_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Anthropic `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-anthropic")
def unset_anthropic_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Anthropic model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove ANTHROPIC_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_ANTHROPIC_MODEL = None
        settings.ANTHROPIC_MODEL_NAME = None
        settings.ANTHROPIC_COST_PER_INPUT_TOKEN = None
        settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN = None
        if clear_secrets:
            settings.ANTHROPIC_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Anthropic model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The Anthropic model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# AWS Bedrock Model Integration #############
#############################################


@app.command(name="set-bedrock")
def set_bedrock_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_credentials: bool = typer.Option(
        False,
        "-a",
        "--prompt-credentials",
        help=(
            "Prompt for AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (secret access key input is hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, credentials are written to dotenv in plaintext."
        ),
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r",
        "--region",
        help="AWS region for bedrock (e.g., `us-east-1`).",
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    access_key_id = None
    secret_access_key = None
    if prompt_credentials:
        access_key_id = coerce_blank_to_none(typer.prompt("AWS Access key Id"))
        secret_access_key = coerce_blank_to_none(
            typer.prompt("AWS Secret Access key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    region = coerce_blank_to_none(region)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_AWS_BEDROCK_MODEL)
        if access_key_id is not None:
            settings.AWS_ACCESS_KEY_ID = access_key_id
        if secret_access_key is not None:
            settings.AWS_SECRET_ACCESS_KEY = secret_access_key
        if model is not None:
            settings.AWS_BEDROCK_MODEL_NAME = model
        if region is not None:
            settings.AWS_BEDROCK_REGION = region
        if cost_per_input_token is not None:
            settings.AWS_BEDROCK_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.AWS_BEDROCK_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.AWS_BEDROCK_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "AWS Bedrock model name is not set. Pass --model (or set AWS_BEDROCK_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using AWS Bedrock `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-bedrock")
def unset_bedrock_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the AWS Bedrock model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY  from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_AWS_BEDROCK_MODEL = None
        settings.AWS_BEDROCK_MODEL_NAME = None
        settings.AWS_BEDROCK_REGION = None
        settings.AWS_BEDROCK_COST_PER_INPUT_TOKEN = None
        settings.AWS_BEDROCK_COST_PER_OUTPUT_TOKEN = None
        if clear_secrets:
            settings.AWS_ACCESS_KEY_ID = None
            settings.AWS_SECRET_ACCESS_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed AWS Bedrock model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The AWS Bedrock model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# Ollama Integration ########################
#############################################


@app.command(name="set-ollama")
def set_ollama_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_LOCAL_MODEL)
        settings.LOCAL_MODEL_API_KEY = "ollama"
        if model is not None:
            settings.OLLAMA_MODEL_NAME = model
        if base_url is not None:
            settings.LOCAL_MODEL_BASE_URL = base_url

    handled, path, updates = edit_ctx.result

    effective_model = settings.OLLAMA_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Ollama model name is not set. Pass --model (or set OLLAMA_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using a local Ollama model `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-ollama")
def unset_ollama_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Ollama related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove LOCAL_MODEL_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        if clear_secrets:
            settings.LOCAL_MODEL_API_KEY = None
        settings.OLLAMA_MODEL_NAME = None
        settings.LOCAL_MODEL_BASE_URL = None
        settings.USE_LOCAL_MODEL = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed local Ollama environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The local Ollama model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


@app.command(name="set-ollama-embeddings")
def set_ollama_embeddings_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider.",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
        settings.LOCAL_EMBEDDING_API_KEY = "ollama"
        if model is not None:
            settings.LOCAL_EMBEDDING_MODEL_NAME = model
        if base_url is not None:
            settings.LOCAL_EMBEDDING_BASE_URL = base_url

    handled, path, updates = edit_ctx.result

    effective_model = settings.LOCAL_EMBEDDING_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Ollama embedding model name is not set. Pass --model (or set LOCAL_EMBEDDING_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using the Ollama embedding model `{escape(effective_model)}` for all evals that require text embeddings."
        ),
    )


@app.command(name="unset-ollama-embeddings")
def unset_ollama_embeddings_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Ollama embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove LOCAL_EMBEDDING_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        if clear_secrets:
            settings.LOCAL_EMBEDDING_API_KEY = None
        settings.LOCAL_EMBEDDING_MODEL_NAME = None
        settings.LOCAL_EMBEDDING_BASE_URL = None
        settings.USE_LOCAL_EMBEDDINGS = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed local Ollama embedding environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: Regular OpenAI embeddings will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The local Ollama embedding model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# Local Model Integration ###################
#############################################


@app.command(name="set-local-model")
def set_local_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for LOCAL_MODEL_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    model_format: Optional[str] = typer.Option(
        None,
        "-f",
        "--format",
        help="Format of the response from the local model (default: json)",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Local Model API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)
    model_format = coerce_blank_to_none(model_format)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_LOCAL_MODEL)
        if model is not None:
            settings.LOCAL_MODEL_NAME = model
        if base_url is not None:
            settings.LOCAL_MODEL_BASE_URL = base_url
        if api_key is not None:
            settings.LOCAL_MODEL_API_KEY = api_key
        if model_format is not None:
            settings.LOCAL_MODEL_FORMAT = model_format

    handled, path, updates = edit_ctx.result

    effective_model = settings.LOCAL_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Local model name is not set. Pass --model (or set LOCAL_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using a local model `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-local-model")
def unset_local_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the local model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove LOCAL_MODEL_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        if clear_secrets:
            settings.LOCAL_MODEL_API_KEY = None
        settings.LOCAL_MODEL_NAME = None
        settings.LOCAL_MODEL_BASE_URL = None
        settings.LOCAL_MODEL_FORMAT = None
        settings.USE_LOCAL_MODEL = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed local model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The local model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# Grok Model Integration ####################
#############################################


@app.command(name="set-grok")
def set_grok_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for GROK_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Grok API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_GROK_MODEL)
        if api_key is not None:
            settings.GROK_API_KEY = api_key
        if model is not None:
            settings.GROK_MODEL_NAME = model
        if cost_per_input_token is not None:
            settings.GROK_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.GROK_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.GROK_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Grok model name is not set. Pass --model (or set GROK_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Grok `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-grok")
def unset_grok_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Grok model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove GROK_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_GROK_MODEL = None
        settings.GROK_MODEL_NAME = None
        settings.GROK_COST_PER_INPUT_TOKEN = None
        settings.GROK_COST_PER_OUTPUT_TOKEN = None
        if clear_secrets:
            settings.GROK_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Grok model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The Grok model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# Moonshot Model Integration ################
#############################################


@app.command(name="set-moonshot")
def set_moonshot_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for MOONSHOT_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Moonshot API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_MOONSHOT_MODEL)
        if model is not None:
            settings.MOONSHOT_MODEL_NAME = model
        if api_key is not None:
            settings.MOONSHOT_API_KEY = api_key
        if cost_per_input_token is not None:
            settings.MOONSHOT_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.MOONSHOT_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.MOONSHOT_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Moonshot model name is not set. Pass --model (or set MOONSHOT_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Moonshot `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-moonshot")
def unset_moonshot_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Moonshot model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove MOONSHOT_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_MOONSHOT_MODEL = None
        settings.MOONSHOT_MODEL_NAME = None
        settings.MOONSHOT_COST_PER_INPUT_TOKEN = None
        settings.MOONSHOT_COST_PER_OUTPUT_TOKEN = None
        if clear_secrets:
            settings.MOONSHOT_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Moonshot model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The Moonshot model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# DeepSeek Model Integration ################
#############################################


@app.command(name="set-deepseek")
def set_deepseek_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for DEEPSEEK_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("DeepSeek API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_DEEPSEEK_MODEL)
        if model is not None:
            settings.DEEPSEEK_MODEL_NAME = model
        if api_key is not None:
            settings.DEEPSEEK_API_KEY = api_key
        if cost_per_input_token is not None:
            settings.DEEPSEEK_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.DEEPSEEK_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.DEEPSEEK_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "DeepSeek model name is not set. Pass --model (or set DEEPSEEK_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using DeepSeek `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-deepseek")
def unset_deepseek_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the DeepSeek model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove DEEPSEEK_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_DEEPSEEK_MODEL = None
        settings.DEEPSEEK_MODEL_NAME = None
        settings.DEEPSEEK_COST_PER_INPUT_TOKEN = None
        settings.DEEPSEEK_COST_PER_OUTPUT_TOKEN = None
        if clear_secrets:
            settings.DEEPSEEK_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed DeepSeek model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The DeepSeek model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# Local Embedding Model Integration #########
#############################################


@app.command(name="set-local-embeddings")
def set_local_embeddings_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for LOCAL_EMBEDDING_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Local Embedding Model API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
        if model is not None:
            settings.LOCAL_EMBEDDING_MODEL_NAME = model
        if base_url is not None:
            settings.LOCAL_EMBEDDING_BASE_URL = base_url
        if api_key is not None:
            settings.LOCAL_EMBEDDING_API_KEY = api_key

    handled, path, updates = edit_ctx.result

    effective_model = settings.LOCAL_EMBEDDING_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Local embedding model name is not set. Pass --model (or set LOCAL_EMBEDDING_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using the local embedding model `{escape(effective_model)}` for all evals that require text embeddings."
        ),
    )


@app.command(name="unset-local-embeddings")
def unset_local_embeddings_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the local embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove LOCAL_MODEL_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.LOCAL_EMBEDDING_MODEL_NAME = None
        settings.LOCAL_EMBEDDING_BASE_URL = None
        settings.USE_LOCAL_EMBEDDINGS = None
        if clear_secrets:
            settings.LOCAL_EMBEDDING_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed local embedding environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The local embeddings model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# Gemini Integration ########################
#############################################


@app.command(name="set-gemini")
def set_gemini_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for GOOGLE_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    project: Optional[str] = typer.Option(
        None,
        "-p",
        "--project",
        help="GCP project ID (used by Vertex AI / Gemini when applicable).",
    ),
    location: Optional[str] = typer.Option(
        None,
        "-l",
        "--location",
        help="GCP location/region for Vertex AI (e.g., `us-central1`).",
    ),
    service_account_file: Optional[Path] = typer.Option(
        None,
        "-S",
        "--service-account-file",
        help=("Path to a Google service account JSON key file."),
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Google API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    project = coerce_blank_to_none(project)
    location = coerce_blank_to_none(location)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_GEMINI_MODEL)

        if model is not None:
            settings.GEMINI_MODEL_NAME = model
        if project is not None:
            settings.GOOGLE_CLOUD_PROJECT = project
        if location is not None:
            settings.GOOGLE_CLOUD_LOCATION = location
        if service_account_file is not None:
            settings.GOOGLE_SERVICE_ACCOUNT_KEY = load_service_account_key_file(
                service_account_file
            )
        if api_key is not None:
            settings.GOOGLE_API_KEY = api_key
            settings.GOOGLE_GENAI_USE_VERTEXAI = False
        elif (
            project is not None
            or location is not None
            or service_account_file is not None
        ):
            settings.GOOGLE_GENAI_USE_VERTEXAI = True

    handled, path, updates = edit_ctx.result

    effective_model = settings.GEMINI_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Gemini model name is not set. Pass --model (or set GEMINI_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Gemini `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-gemini")
def unset_gemini_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Gemini related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove GOOGLE_API_KEY and GOOGLE_SERVICE_ACCOUNT_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_GEMINI_MODEL = None
        settings.GOOGLE_GENAI_USE_VERTEXAI = None
        settings.GOOGLE_CLOUD_PROJECT = None
        settings.GOOGLE_CLOUD_LOCATION = None
        settings.GEMINI_MODEL_NAME = None
        if clear_secrets:
            settings.GOOGLE_API_KEY = None
            settings.GOOGLE_SERVICE_ACCOUNT_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Gemini model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The Gemini model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


@app.command(name="set-litellm")
def set_litellm_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for LITELLM_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    proxy_prompt_api_key: bool = typer.Option(
        False,
        "-K",
        "--proxy-prompt-api-key",
        help=(
            "Prompt for LITELLM_PROXY_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    proxy_base_url: Optional[str] = typer.Option(
        None,
        "-U",
        "--proxy-base-url",
        help="Override the LITELLM_PROXY_API_BASE URL (useful for proxies, gateways, or self-hosted endpoints).",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("LiteLLM API key", hide_input=True)
        )

    proxy_api_key = None
    if proxy_prompt_api_key:
        proxy_api_key = coerce_blank_to_none(
            typer.prompt("LiteLLM Proxy API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)
    proxy_base_url = coerce_blank_to_none(proxy_base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_LITELLM)
        if model is not None:
            settings.LITELLM_MODEL_NAME = model
        if api_key is not None:
            settings.LITELLM_API_KEY = api_key
        if base_url is not None:
            settings.LITELLM_API_BASE = base_url
        if proxy_api_key is not None:
            settings.LITELLM_PROXY_API_KEY = proxy_api_key
        if proxy_base_url is not None:
            settings.LITELLM_PROXY_API_BASE = proxy_base_url

    handled, path, updates = edit_ctx.result

    effective_model = settings.LITELLM_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "LiteLLM model name is not set. Pass --model (or set LITELLM_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using LiteLLM `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-litellm")
def unset_litellm_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the LiteLLM related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove LITELLM_API_KEY and LITELLM_PROXY_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_LITELLM = None
        settings.LITELLM_MODEL_NAME = None
        settings.LITELLM_API_BASE = None
        settings.LITELLM_PROXY_API_BASE = None
        if clear_secrets:
            settings.LITELLM_API_KEY = None
            settings.LITELLM_PROXY_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed LiteLLM model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The LiteLLM model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# Portkey                       #############
#############################################


@app.command(name="set-portkey")
def set_portkey_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for PORTKEY_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "-P",
        "--provider",
        help="Override the PORTKEY_PROVIDER_NAME.",
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Portkey API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)
    provider = coerce_blank_to_none(provider)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_PORTKEY_MODEL)
        if model is not None:
            settings.PORTKEY_MODEL_NAME = model
        if api_key is not None:
            settings.PORTKEY_API_KEY = api_key
        if base_url is not None:
            settings.PORTKEY_BASE_URL = base_url
        if provider is not None:
            settings.PORTKEY_PROVIDER_NAME = provider

    handled, path, updates = edit_ctx.result

    effective_model = settings.PORTKEY_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Portkey model name is not set. Pass --model (or set PORTKEY_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Portkey `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-portkey")
def unset_portkey_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Portkey related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove PORTKEY_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_PORTKEY_MODEL = None
        settings.PORTKEY_MODEL_NAME = None
        settings.PORTKEY_BASE_URL = None
        settings.PORTKEY_PROVIDER_NAME = None
        if clear_secrets:
            settings.PORTKEY_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Portkey model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The Portkey model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


#############################################
# OpenRouter Integration ####################
#############################################


@app.command(name="set-openrouter")
def set_openrouter_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider (e.g., `openai/gpt-4.1`).",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for OPENROUTER_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider (default: https://openrouter.ai/api/v1).",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "-t",
        "--temperature",
        help="Override the global TEMPERATURE used by LLM providers (e.g., 0.0 for deterministic behavior).",
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token used for cost tracking. "
            "If unset and OpenRouter does not return pricing metadata, "
            "costs will not be calculated."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token used for cost tracking. "
            "If unset and OpenRouter does not return pricing metadata, "
            "costs will not be calculated."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("OpenRouter API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_OPENROUTER_MODEL)
        if model is not None:
            settings.OPENROUTER_MODEL_NAME = model
        if api_key is not None:
            settings.OPENROUTER_API_KEY = api_key
        if base_url is not None:
            settings.OPENROUTER_BASE_URL = base_url
        if temperature is not None:
            settings.TEMPERATURE = temperature
        if cost_per_input_token is not None:
            settings.OPENROUTER_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.OPENROUTER_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.OPENROUTER_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "OpenRouter model name is not set. Pass --model (or set OPENROUTER_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using OpenRouter `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-openrouter")
def unset_openrouter_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the OpenRouter model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove OPENROUTER_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_OPENROUTER_MODEL = None
        settings.OPENROUTER_MODEL_NAME = None
        settings.OPENROUTER_BASE_URL = None
        settings.OPENROUTER_COST_PER_INPUT_TOKEN = None
        settings.OPENROUTER_COST_PER_OUTPUT_TOKEN = None
        # Intentionally do NOT touch TEMPERATURE here; it's a global dial.
        if clear_secrets:
            settings.OPENROUTER_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed OpenRouter model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The OpenRouter model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


if __name__ == "__main__":
    app()
