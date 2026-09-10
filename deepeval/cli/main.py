"""
DeepEval CLI.

The `set-*` / `unset-*` provider commands live in `deepeval.cli.providers`,
split into `llms`, `tts` and `stt`. General behavior for all of them:

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

import typer
import importlib.metadata
from typing import List, Optional
from rich import print
from rich.markup import escape
from rich.console import Console
from rich.table import Table
from enum import Enum
from pydantic import SecretStr
from pydantic_core import PydanticUndefined
from deepeval.telemetry import capture_cli_command
from deepeval.config.settings import get_settings
from deepeval.utils import open_browser
from deepeval.test_run.test_run import (
    global_test_run_manager,
)
from deepeval.cli.app import app
from deepeval.cli.auth import LOGIN_HELP, login_command, logout_command
from deepeval.cli.diagnose import diagnose_command
from deepeval.cli.generate.command import generate_command
from deepeval.cli.inspect import inspect_command
from deepeval.cli.test.command import app as test_app
from deepeval.cli.utils import (
    handle_save_result as _handle_save_result,
    is_optional,
    parse_and_validate,
    resolve_field_names,
    upload_and_open_link,
)
from deepeval.confident.api import (
    is_confident,
    Api,
    Endpoints,
    HttpMethods,
)

app.add_typer(test_app, name="test")
app.command(name="generate")(generate_command)
app.command(name="inspect")(inspect_command)
app.command(name="diagnose")(diagnose_command)
app.command(name="login", help=LOGIN_HELP)(login_command)
app.command(name="logout")(logout_command)


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


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the DeepEval version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    # The single hook that covers every registered command, including
    # `deepeval test run` via the mounted sub-app. `ctx.command` is the group
    # that just dispatched, so its own table is the list of valid names.
    capture_cli_command(
        ctx.invoked_subcommand, getattr(ctx.command, "commands", {})
    )


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
    if region == Regions.EU:
        flag = "🇪🇺"
    else:
        flag = "🇺🇸"

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


@app.command()
def view():
    if is_confident():
        last_test_run_link = global_test_run_manager.get_latest_test_run_link()
        if last_test_run_link:
            print(f"🔗 View test run: {last_test_run_link}")
            open_browser(last_test_run_link)
        else:
            upload_and_open_link()
    else:
        upload_and_open_link()


@app.command(
    name="gate",
    help=(
        "Check your project against its governance policy and fail (non-zero exit) "
        "if it doesn't pass. Your project must be associated with a governance policy "
        "on Confident AI; if it isn't, contact your organization administrator."
    ),
)
def gate(
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI). Exit code still reflects the verdict.",
    ),
):
    """Check your project against its governance policy."""
    try:
        api = Api()
    except ValueError as e:
        if not quiet:
            print(f"❌ {e}")
        raise typer.Exit(code=1)

    try:
        data, _ = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.GOVERNANCE_ASSESS_ENDPOINT,
        )
    except Exception as e:
        if not quiet:
            print(
                f"❌ Could not assess governance for your project: {e}\n"
                "Make sure your project is associated with a governance policy. "
                "If it isn't, please contact your organization administrator."
            )
        raise typer.Exit(code=1)

    data = data or {}
    passed = bool(data.get("passed"))
    policy = data.get("governancePolicy") or {}
    policy_name = policy.get("name") or "governance policy"

    if passed:
        if not quiet:
            print(
                f"✅ Governance gate passed against [bold]{escape(str(policy_name))}[/bold]."
            )
        raise typer.Exit(code=0)

    if not quiet:
        print(
            f"❌ Governance gate failed against [bold]{escape(str(policy_name))}[/bold]. "
            "One or more controls did not pass."
        )
    raise typer.Exit(code=1)


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


# Last, so provider commands list after the core ones in `deepeval --help`.
from deepeval.cli import providers  # noqa: E402,F401


if __name__ == "__main__":
    app()
