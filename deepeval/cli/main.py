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
from typing import Optional
from rich import print
from rich.markup import escape
import webbrowser
import threading
import random
import string
import socket
import typer
from enum import Enum
from pydantic import SecretStr
from deepeval.key_handler import (
    EmbeddingKeyValues,
    ModelKeyValues,
)
from deepeval.telemetry import capture_login_event, capture_view_event
from deepeval.cli.test import app as test_app
from deepeval.cli.server import start_server
from deepeval.config.settings import get_settings
from deepeval.utils import delete_file_if_exists, open_browser
from deepeval.test_run.test_run import (
    LATEST_TEST_RUN_FILE_PATH,
    global_test_run_manager,
)
from deepeval.cli.utils import (
    render_login_message,
    upload_and_open_link,
    PROD,
)
from deepeval.confident.api import (
    is_confident,
)

app = typer.Typer(name="deepeval")
app.add_typer(test_app, name="test")


class Regions(Enum):
    US = "US"
    EU = "EU"


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


@app.command(name="set-confident-region")
def set_confident_region_command(
    region: Regions = typer.Argument(
        ..., help="The data region to use (US or EU)"
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    """Set the Confident AI data region."""
    # Add flag emojis based on region
    flag = "🇺🇸" if region == Regions.US else "🇪🇺"

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.CONFIDENT_REGION = region.value

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using the {flag}  {region.value} data region for Confident AI."
    )


@app.command()
def login(
    api_key: str = typer.Option(
        "",
        help="API key to get from https://app.confident-ai.com. Required if you want to log events to the server.",
    ),
    confident_api_key: Optional[str] = typer.Option(
        None,
        "--confident-api-key",
        "-c",
        help="Confident API key (non-interactive). If omitted, you'll be prompted to enter one. In all cases the key is saved to a dotenv file (default: .env.local) unless overridden with --save.",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Where to persist settings. Format: dotenv[:path]. Defaults to .env.local. If omitted, login still writes to .env.local.",
    ),
):
    with capture_login_event() as span:
        completed = False
        try:
            # Resolve the key from CLI flag or interactive flow
            if confident_api_key:
                key = confident_api_key.strip()
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
                if api_key == "":
                    while True:
                        api_key = input("🔐 Enter your API Key: ").strip()
                        if api_key:
                            break
                        else:
                            print(
                                "❌ API Key cannot be empty. Please try again.\n"
                            )
                key = api_key.strip()

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
        "--save",
        help="Where to remove the saved key from. Use format dotenv[:path]. If omitted, uses DEEPEVAL_DEFAULT_SAVE or .env.local. The JSON keystore is always cleared.",
    )
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

    if updated:
        if not handled and save is not None:
            # invalid --save format (unsupported)
            print("Unsupported --save option. Use --save=dotenv[:path].")
        elif path:
            # persisted to a file
            print(f"Removed Confident AI key(s) from {path}.")

    delete_file_if_exists(LATEST_TEST_RUN_FILE_PATH)

    print("\n🎉🥳 You've successfully logged out! :raising_hands: ")


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


@app.command(name="set-debug")
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
    # Advanced / potentially surprising
    error_reporting: Optional[bool] = typer.Option(
        None,
        "--error-reporting/--no-error-reporting",
        help="Enable / disable ERROR_REPORTING.",
    ),
    ignore_errors: Optional[bool] = typer.Option(
        None,
        "--ignore-errors/--no-ignore-errors",
        help="Enable / disable IGNORE_DEEPEVAL_ERRORS (not recommended in normal debugging).",
    ),
    # Persistence
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    """
    Configure verbose debug behavior for DeepEval.

    This command lets you mix-and-match verbosity flags (global LOG_LEVEL, verbose mode),
    retry logger levels, gRPC wire logging, and Confident trace toggles. Values apply
    immediately to the current process and can be persisted to a dotenv file with --save.

    Examples:
        deepeval set-debug --log-level DEBUG --verbose --grpc --retry-before-level DEBUG --retry-after-level INFO
        deepeval set-debug --trace-verbose --trace-env staging --save dotenv:.env.local
    """
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        # Core verbosity
        if log_level is not None:
            settings.LOG_LEVEL = log_level
        if verbose is not None:
            settings.DEEPEVAL_VERBOSE_MODE = verbose

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

        # Advanced
        if error_reporting is not None:
            settings.ERROR_REPORTING = error_reporting
        if ignore_errors is not None:
            settings.IGNORE_DEEPEVAL_ERRORS = ignore_errors

    handled, path, updated = edit_ctx.result

    if not updated:
        # no changes were made, so there is nothing to do.
        return

    if not handled and save is not None:
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(":loud_sound: Debug options updated.")


@app.command(name="unset-debug")
def unset_debug(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the debug-related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    """
    Restore default behavior by unsetting debug related variables.

    Behavior:
    - Resets LOG_LEVEL back to 'info'.
    - Unsets DEEPEVAL_VERBOSE_MODE, retry log-level overrides, gRPC and Confident trace flags.
    - If --save is provided (or DEEPEVAL_DEFAULT_SAVE is set), removes these keys from the target dotenv file.
    """
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        # Back to normal global level
        settings.LOG_LEVEL = "info"
        settings.CONFIDENT_TRACE_ENVIRONMENT = "development"
        settings.CONFIDENT_TRACE_VERBOSE = True
        settings.CONFIDENT_METRIC_LOGGING_VERBOSE = True
        settings.CONFIDENT_METRIC_LOGGING_ENABLED = True

        # Clear optional toggles/overrides
        settings.DEEPEVAL_VERBOSE_MODE = None
        settings.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL = None
        settings.DEEPEVAL_RETRY_AFTER_LOG_LEVEL = None

        settings.DEEPEVAL_GRPC_LOGGING = None
        settings.GRPC_VERBOSITY = None
        settings.GRPC_TRACE = None

        settings.CONFIDENT_TRACE_FLUSH = None
        settings.CONFIDENT_METRIC_LOGGING_FLUSH = None

        settings.ERROR_REPORTING = None
        settings.IGNORE_DEEPEVAL_ERRORS = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        print(f"Removed debug-related environment variables from {path}.")
    else:
        print("Debug settings reverted to defaults for this session.")

    print(":mute: Debug options unset.")


#############################################
# OpenAI Integration ########################
#############################################


@app.command(name="set-openai")
def set_openai_env(
    model: str = typer.Option(..., "--model", help="OpenAI model name"),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "--cost_per_input_token",
        help=(
            "USD per input token. Optional for known OpenAI models (pricing is preconfigured); "
            "REQUIRED if you use a custom/unsupported model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "--cost_per_output_token",
        help=(
            "USD per output token. Optional for known OpenAI models (pricing is preconfigured); "
            "REQUIRED if you use a custom/unsupported model."
        ),
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
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
      `--cost_per_input_token` and `--cost_per_output_token`.

    Secrets & saving:
    - Set your `OPENAI_API_KEY` via environment or a dotenv file.
    - Pass `--save=dotenv[:path]` to write configuration to a dotenv file
      (default: `.env.local`). This command does not set or persist OPENAI_API_KEY. Set it
      via your environment or a dotenv file (e.g., add OPENAI_API_KEY=... to .env.local)
      before running this command, or manage it with whatever command you use for secrets.

    Args:
        model: OpenAI model name, such as `gpt-4o-mini`.
        cost_per_input_token: USD per input token (optional for known models).
        cost_per_output_token: USD per output token (optional for known models).
        save: Persist config (and supported secrets) to a dotenv file; format `dotenv[:path]`.

    Example:
        deepeval set-openai \\
          --model gpt-4o-mini \\
          --cost_per_input_token 0.0005 \\
          --cost_per_output_token 0.0015 \\
          --save dotenv:.env.local
    """
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_OPENAI_MODEL)
        settings.OPENAI_MODEL_NAME = model
        if cost_per_input_token is not None:
            settings.OPENAI_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.OPENAI_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using OpenAI's `{escape(model)}` for all evals that require an LLM."
    )


@app.command(name="unset-openai")
def unset_openai_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the OpenAI related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
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

    Example:
        deepeval unset-openai --save dotenv:.env.local
    """

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.OPENAI_MODEL_NAME = None
        settings.OPENAI_COST_PER_INPUT_TOKEN = None
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = None
        settings.USE_OPENAI_MODEL = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed OpenAI environment variables from {path}.")

    if is_openai_configured():
        print(
            ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
        )
    else:
        print(
            "OpenAI has been unset. No active provider is configured. Set one with the CLI, or add credentials to .env[.local]."
        )


#############################################
# Azure Integration ########################
#############################################


@app.command(name="set-azure-openai")
def set_azure_openai_env(
    azure_openai_api_key: str = typer.Option(
        ...,
        "--openai-api-key",
        help="Azure OpenAI API key",
    ),
    azure_openai_endpoint: str = typer.Option(
        ..., "--openai-endpoint", help="Azure OpenAI endpoint"
    ),
    openai_api_version: str = typer.Option(
        ..., "--openai-api-version", help="OpenAI API version"
    ),
    openai_model_name: str = typer.Option(
        ..., "--openai-model-name", help="OpenAI model name"
    ),
    azure_deployment_name: str = typer.Option(
        ..., "--deployment-name", help="Azure deployment name"
    ),
    azure_model_version: Optional[str] = typer.Option(
        None, "--model-version", help="Azure model version (optional)"
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_AZURE_OPENAI)
        settings.AZURE_OPENAI_API_KEY = azure_openai_api_key
        settings.AZURE_OPENAI_ENDPOINT = azure_openai_endpoint
        settings.OPENAI_API_VERSION = openai_api_version
        settings.AZURE_DEPLOYMENT_NAME = azure_deployment_name
        settings.AZURE_MODEL_NAME = openai_model_name
        if azure_model_version is not None:
            settings.AZURE_MODEL_VERSION = azure_model_version

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using Azure OpenAI's `{escape(openai_model_name)}` for all evals that require an LLM."
    )


@app.command(name="set-azure-openai-embedding")
def set_azure_openai_embedding_env(
    azure_embedding_deployment_name: str = typer.Option(
        ...,
        "--embedding-deployment-name",
        help="Azure embedding deployment name",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(
            EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING
        )
        settings.AZURE_EMBEDDING_DEPLOYMENT_NAME = (
            azure_embedding_deployment_name
        )

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        ":raising_hands: Congratulations! You're now using Azure OpenAI Embeddings within DeepEval."
    )


@app.command(name="unset-azure-openai")
def unset_azure_openai_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the Azure OpenAI–related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    )
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.AZURE_OPENAI_API_KEY = None
        settings.AZURE_OPENAI_ENDPOINT = None
        settings.OPENAI_API_VERSION = None
        settings.AZURE_DEPLOYMENT_NAME = None
        settings.AZURE_MODEL_NAME = None
        settings.AZURE_MODEL_VERSION = None
        settings.USE_AZURE_OPENAI = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed Azure OpenAI environment variables from {path}.")

    if is_openai_configured():
        print(
            ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
        )
    else:
        print(
            "Azure OpenAI has been unset. No active provider is configured. Set one with the CLI, or add credentials to .env[.local]."
        )


@app.command(name="unset-azure-openai-embedding")
def unset_azure_openai_embedding_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the Azure OpenAI embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.AZURE_EMBEDDING_DEPLOYMENT_NAME = None
        settings.USE_AZURE_OPENAI_EMBEDDING = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Removed Azure OpenAI embedding environment variables from {path}."
        )

    if is_openai_configured():
        print(
            ":raised_hands: Regular OpenAI embeddings will still be used by default because OPENAI_API_KEY is set."
        )
    else:
        print(
            "The Azure OpenAI embedding model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
        )


#############################################
# Ollama Integration ########################
#############################################


@app.command(name="set-ollama")
def set_ollama_model_env(
    model_name: str = typer.Argument(..., help="Name of the Ollama model"),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "-b",
        "--base-url",
        help="Base URL for the local model API",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_LOCAL_MODEL)
        settings.LOCAL_MODEL_API_KEY = "ollama"
        settings.LOCAL_MODEL_NAME = model_name
        settings.LOCAL_MODEL_BASE_URL = base_url

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using a local Ollama model `{escape(model_name)}` for all evals that require an LLM."
    )


@app.command(name="unset-ollama")
def unset_ollama_model_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the Ollama related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.LOCAL_MODEL_API_KEY = None
        settings.LOCAL_MODEL_NAME = None
        settings.LOCAL_MODEL_BASE_URL = None
        settings.USE_LOCAL_MODEL = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed local Ollama environment variables from {path}.")

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
    model_name: str = typer.Argument(
        ..., help="Name of the Ollama embedding model"
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "-b",
        "--base-url",
        help="Base URL for the Ollama embedding model API",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
        settings.LOCAL_EMBEDDING_API_KEY = "ollama"
        settings.LOCAL_EMBEDDING_MODEL_NAME = model_name
        settings.LOCAL_EMBEDDING_BASE_URL = base_url

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using the Ollama embedding model `{escape(model_name)}` for all evals that require text embeddings."
    )


@app.command(name="unset-ollama-embeddings")
def unset_ollama_embeddings_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the Ollama embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.LOCAL_EMBEDDING_API_KEY = None
        settings.LOCAL_EMBEDDING_MODEL_NAME = None
        settings.LOCAL_EMBEDDING_BASE_URL = None
        settings.USE_LOCAL_EMBEDDINGS = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Removed local Ollama embedding environment variables from {path}."
        )

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
    model_name: str = typer.Option(
        ..., "--model-name", help="Name of the local model"
    ),
    base_url: str = typer.Option(
        ..., "--base-url", help="Base URL for the local model API"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for the local model. Persisted to dotenv if --save is used; never written to the legacy JSON keystore.",
    ),
    model_format: Optional[str] = typer.Option(
        "json",
        "--format",
        help="Format of the response from the local model (default: json)",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_LOCAL_MODEL)
        settings.LOCAL_MODEL_NAME = model_name
        settings.LOCAL_MODEL_BASE_URL = base_url
        if model_format:
            settings.LOCAL_MODEL_FORMAT = model_format
        if api_key:
            settings.LOCAL_MODEL_API_KEY = api_key

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using a local model `{escape(model_name)}` for all evals that require an LLM."
    )


@app.command(name="unset-local-model")
def unset_local_model_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the local model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.LOCAL_MODEL_API_KEY = None
        settings.LOCAL_MODEL_NAME = None
        settings.LOCAL_MODEL_BASE_URL = None
        settings.LOCAL_MODEL_FORMAT = None
        settings.USE_LOCAL_MODEL = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed local model environment variables from {path}.")

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
    model_name: str = typer.Option(
        ..., "--model-name", help="Name of the Grok model"
    ),
    api_key: str = typer.Option(
        ...,
        "--api-key",
        help="API key for the Grok model. Persisted to dotenv if --save is used; never written to the legacy JSON keystore.",
    ),
    temperature: float = typer.Option(
        0, "--temperature", help="Temperature for the Grok model"
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_GROK_MODEL)
        settings.GROK_API_KEY = api_key
        settings.GROK_MODEL_NAME = model_name
        settings.TEMPERATURE = temperature

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using a Grok's `{escape(model_name)}` for all evals that require an LLM."
    )


@app.command(name="unset-grok")
def unset_grok_model_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the Grok model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.GROK_API_KEY = None
        settings.GROK_MODEL_NAME = None
        settings.TEMPERATURE = None
        settings.USE_GROK_MODEL = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed Grok model environment variables from {path}.")

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
    model_name: str = typer.Option(
        ..., "--model-name", help="Name of the Moonshot model"
    ),
    api_key: str = typer.Option(
        ...,
        "--api-key",
        help="API key for the Moonshot model. Persisted to dotenv if --save is used; never written to the legacy JSON keystore.",
    ),
    temperature: float = typer.Option(
        0, "--temperature", help="Temperature for the Moonshot model"
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_MOONSHOT_MODEL)
        settings.MOONSHOT_API_KEY = api_key
        settings.MOONSHOT_MODEL_NAME = model_name
        settings.TEMPERATURE = temperature

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using Moonshot's `{escape(model_name)}` for all evals that require an LLM."
    )


@app.command(name="unset-moonshot")
def unset_moonshot_model_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the Moonshot model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.MOONSHOT_API_KEY = None
        settings.MOONSHOT_MODEL_NAME = None
        settings.TEMPERATURE = None
        settings.USE_MOONSHOT_MODEL = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed Moonshot model environment variables from {path}.")

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
    model_name: str = typer.Option(
        ..., "--model-name", help="Name of the DeepSeek model"
    ),
    api_key: str = typer.Option(
        ...,
        "--api-key",
        help="API key for the DeepSeek model. Persisted to dotenv if --save is used; never written to the legacy JSON keystore.",
    ),
    temperature: float = typer.Option(
        0, "--temperature", help="Temperature for the DeepSeek model"
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_DEEPSEEK_MODEL)
        settings.DEEPSEEK_API_KEY = api_key
        settings.DEEPSEEK_MODEL_NAME = model_name
        settings.TEMPERATURE = temperature

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using DeepSeek's `{escape(model_name)}` for all evals that require an LLM."
    )


@app.command(name="unset-deepseek")
def unset_deepseek_model_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the DeepSeek model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.DEEPSEEK_API_KEY = None
        settings.DEEPSEEK_MODEL_NAME = None
        settings.TEMPERATURE = None
        settings.USE_DEEPSEEK_MODEL = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed DeepSeek model environment variables from {path}.")

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
    model_name: str = typer.Option(
        ..., "--model-name", help="Name of the local embeddings model"
    ),
    base_url: str = typer.Option(
        ..., "--base-url", help="Base URL for the local embeddings API"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for the local embeddings. Persisted to dotenv if --save is used; never written to the legacy JSON keystore.",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
        settings.LOCAL_EMBEDDING_MODEL_NAME = model_name
        settings.LOCAL_EMBEDDING_BASE_URL = base_url
        if api_key:
            settings.LOCAL_EMBEDDING_API_KEY = api_key

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using the local embedding model `{escape(model_name)}` for all evals that require text embeddings."
    )


@app.command(name="unset-local-embeddings")
def unset_local_embeddings_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the local embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.LOCAL_EMBEDDING_API_KEY = None
        settings.LOCAL_EMBEDDING_MODEL_NAME = None
        settings.LOCAL_EMBEDDING_BASE_URL = None
        settings.USE_LOCAL_EMBEDDINGS = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed local embedding environment variables from {path}.")

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
    model_name: Optional[str] = typer.Option(
        None, "--model-name", help="Gemini Model name"
    ),
    google_api_key: Optional[str] = typer.Option(
        None,
        "--google-api-key",
        help="Google API Key for Gemini",
    ),
    google_cloud_project: Optional[str] = typer.Option(
        None, "--project-id", help="Google Cloud project ID"
    ),
    google_cloud_location: Optional[str] = typer.Option(
        None, "--location", help="Google Cloud location"
    ),
    google_service_account_key: Optional[str] = typer.Option(
        None,
        "--service-account-key",
        help="Google Service Account Key for Gemini",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    if not google_api_key and not (
        google_cloud_project and google_cloud_location
    ):
        typer.echo(
            "You must provide either --google-api-key or both --project-id and --location.",
            err=True,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_GEMINI_MODEL)

        if google_api_key is not None:
            settings.GOOGLE_API_KEY = google_api_key
            settings.GOOGLE_GENAI_USE_VERTEXAI = False
        else:
            settings.GOOGLE_GENAI_USE_VERTEXAI = True
        if google_cloud_project:
            settings.GOOGLE_CLOUD_PROJECT = google_cloud_project
        if google_cloud_location:
            settings.GOOGLE_CLOUD_LOCATION = google_cloud_location
        if google_service_account_key:
            settings.GOOGLE_SERVICE_ACCOUNT_KEY = google_service_account_key
        if model_name:
            settings.GEMINI_MODEL_NAME = model_name

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    _model_name = (
        model_name if model_name is not None else settings.GEMINI_MODEL_NAME
    )
    if _model_name is not None:
        print(
            f":raising_hands: Congratulations! You're now using Gemini's `{escape(_model_name)}` for all evals that require an LLM."
        )
    else:
        print(
            ":raising_hands: Congratulations! You're now using Gemini's model for all evals that require an LLM."
        )


@app.command(name="unset-gemini")
def unset_gemini_model_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the Gemini related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.GOOGLE_API_KEY = None
        settings.GOOGLE_GENAI_USE_VERTEXAI = None
        settings.GOOGLE_CLOUD_PROJECT = None
        settings.GOOGLE_CLOUD_LOCATION = None
        settings.GEMINI_MODEL_NAME = None
        settings.USE_GEMINI_MODEL = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed Gemini model environment variables from {path}.")

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
    model_name: str = typer.Argument(..., help="Name of the LiteLLM model"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for the model. Persisted to dotenv if --save is used; never written to the legacy JSON keystore.",
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="Base URL for the model API (if required)"
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_LITELLM)
        settings.LITELLM_MODEL_NAME = model_name
        if api_key is not None:
            settings.LITELLM_API_KEY = api_key
        if api_base is not None:
            settings.LITELLM_API_BASE = api_base

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(
            f"Saved environment variables to {path} (ensure it's git-ignored)."
        )
    else:
        # updated in-memory & process env only
        print(
            "Settings updated for this session. To persist, use --save=dotenv[:path] "
            "(default .env.local) or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        f":raising_hands: Congratulations! You're now using LiteLLM's `{escape(model_name)}` for all evals that require an LLM."
    )


@app.command(name="unset-litellm")
def unset_litellm_model_env(
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Remove only the LiteLLM related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.LITELLM_API_KEY = None
        settings.LITELLM_MODEL_NAME = None
        settings.LITELLM_API_BASE = None
        settings.USE_LITELLM = None

    handled, path, _ = edit_ctx.result

    if not handled and save is not None:
        # invalid --save format (unsupported)
        print("Unsupported --save option. Use --save=dotenv[:path].")
    elif path:
        # persisted to a file
        print(f"Removed LiteLLM model environment variables from {path}.")

    if is_openai_configured():
        print(
            ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
        )
    else:
        print(
            "The LiteLLM model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
        )


if __name__ == "__main__":
    app()
