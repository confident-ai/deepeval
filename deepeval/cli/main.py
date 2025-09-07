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
import webbrowser
import threading
import random
import string
import socket
import typer
from enum import Enum
from deepeval.key_handler import (
    KEY_FILE_HANDLER,
    KeyValues,
    EmbeddingKeyValues,
    ModelKeyValues,
)
from deepeval.telemetry import capture_login_event, capture_view_event
from deepeval.cli.test import app as test_app
from deepeval.cli.server import start_server
from deepeval.utils import delete_file_if_exists, open_browser
from deepeval.test_run.test_run import (
    LATEST_TEST_RUN_FILE_PATH,
    global_test_run_manager,
)
from deepeval.cli.utils import (
    render_login_message,
    upload_and_open_link,
    PROD,
    clear_evaluation_model_keys,
    clear_embedding_model_keys,
    resolve_save_target,
    save_environ_to_store,
    unset_environ_in_store,
    switch_model_provider,
)
from deepeval.confident.api import (
    get_confident_api_key,
    is_confident,
    set_confident_api_key,
    CONFIDENT_API_KEY_ENV_VAR,
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
    api_key = os.getenv("OPENAI_API_KEY") or KEY_FILE_HANDLER.fetch_data(
        ModelKeyValues.OPENAI_API_KEY
    )
    return bool(api_key)


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
    KEY_FILE_HANDLER.write_key(KeyValues.CONFIDENT_REGION, region.value)
    save_target = resolve_save_target(save)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                KeyValues.CONFIDENT_REGION: region.value,
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")
    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
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

            save_target = resolve_save_target(save) or "dotenv:.env.local"
            handled, path = save_environ_to_store(
                save_target,
                {KeyValues.API_KEY: key, CONFIDENT_API_KEY_ENV_VAR: key},
            )
            if handled:
                print(
                    f"Saved environment variables to {path} (ensure it's git-ignored)."
                )
            else:
                print("Unsupported --save option. Use --save=dotenv[:path].")

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
        help="Where to remove the saved key from. Use format dotenv[:path]. If omitted, logout removes from .env.local. JSON keystore is always cleared.",
    )
):
    """
    Log out of Confident AI.

    Behavior:
    - Always clears the Confident API key from the JSON keystore and process env.
    - Also removes credentials from a dotenv file; defaults to .env.local.
      Override the target with --save=dotenv[:path].
    """
    set_confident_api_key(None)

    # Remove from dotenv file (both names)
    save_target = resolve_save_target(save) or "dotenv:.env.local"
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                KeyValues.API_KEY,
                CONFIDENT_API_KEY_ENV_VAR,
            ],
        )
        if handled:
            print(f"Removed Confident AI key(s) from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")
    else:
        print(
            "Tip: remove keys from a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

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


@app.command(name="enable-grpc-logging")
def enable_grpc_logging():
    os.environ["DEEPEVAL_GRPC_LOGGING"] = "1"


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
    - secrets are ever written to `.deepeval/.deepeval` (JSON).

    Pricing rules:
    - If `model` is a known OpenAI model, you may omit costs (built‑in pricing is used).
    - If `model` is custom/unsupported, you must provide both
      `--cost_per_input_token` and `--cost_per_output_token`.

    Secrets & saving:
    - Set your `OPENAI_API_KEY` via environment or a dotenv file.
    - Pass `--save=dotenv[:path]` to write configuration to a dotenv file
      (default: `.env.local`). Supported secrets, such as `OPENAI_API_KEY`, are
      persisted there if present in your environment.

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

    clear_evaluation_model_keys()
    KEY_FILE_HANDLER.write_key(ModelKeyValues.OPENAI_MODEL_NAME, model)
    if cost_per_input_token is not None:
        KEY_FILE_HANDLER.write_key(
            ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
            str(cost_per_input_token),
        )
    if cost_per_output_token is not None:
        KEY_FILE_HANDLER.write_key(
            ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
            str(cost_per_output_token),
        )

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_OPENAI_MODEL, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.OPENAI_MODEL_NAME: model,
                **(
                    {
                        ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN: str(
                            cost_per_input_token
                        )
                    }
                    if cost_per_input_token is not None
                    else {}
                ),
                **(
                    {
                        ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN: str(
                            cost_per_output_token
                        )
                    }
                    if cost_per_output_token is not None
                    else {}
                ),
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )
    print(
        f":raising_hands: Congratulations! You're now using OpenAI's `{model}` for all evals that require an LLM."
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

    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_OPENAI_MODEL)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.OPENAI_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.USE_OPENAI_MODEL,
                ModelKeyValues.OPENAI_MODEL_NAME,
                ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN,
                ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN,
            ],
        )
        if handled:
            print(f"Removed OpenAI environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

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
        help="Azure OpenAI API key (NOT persisted; set in .env[.local])",
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

    clear_evaluation_model_keys()

    KEY_FILE_HANDLER.write_key(
        ModelKeyValues.AZURE_MODEL_NAME, openai_model_name
    )
    KEY_FILE_HANDLER.write_key(
        ModelKeyValues.AZURE_OPENAI_ENDPOINT, azure_openai_endpoint
    )
    KEY_FILE_HANDLER.write_key(
        ModelKeyValues.OPENAI_API_VERSION, openai_api_version
    )
    KEY_FILE_HANDLER.write_key(
        ModelKeyValues.AZURE_DEPLOYMENT_NAME, azure_deployment_name
    )

    if azure_model_version is not None:
        KEY_FILE_HANDLER.write_key(
            ModelKeyValues.AZURE_MODEL_VERSION, azure_model_version
        )

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_AZURE_OPENAI, save_target)

    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.AZURE_OPENAI_API_KEY: azure_openai_api_key,
                ModelKeyValues.AZURE_OPENAI_ENDPOINT: azure_openai_endpoint,
                ModelKeyValues.OPENAI_API_VERSION: openai_api_version,
                ModelKeyValues.AZURE_DEPLOYMENT_NAME: azure_deployment_name,
                ModelKeyValues.AZURE_MODEL_NAME: openai_model_name,
                **(
                    {ModelKeyValues.AZURE_MODEL_VERSION: azure_model_version}
                    if azure_model_version
                    else {}
                ),
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        ":raising_hands: Congratulations! You're now using Azure OpenAI for all evals that require an LLM."
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
    clear_embedding_model_keys()
    KEY_FILE_HANDLER.write_key(
        EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME,
        azure_embedding_deployment_name,
    )

    save_target = resolve_save_target(save)
    switch_model_provider(
        EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING, save_target
    )
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME: azure_embedding_deployment_name,
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.AZURE_OPENAI_API_KEY)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.AZURE_OPENAI_ENDPOINT)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.OPENAI_API_VERSION)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.AZURE_DEPLOYMENT_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.AZURE_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.AZURE_MODEL_VERSION)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_AZURE_OPENAI)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.AZURE_OPENAI_API_KEY,
                ModelKeyValues.AZURE_OPENAI_ENDPOINT,
                ModelKeyValues.OPENAI_API_VERSION,
                ModelKeyValues.AZURE_DEPLOYMENT_NAME,
                ModelKeyValues.AZURE_MODEL_NAME,
                ModelKeyValues.AZURE_MODEL_VERSION,
                ModelKeyValues.USE_AZURE_OPENAI,
            ],
        )
        if handled:
            print(f"Removed Azure OpenAI environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    if is_openai_configured():
        print(
            ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
        )
    else:
        print(
            "Azure OpenAI configuration removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
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
    KEY_FILE_HANDLER.remove_key(
        EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME
    )
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING,
            ],
        )
        if handled:
            print(
                f"Removed Azure OpenAI embedding environment variables from {path}."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

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
    clear_evaluation_model_keys()
    KEY_FILE_HANDLER.write_key(ModelKeyValues.LOCAL_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(ModelKeyValues.LOCAL_MODEL_BASE_URL, base_url)

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_LOCAL_MODEL, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.LOCAL_MODEL_NAME: model_name,
                ModelKeyValues.LOCAL_MODEL_BASE_URL: base_url,
                ModelKeyValues.LOCAL_MODEL_API_KEY: "ollama",
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        ":raising_hands: Congratulations! You're now using a local Ollama model for all evals that require an LLM."
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LOCAL_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LOCAL_MODEL_BASE_URL)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_LOCAL_MODEL)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LOCAL_MODEL_API_KEY)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.LOCAL_MODEL_NAME,
                ModelKeyValues.LOCAL_MODEL_BASE_URL,
                ModelKeyValues.USE_LOCAL_MODEL,
                ModelKeyValues.LOCAL_MODEL_API_KEY,
            ],
        )
        if handled:
            print(f"Removed Ollama environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    if is_openai_configured():
        print(
            ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
        )
    else:
        print(
            "local Ollama model configuration removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
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
    clear_embedding_model_keys()
    KEY_FILE_HANDLER.write_key(
        EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME, model_name
    )
    KEY_FILE_HANDLER.write_key(
        EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL, base_url
    )

    save_target = resolve_save_target(save)
    switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: model_name,
                EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: base_url,
                EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY: "ollama",
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        ":raising_hands: Congratulations! You're now using Ollama embeddings for all evals that require text embeddings."
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
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL)
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY)
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME,
                EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL,
                EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY,
                EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
            ],
        )
        if handled:
            print(
                f"Removed Ollama embedding environment variables from {path}."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    if is_openai_configured():
        print(
            ":raised_hands: Regular OpenAI will still be used by default because OPENAI_API_KEY is set."
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
        help="API key for the local model (if required) (NOT persisted; set in .env[.local])",
    ),
    format: Optional[str] = typer.Option(
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
    clear_evaluation_model_keys()
    KEY_FILE_HANDLER.write_key(ModelKeyValues.LOCAL_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(ModelKeyValues.LOCAL_MODEL_BASE_URL, base_url)

    if format:
        KEY_FILE_HANDLER.write_key(ModelKeyValues.LOCAL_MODEL_FORMAT, format)

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_LOCAL_MODEL, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.LOCAL_MODEL_NAME: model_name,
                ModelKeyValues.LOCAL_MODEL_BASE_URL: base_url,
                **(
                    {ModelKeyValues.LOCAL_MODEL_API_KEY: api_key}
                    if api_key
                    else {}
                ),
                **(
                    {ModelKeyValues.LOCAL_MODEL_FORMAT: format}
                    if format
                    else {}
                ),
            },
        )

        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        ":raising_hands: Congratulations! You're now using a local model for all evals that require an LLM."
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LOCAL_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LOCAL_MODEL_BASE_URL)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LOCAL_MODEL_API_KEY)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LOCAL_MODEL_FORMAT)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_LOCAL_MODEL)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.LOCAL_MODEL_NAME,
                ModelKeyValues.LOCAL_MODEL_BASE_URL,
                ModelKeyValues.USE_LOCAL_MODEL,
                ModelKeyValues.LOCAL_MODEL_API_KEY,
                ModelKeyValues.LOCAL_MODEL_FORMAT,
            ],
        )
        if handled:
            print(f"Removed local model environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")
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
        help="API key for the Grok model (NOT persisted; set in .env[.local])",
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
    clear_evaluation_model_keys()
    KEY_FILE_HANDLER.write_key(ModelKeyValues.GROK_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(ModelKeyValues.TEMPERATURE, str(temperature))

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_GROK_MODEL, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.GROK_MODEL_NAME: model_name,
                ModelKeyValues.GROK_API_KEY: api_key,
                ModelKeyValues.TEMPERATURE: str(temperature),
            },
        )

        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")
    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )
    print(
        ":raising_hands: Congratulations! You're now using a Grok model for all evals that require an LLM."
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.GROK_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.GROK_API_KEY)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.TEMPERATURE)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_GROK_MODEL)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.GROK_MODEL_NAME,
                ModelKeyValues.GROK_API_KEY,
                ModelKeyValues.TEMPERATURE,
                ModelKeyValues.USE_GROK_MODEL,
            ],
        )
        if handled:
            print(f"Removed Grok model environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

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
        help="API key for the Moonshot model (NOT persisted; set in .env[.local])",
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
    clear_evaluation_model_keys()
    KEY_FILE_HANDLER.write_key(ModelKeyValues.MOONSHOT_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(ModelKeyValues.TEMPERATURE, str(temperature))

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_MOONSHOT_MODEL, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.MOONSHOT_MODEL_NAME: model_name,
                ModelKeyValues.MOONSHOT_API_KEY: api_key,
                ModelKeyValues.TEMPERATURE: str(temperature),
            },
        )

        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )
    print(
        ":raising_hands: Congratulations! You're now using a Moonshot model for all evals that require an LLM."
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.MOONSHOT_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.MOONSHOT_API_KEY)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.TEMPERATURE)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_MOONSHOT_MODEL)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.MOONSHOT_MODEL_NAME,
                ModelKeyValues.MOONSHOT_API_KEY,
                ModelKeyValues.TEMPERATURE,
                ModelKeyValues.USE_MOONSHOT_MODEL,
            ],
        )
        if handled:
            print(f"Removed Moonshot model environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

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
        help="API key for the DeepSeek model (NOT persisted; set in .env[.local])",
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
    clear_evaluation_model_keys()
    KEY_FILE_HANDLER.write_key(ModelKeyValues.DEEPSEEK_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(ModelKeyValues.TEMPERATURE, str(temperature))

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_DEEPSEEK_MODEL, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.DEEPSEEK_MODEL_NAME: model_name,
                ModelKeyValues.DEEPSEEK_API_KEY: api_key,
                ModelKeyValues.TEMPERATURE: str(temperature),
            },
        )

        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )
    print(
        ":raising_hands: Congratulations! You're now using a DeepSeek model for all evals that require an LLM."
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.DEEPSEEK_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.DEEPSEEK_API_KEY)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.TEMPERATURE)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_DEEPSEEK_MODEL)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.DEEPSEEK_MODEL_NAME,
                ModelKeyValues.DEEPSEEK_API_KEY,
                ModelKeyValues.TEMPERATURE,
                ModelKeyValues.USE_DEEPSEEK_MODEL,
            ],
        )
        if handled:
            print(f"Removed DeepSeek model environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    if is_openai_configured():
        print(
            ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
        )
    else:
        print(
            "The Deepseek model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
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
        help="API key for the local embeddings (if required) (NOT persisted; set in .env[.local])",
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    clear_embedding_model_keys()
    KEY_FILE_HANDLER.write_key(
        EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME, model_name
    )
    KEY_FILE_HANDLER.write_key(
        EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL, base_url
    )

    save_target = resolve_save_target(save)
    switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME: model_name,
                EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL: base_url,
                **(
                    {EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY: api_key}
                    if api_key
                    else {}
                ),
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")
    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )
    print(
        ":raising_hands: Congratulations! You're now using local embeddings for all evals that require text embeddings."
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
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL)
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY)
    KEY_FILE_HANDLER.remove_key(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME,
                EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL,
                EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY,
                EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
            ],
        )
        if handled:
            print(f"Removed local embedding environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

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
        help="Google API Key for Gemini (NOT persisted; set in .env[.local])",
    ),
    google_cloud_project: Optional[str] = typer.Option(
        None, "--project-id", help="Google Cloud project ID"
    ),
    google_cloud_location: Optional[str] = typer.Option(
        None, "--location", help="Google Cloud location"
    ),
    save: Optional[str] = typer.Option(
        None,
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
):
    clear_evaluation_model_keys()
    if not google_api_key and not (
        google_cloud_project and google_cloud_location
    ):
        typer.echo(
            "You must provide either --google-api-key or both --project-id and --location.",
            err=True,
        )
        raise typer.Exit(code=1)
    if model_name is not None:
        KEY_FILE_HANDLER.write_key(ModelKeyValues.GEMINI_MODEL_NAME, model_name)

    if google_api_key is None:
        KEY_FILE_HANDLER.write_key(
            ModelKeyValues.GOOGLE_GENAI_USE_VERTEXAI, "YES"
        )

    if google_cloud_project is not None:
        KEY_FILE_HANDLER.write_key(
            ModelKeyValues.GOOGLE_CLOUD_PROJECT, google_cloud_project
        )
    if google_cloud_location is not None:
        KEY_FILE_HANDLER.write_key(
            ModelKeyValues.GOOGLE_CLOUD_LOCATION, google_cloud_location
        )

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_GEMINI_MODEL, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                **(
                    {ModelKeyValues.GOOGLE_API_KEY: google_api_key}
                    if google_api_key
                    else {ModelKeyValues.GOOGLE_GENAI_USE_VERTEXAI: "YES"}
                ),
                **(
                    {ModelKeyValues.GEMINI_MODEL_NAME: model_name}
                    if model_name
                    else {}
                ),
                **(
                    {ModelKeyValues.GOOGLE_CLOUD_PROJECT: google_cloud_project}
                    if google_cloud_project
                    else {}
                ),
                **(
                    {
                        ModelKeyValues.GOOGLE_CLOUD_LOCATION: google_cloud_location
                    }
                    if google_cloud_location
                    else {}
                ),
            },
        )

        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )

    print(
        ":raising_hands: Congratulations! You're now using a Gemini model for all evals that require an LLM."
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_GEMINI_MODEL)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.GEMINI_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.GOOGLE_API_KEY)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.GOOGLE_CLOUD_PROJECT)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.GOOGLE_CLOUD_LOCATION)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.GOOGLE_GENAI_USE_VERTEXAI)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.USE_GEMINI_MODEL,
                ModelKeyValues.GEMINI_MODEL_NAME,
                ModelKeyValues.GOOGLE_API_KEY,
                ModelKeyValues.GOOGLE_CLOUD_PROJECT,
                ModelKeyValues.GOOGLE_CLOUD_LOCATION,
                ModelKeyValues.GOOGLE_GENAI_USE_VERTEXAI,
            ],
        )
        if handled:
            print(f"Removed Gemini environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")

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
        help="API key for the model (if required) (NOT persisted; set in .env[.local])",
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
    clear_evaluation_model_keys()
    KEY_FILE_HANDLER.write_key(ModelKeyValues.LITELLM_MODEL_NAME, model_name)

    if api_base:
        KEY_FILE_HANDLER.write_key(ModelKeyValues.LITELLM_API_BASE, api_base)

    save_target = resolve_save_target(save)
    switch_model_provider(ModelKeyValues.USE_LITELLM, save_target)
    if save_target:
        handled, path = save_environ_to_store(
            save_target,
            {
                ModelKeyValues.LITELLM_MODEL_NAME: model_name,
                **(
                    {ModelKeyValues.LITELLM_API_KEY: api_key} if api_key else {}
                ),
                **(
                    {ModelKeyValues.LITELLM_API_BASE: api_base}
                    if api_base
                    else {}
                ),
            },
        )
        if handled:
            print(
                f"Saved environment variables to {path} (ensure it's git-ignored)."
            )
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")
    else:
        print(
            "Tip: persist these settings to a dotenv file with --save=dotenv[:path] (default .env.local) "
            "or set DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local"
        )
    print(
        ":raising_hands: Congratulations! You're now using a LiteLLM model for all evals that require an LLM."
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
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LITELLM_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LITELLM_API_KEY)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.LITELLM_API_BASE)
    KEY_FILE_HANDLER.remove_key(ModelKeyValues.USE_LITELLM)

    save_target = resolve_save_target(save)
    if save_target:
        handled, path = unset_environ_in_store(
            save_target,
            [
                ModelKeyValues.LITELLM_MODEL_NAME,
                ModelKeyValues.LITELLM_API_KEY,
                ModelKeyValues.LITELLM_API_BASE,
                ModelKeyValues.USE_LITELLM,
            ],
        )
        if handled:
            print(f"Removed LiteLLM environment variables from {path}.")
        else:
            print("Unsupported --save option. Use --save=dotenv[:path].")
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
