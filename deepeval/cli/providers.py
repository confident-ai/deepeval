"""Registry behind the per-provider ``set-*`` / ``unset-*`` CLI commands.

Every model provider deepeval can be pointed at contributes a matching pair of
commands, but the two halves share very different amounts of structure.

``unset-*`` is uniform: null a fixed list of settings attributes, null the
provider's secrets too when ``--clear-secrets`` is passed, then report what
happened. One implementation covers all of them and ``UNSET_PROVIDERS`` supplies
the data.

``set-*`` is not uniform. Providers take genuinely different options — Azure
needs a deployment name, Gemini a service-account file, LiteLLM a proxy URL — so
only the "API key, model name, per-token costs" family is generated here, from
``API_KEY_MODELS``. The rest stay hand-written in ``main.py``, where their real
options stay visible.

Wording is stored verbatim rather than derived from one noun per provider,
because no such noun exists: ``unset-ollama`` calls itself "Ollama" in its
``--save`` help, "local Ollama" in its confirmation line, and "local Ollama
model" in its fallback. The ``_save_help`` / ``_secrets_help`` / ``_removed_msg``
helpers below only factor out the parts that really are constant, and a spec
passes a literal string wherever a provider's phrasing breaks the pattern.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import typer
from rich import print
from rich.markup import escape

from deepeval.cli.utils import (
    coerce_blank_to_none,
    handle_save_result,
    is_openai_configured,
    load_service_account_key_file,
)
from deepeval.config.settings import get_settings
from deepeval.key_handler import EmbeddingKeyValues, ModelKeyValues


# Shared option help. These are identical across every provider command.
_QUIET_HELP = "Suppress printing to the terminal (useful for CI)."
_UNSET_OPENAI_ACTIVE = (
    ":raised_hands: OpenAI will still be used by default because "
    "OPENAI_API_KEY is set."
)
_SET_SAVE_HELP = (
    "Persist CLI parameters as environment variables in a dotenv file. "
    "Usage: --save=dotenv[:path] (default: .env.local)"
)
_MODEL_HELP = "Model identifier to use for this provider"
_MODEL_HELP_GPT = "Model identifier to use for this provider (e.g., `gpt-4.1`)."

# Sentinel so an Opt can distinguish "no override" from an override of None.
_UNSET = object()
_COST_INPUT_HELP = (
    "USD per input token override used for cost tracking. Preconfigured for "
    "known models; REQUIRED if you use a custom/unknown model."
)
_COST_OUTPUT_HELP = (
    "USD per output token override used for cost tracking. Preconfigured for "
    "known models; REQUIRED if you use a custom/unknown model."
)


def _prompt_api_key_help(secret_attr: str) -> str:
    return (
        f"Prompt for {secret_attr} (input hidden). Not suitable for CI. "
        "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to "
        "dotenv in plaintext."
    )


def _llm_success(subject: str) -> str:
    return (
        f":raising_hands: Congratulations! You're now using {subject} "
        "`{model}` for all evals that require an LLM."
    )


def _embedding_success(subject: str) -> str:
    return (
        f":raising_hands: Congratulations! You're now using {subject} "
        "`{model}` for all evals that require text embeddings."
    )


def _save_help(noun: str) -> str:
    return (
        f"Remove only the {noun} related environment variables from a dotenv "
        "file. Usage: --save=dotenv[:path] (default: .env.local)"
    )


def _secrets_help(*secret_attrs: str) -> str:
    return f"Also remove {' and '.join(secret_attrs)} from the dotenv store."


def _removed_msg(noun: str) -> str:
    return f"Removed {noun} environment variables from {{path}}."


def _no_model_msg(subject: str) -> str:
    return (
        f"The {subject} configuration has been removed. No model is currently "
        "configured, but you can set one with the CLI or add credentials to "
        ".env[.local]."
    )


def _unset_msg(subject: str) -> str:
    return (
        f"{subject} has been unset. No active provider is configured. Set one "
        "with the CLI, or add credentials to .env[.local]."
    )


_UNSET_OPENAI_HELP = """
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


@dataclass(frozen=True)
class UnsetProviderSpec:
    """Everything one ``unset-*`` command needs beyond the shared logic."""

    command: str
    settings_attrs: Tuple[str, ...]
    save_help: str
    updated_msg: str
    fallback_msg: str
    secret_attrs: Tuple[str, ...] = ()
    clear_secrets_help: Optional[str] = None
    openai_active_msg: str = _UNSET_OPENAI_ACTIVE
    help_text: Optional[str] = None


_UNSET_SPECS: Tuple[UnsetProviderSpec, ...] = (
    UnsetProviderSpec(
        command="unset-openai",
        settings_attrs=(
            "OPENAI_MODEL_NAME",
            "OPENAI_COST_PER_INPUT_TOKEN",
            "OPENAI_COST_PER_OUTPUT_TOKEN",
            "USE_OPENAI_MODEL",
        ),
        secret_attrs=("OPENAI_API_KEY",),
        save_help=_save_help("OpenAI"),
        clear_secrets_help=_secrets_help("OPENAI_API_KEY"),
        updated_msg=_removed_msg("OpenAI"),
        fallback_msg=_unset_msg("OpenAI"),
        help_text=_UNSET_OPENAI_HELP,
    ),
    UnsetProviderSpec(
        command="unset-azure-openai",
        settings_attrs=(
            "AZURE_OPENAI_ENDPOINT",
            "OPENAI_API_VERSION",
            "AZURE_DEPLOYMENT_NAME",
            "AZURE_MODEL_NAME",
            "AZURE_MODEL_VERSION",
            "USE_AZURE_OPENAI",
        ),
        secret_attrs=("AZURE_OPENAI_API_KEY",),
        save_help="Remove only the Azure OpenAI–related environment variables from a dotenv file. Usage: --save=dotenv[:path] (default: .env.local)",
        clear_secrets_help=_secrets_help("AZURE_OPENAI_API_KEY"),
        updated_msg=_removed_msg("Azure OpenAI"),
        fallback_msg=_unset_msg("Azure OpenAI"),
    ),
    UnsetProviderSpec(
        command="unset-azure-openai-embedding",
        settings_attrs=(
            "AZURE_EMBEDDING_MODEL_NAME",
            "AZURE_EMBEDDING_DEPLOYMENT_NAME",
            "USE_AZURE_OPENAI_EMBEDDING",
        ),
        save_help=_save_help("Azure OpenAI embedding"),
        updated_msg=_removed_msg("Azure OpenAI embedding"),
        fallback_msg=_unset_msg("Azure OpenAI embedding"),
    ),
    UnsetProviderSpec(
        command="unset-anthropic",
        settings_attrs=(
            "USE_ANTHROPIC_MODEL",
            "ANTHROPIC_MODEL_NAME",
            "ANTHROPIC_COST_PER_INPUT_TOKEN",
            "ANTHROPIC_COST_PER_OUTPUT_TOKEN",
        ),
        secret_attrs=("ANTHROPIC_API_KEY",),
        save_help=_save_help("Anthropic model"),
        clear_secrets_help=_secrets_help("ANTHROPIC_API_KEY"),
        updated_msg=_removed_msg("Anthropic model"),
        fallback_msg=_no_model_msg("Anthropic model"),
    ),
    UnsetProviderSpec(
        command="unset-bedrock",
        settings_attrs=(
            "USE_AWS_BEDROCK_MODEL",
            "AWS_BEDROCK_MODEL_NAME",
            "AWS_BEDROCK_REGION",
            "AWS_BEDROCK_COST_PER_INPUT_TOKEN",
            "AWS_BEDROCK_COST_PER_OUTPUT_TOKEN",
        ),
        secret_attrs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        save_help=_save_help("AWS Bedrock model"),
        clear_secrets_help="Also remove AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY  from the dotenv store.",
        updated_msg=_removed_msg("AWS Bedrock model"),
        fallback_msg=_no_model_msg("AWS Bedrock model"),
    ),
    UnsetProviderSpec(
        command="unset-ollama",
        settings_attrs=(
            "OLLAMA_MODEL_NAME",
            "LOCAL_MODEL_BASE_URL",
            "USE_LOCAL_MODEL",
        ),
        secret_attrs=("LOCAL_MODEL_API_KEY",),
        save_help=_save_help("Ollama"),
        clear_secrets_help=_secrets_help("LOCAL_MODEL_API_KEY"),
        updated_msg=_removed_msg("local Ollama"),
        fallback_msg=_no_model_msg("local Ollama model"),
    ),
    UnsetProviderSpec(
        command="unset-ollama-embeddings",
        settings_attrs=(
            "LOCAL_EMBEDDING_MODEL_NAME",
            "LOCAL_EMBEDDING_BASE_URL",
            "USE_LOCAL_EMBEDDINGS",
        ),
        secret_attrs=("LOCAL_EMBEDDING_API_KEY",),
        save_help=_save_help("Ollama embedding"),
        clear_secrets_help=_secrets_help("LOCAL_EMBEDDING_API_KEY"),
        updated_msg=_removed_msg("local Ollama embedding"),
        fallback_msg=_no_model_msg("local Ollama embedding model"),
        openai_active_msg=":raised_hands: Regular OpenAI embeddings will still be used by default because OPENAI_API_KEY is set.",
    ),
    UnsetProviderSpec(
        command="unset-local-model",
        settings_attrs=(
            "LOCAL_MODEL_NAME",
            "LOCAL_MODEL_BASE_URL",
            "LOCAL_MODEL_FORMAT",
            "USE_LOCAL_MODEL",
        ),
        secret_attrs=("LOCAL_MODEL_API_KEY",),
        save_help=_save_help("local model"),
        clear_secrets_help=_secrets_help("LOCAL_MODEL_API_KEY"),
        updated_msg=_removed_msg("local model"),
        fallback_msg=_no_model_msg("local model"),
    ),
    UnsetProviderSpec(
        command="unset-grok",
        settings_attrs=(
            "USE_GROK_MODEL",
            "GROK_MODEL_NAME",
            "GROK_COST_PER_INPUT_TOKEN",
            "GROK_COST_PER_OUTPUT_TOKEN",
        ),
        secret_attrs=("GROK_API_KEY",),
        save_help=_save_help("Grok model"),
        clear_secrets_help=_secrets_help("GROK_API_KEY"),
        updated_msg=_removed_msg("Grok model"),
        fallback_msg=_no_model_msg("Grok model"),
    ),
    UnsetProviderSpec(
        command="unset-moonshot",
        settings_attrs=(
            "USE_MOONSHOT_MODEL",
            "MOONSHOT_MODEL_NAME",
            "MOONSHOT_COST_PER_INPUT_TOKEN",
            "MOONSHOT_COST_PER_OUTPUT_TOKEN",
        ),
        secret_attrs=("MOONSHOT_API_KEY",),
        save_help=_save_help("Moonshot model"),
        clear_secrets_help=_secrets_help("MOONSHOT_API_KEY"),
        updated_msg=_removed_msg("Moonshot model"),
        fallback_msg=_no_model_msg("Moonshot model"),
    ),
    UnsetProviderSpec(
        command="unset-deepseek",
        settings_attrs=(
            "USE_DEEPSEEK_MODEL",
            "DEEPSEEK_MODEL_NAME",
            "DEEPSEEK_COST_PER_INPUT_TOKEN",
            "DEEPSEEK_COST_PER_OUTPUT_TOKEN",
        ),
        secret_attrs=("DEEPSEEK_API_KEY",),
        save_help=_save_help("DeepSeek model"),
        clear_secrets_help=_secrets_help("DEEPSEEK_API_KEY"),
        updated_msg=_removed_msg("DeepSeek model"),
        fallback_msg=_no_model_msg("DeepSeek model"),
    ),
    UnsetProviderSpec(
        command="unset-local-embeddings",
        settings_attrs=(
            "LOCAL_EMBEDDING_MODEL_NAME",
            "LOCAL_EMBEDDING_BASE_URL",
            "USE_LOCAL_EMBEDDINGS",
        ),
        secret_attrs=("LOCAL_EMBEDDING_API_KEY",),
        save_help=_save_help("local embedding"),
        clear_secrets_help="Also remove LOCAL_MODEL_API_KEY from the dotenv store.",
        updated_msg=_removed_msg("local embedding"),
        fallback_msg=_no_model_msg("local embeddings model"),
    ),
    UnsetProviderSpec(
        command="unset-gemini",
        settings_attrs=(
            "USE_GEMINI_MODEL",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GEMINI_MODEL_NAME",
        ),
        secret_attrs=(
            "GOOGLE_API_KEY",
            "GOOGLE_SERVICE_ACCOUNT_KEY",
        ),
        save_help=_save_help("Gemini"),
        clear_secrets_help=_secrets_help(
            "GOOGLE_API_KEY", "GOOGLE_SERVICE_ACCOUNT_KEY"
        ),
        updated_msg=_removed_msg("Gemini model"),
        fallback_msg=_no_model_msg("Gemini model"),
    ),
    UnsetProviderSpec(
        command="unset-litellm",
        settings_attrs=(
            "USE_LITELLM",
            "LITELLM_MODEL_NAME",
            "LITELLM_API_BASE",
            "LITELLM_PROXY_API_BASE",
        ),
        secret_attrs=(
            "LITELLM_API_KEY",
            "LITELLM_PROXY_API_KEY",
        ),
        save_help=_save_help("LiteLLM"),
        clear_secrets_help=_secrets_help(
            "LITELLM_API_KEY", "LITELLM_PROXY_API_KEY"
        ),
        updated_msg=_removed_msg("LiteLLM model"),
        fallback_msg=_no_model_msg("LiteLLM model"),
    ),
    UnsetProviderSpec(
        command="unset-portkey",
        settings_attrs=(
            "USE_PORTKEY_MODEL",
            "PORTKEY_MODEL_NAME",
            "PORTKEY_BASE_URL",
            "PORTKEY_PROVIDER_NAME",
        ),
        secret_attrs=("PORTKEY_API_KEY",),
        save_help=_save_help("Portkey"),
        clear_secrets_help=_secrets_help("PORTKEY_API_KEY"),
        updated_msg=_removed_msg("Portkey model"),
        fallback_msg=_no_model_msg("Portkey model"),
    ),
    UnsetProviderSpec(
        command="unset-openrouter",
        settings_attrs=(
            "USE_OPENROUTER_MODEL",
            "OPENROUTER_MODEL_NAME",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_COST_PER_INPUT_TOKEN",
            "OPENROUTER_COST_PER_OUTPUT_TOKEN",
        ),
        secret_attrs=("OPENROUTER_API_KEY",),
        save_help=_save_help("OpenRouter model"),
        clear_secrets_help=_secrets_help("OPENROUTER_API_KEY"),
        updated_msg=_removed_msg("OpenRouter model"),
        fallback_msg=_no_model_msg("OpenRouter model"),
    ),
)

UNSET_PROVIDERS = {spec.command: spec for spec in _UNSET_SPECS}


_SET_OPENAI_HELP = """
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


# ---------------------------------------------------------------------------
# ``set-*`` commands
#
# Providers take different options, but they are drawn from a shared
# vocabulary: nearly every one accepts ``--model``, most accept ``--base-url``,
# several prompt for an API key. ``_OPTION_CATALOGUE`` holds that vocabulary —
# one entry per option, fixing its flags, type and default help — and a
# ``SetProviderSpec`` just lists the options it takes and the settings
# attribute each one writes.
#
# The command callback is then built with a signature synthesized from that
# list, so ``--help`` shows exactly the options the provider really supports
# rather than a union of everything.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OptionDef:
    """Fixed identity of one CLI option: its flags, type and default help."""

    flags: Tuple[str, ...]
    annotation: Any
    default: Any = None
    help: str = ""
    extra: Optional[Dict[str, Any]] = None


_OPTION_CATALOGUE: Dict[str, _OptionDef] = {
    "model": _OptionDef(("-m", "--model"), Optional[str], help=_MODEL_HELP),
    "base_url": _OptionDef(
        ("-u", "--base-url"),
        Optional[str],
        help="Override the API endpoint/base URL used by this provider.",
    ),
    "api_version": _OptionDef(
        ("-v", "--api-version"),
        Optional[str],
        help="Azure OpenAI API version (passed to the Azure OpenAI client).",
    ),
    "model_version": _OptionDef(
        ("-V", "--model-version"), Optional[str], help="Azure model version"
    ),
    "deployment_name": _OptionDef(("-d", "--deployment-name"), Optional[str]),
    "region": _OptionDef(
        ("-r", "--region"),
        Optional[str],
        help="AWS region for bedrock (e.g., `us-east-1`).",
    ),
    "project": _OptionDef(
        ("-p", "--project"),
        Optional[str],
        help="GCP project ID (used by Vertex AI / Gemini when applicable).",
    ),
    "location": _OptionDef(
        ("-l", "--location"),
        Optional[str],
        help="GCP location/region for Vertex AI (e.g., `us-central1`).",
    ),
    "service_account_file": _OptionDef(
        ("-S", "--service-account-file"),
        Optional[Path],
        help="Path to a Google service account JSON key file.",
        extra={"exists": True, "dir_okay": False, "readable": True},
    ),
    "model_format": _OptionDef(
        ("-f", "--format"),
        Optional[str],
        help="Format of the response from the local model (default: json)",
    ),
    "provider": _OptionDef(
        ("-P", "--provider"),
        Optional[str],
        help="Override the PORTKEY_PROVIDER_NAME.",
    ),
    "proxy_base_url": _OptionDef(
        ("-U", "--proxy-base-url"),
        Optional[str],
        help=(
            "Override the LITELLM_PROXY_API_BASE URL (useful for proxies, "
            "gateways, or self-hosted endpoints)."
        ),
    ),
    "temperature": _OptionDef(
        ("-t", "--temperature"),
        Optional[float],
        help=(
            "Override the global TEMPERATURE used by LLM providers "
            "(e.g., 0.0 for deterministic behavior)."
        ),
    ),
    "cost_per_input_token": _OptionDef(
        ("-i", "--cost-per-input-token"), Optional[float], help=_COST_INPUT_HELP
    ),
    "cost_per_output_token": _OptionDef(
        ("-o", "--cost-per-output-token"),
        Optional[float],
        help=_COST_OUTPUT_HELP,
    ),
    "prompt_api_key": _OptionDef(
        ("-k", "--prompt-api-key"), bool, default=False
    ),
    "proxy_prompt_api_key": _OptionDef(
        ("-K", "--proxy-prompt-api-key"), bool, default=False
    ),
    "prompt_credentials": _OptionDef(
        ("-a", "--prompt-credentials"), bool, default=False
    ),
}


@dataclass(frozen=True)
class Opt:
    """A value option, and the settings attribute it writes.

    ``help``, ``default`` and ``annotation`` fall back to the catalogue entry
    for ``param``; a provider overrides them only where its wording or typing
    genuinely differs.
    """

    param: str
    attr: str
    help: Optional[str] = None
    default: Any = _UNSET
    annotation: Any = None
    transform: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class Secret:
    """One hidden value read interactively when its flag is passed."""

    label: str
    attr: str
    hide_input: bool = True


@dataclass(frozen=True)
class SecretOpt:
    """A boolean flag that prompts for one or more secrets when passed.

    ``--prompt-credentials`` on Bedrock collects two values from a single
    flag, so this holds a tuple rather than a single secret.
    """

    param: str
    secrets: Tuple[Secret, ...]
    help: Optional[str] = None


@dataclass(frozen=True)
class SetProviderSpec:
    """Everything one ``set-*`` command needs beyond the shared logic."""

    command: str
    provider_key: Any
    model_attr: str
    error_subject: str
    success_msg: str
    options: Tuple[Any, ...]
    constants: Tuple[Tuple[str, Any], ...] = ()
    finalize: Optional[Callable[[Any, Dict[str, Any]], None]] = None
    help_text: Optional[str] = None


def _gemini_vertexai_mode(settings, assigned: Dict[str, Any]) -> None:
    """Pick between API-key and Vertex AI auth from whichever flags were given.

    An explicit API key means direct Gemini; a project, location or service
    account means Vertex AI. Runs after the option assignments, and reads the
    same settings-attribute keys the specs are written in.
    """
    if "GOOGLE_API_KEY" in assigned:
        settings.GOOGLE_GENAI_USE_VERTEXAI = False
    elif any(
        key in assigned
        for key in (
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_SERVICE_ACCOUNT_KEY",
        )
    ):
        settings.GOOGLE_GENAI_USE_VERTEXAI = True


_SET_SPECS: Tuple[SetProviderSpec, ...] = (
    SetProviderSpec(
        command="set-openai",
        provider_key=ModelKeyValues.USE_OPENAI_MODEL,
        model_attr="OPENAI_MODEL_NAME",
        error_subject="OpenAI",
        success_msg=_llm_success("OpenAI's"),
        help_text=_SET_OPENAI_HELP,
        options=(
            Opt("model", "OPENAI_MODEL_NAME", help=_MODEL_HELP_GPT),
            SecretOpt(
                "prompt_api_key",
                (Secret("OpenAI API key", "OPENAI_API_KEY"),),
            ),
            Opt("cost_per_input_token", "OPENAI_COST_PER_INPUT_TOKEN"),
            Opt("cost_per_output_token", "OPENAI_COST_PER_OUTPUT_TOKEN"),
        ),
    ),
    SetProviderSpec(
        command="set-azure-openai",
        provider_key=ModelKeyValues.USE_AZURE_OPENAI,
        model_attr="AZURE_MODEL_NAME",
        error_subject="Azure OpenAI",
        success_msg=_llm_success("Azure OpenAI's"),
        options=(
            Opt("model", "AZURE_MODEL_NAME", help=_MODEL_HELP_GPT),
            SecretOpt(
                "prompt_api_key",
                (Secret("Azure OpenAI API key", "AZURE_OPENAI_API_KEY"),),
            ),
            Opt("base_url", "AZURE_OPENAI_ENDPOINT"),
            Opt("api_version", "OPENAI_API_VERSION"),
            Opt("model_version", "AZURE_MODEL_VERSION"),
            Opt(
                "deployment_name",
                "AZURE_DEPLOYMENT_NAME",
                help="Azure OpenAI deployment name",
            ),
        ),
    ),
    SetProviderSpec(
        command="set-azure-openai-embedding",
        provider_key=EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING,
        model_attr="AZURE_EMBEDDING_MODEL_NAME",
        error_subject="Azure OpenAI embedding",
        success_msg=_embedding_success("Azure OpenAI embedding model"),
        options=(
            Opt("model", "AZURE_EMBEDDING_MODEL_NAME", help=_MODEL_HELP_GPT),
            Opt(
                "deployment_name",
                "AZURE_EMBEDDING_DEPLOYMENT_NAME",
                help="Azure embedding deployment name",
            ),
        ),
    ),
    SetProviderSpec(
        command="set-anthropic",
        provider_key=ModelKeyValues.USE_ANTHROPIC_MODEL,
        model_attr="ANTHROPIC_MODEL_NAME",
        error_subject="Anthropic",
        success_msg=_llm_success("Anthropic"),
        options=(
            Opt("model", "ANTHROPIC_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (Secret("Anthropic API key", "ANTHROPIC_API_KEY"),),
            ),
            Opt("cost_per_input_token", "ANTHROPIC_COST_PER_INPUT_TOKEN"),
            Opt("cost_per_output_token", "ANTHROPIC_COST_PER_OUTPUT_TOKEN"),
        ),
    ),
    SetProviderSpec(
        command="set-bedrock",
        provider_key=ModelKeyValues.USE_AWS_BEDROCK_MODEL,
        model_attr="AWS_BEDROCK_MODEL_NAME",
        error_subject="AWS Bedrock",
        success_msg=_llm_success("AWS Bedrock"),
        options=(
            Opt("model", "AWS_BEDROCK_MODEL_NAME"),
            SecretOpt(
                "prompt_credentials",
                (
                    Secret(
                        "AWS Access key Id",
                        "AWS_ACCESS_KEY_ID",
                        hide_input=False,
                    ),
                    Secret("AWS Secret Access key", "AWS_SECRET_ACCESS_KEY"),
                ),
                help=(
                    "Prompt for AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                    "(secret access key input is hidden). Not suitable for CI. "
                    "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, credentials "
                    "are written to dotenv in plaintext."
                ),
            ),
            Opt("region", "AWS_BEDROCK_REGION"),
            Opt("cost_per_input_token", "AWS_BEDROCK_COST_PER_INPUT_TOKEN"),
            Opt("cost_per_output_token", "AWS_BEDROCK_COST_PER_OUTPUT_TOKEN"),
        ),
    ),
    SetProviderSpec(
        command="set-ollama",
        provider_key=ModelKeyValues.USE_LOCAL_MODEL,
        model_attr="OLLAMA_MODEL_NAME",
        error_subject="Ollama",
        success_msg=_llm_success("a local Ollama model"),
        constants=(("LOCAL_MODEL_API_KEY", "ollama"),),
        options=(
            Opt("model", "OLLAMA_MODEL_NAME"),
            Opt(
                "base_url",
                "LOCAL_MODEL_BASE_URL",
                default="http://localhost:11434",
                annotation=str,
            ),
        ),
    ),
    SetProviderSpec(
        command="set-ollama-embeddings",
        provider_key=EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
        model_attr="LOCAL_EMBEDDING_MODEL_NAME",
        error_subject="Ollama embedding",
        success_msg=_embedding_success("the Ollama embedding model"),
        constants=(("LOCAL_EMBEDDING_API_KEY", "ollama"),),
        options=(
            Opt(
                "model",
                "LOCAL_EMBEDDING_MODEL_NAME",
                help="Model identifier to use for this provider.",
            ),
            Opt(
                "base_url",
                "LOCAL_EMBEDDING_BASE_URL",
                default="http://localhost:11434",
                annotation=str,
            ),
        ),
    ),
    SetProviderSpec(
        command="set-local-model",
        provider_key=ModelKeyValues.USE_LOCAL_MODEL,
        model_attr="LOCAL_MODEL_NAME",
        error_subject="Local",
        success_msg=_llm_success("a local model"),
        options=(
            Opt("model", "LOCAL_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (Secret("Local Model API key", "LOCAL_MODEL_API_KEY"),),
            ),
            Opt("base_url", "LOCAL_MODEL_BASE_URL"),
            Opt("model_format", "LOCAL_MODEL_FORMAT"),
        ),
    ),
    SetProviderSpec(
        command="set-grok",
        provider_key=ModelKeyValues.USE_GROK_MODEL,
        model_attr="GROK_MODEL_NAME",
        error_subject="Grok",
        success_msg=_llm_success("Grok"),
        options=(
            Opt("model", "GROK_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key", (Secret("Grok API key", "GROK_API_KEY"),)
            ),
            Opt("cost_per_input_token", "GROK_COST_PER_INPUT_TOKEN"),
            Opt("cost_per_output_token", "GROK_COST_PER_OUTPUT_TOKEN"),
        ),
    ),
    SetProviderSpec(
        command="set-moonshot",
        provider_key=ModelKeyValues.USE_MOONSHOT_MODEL,
        model_attr="MOONSHOT_MODEL_NAME",
        error_subject="Moonshot",
        success_msg=_llm_success("Moonshot"),
        options=(
            Opt("model", "MOONSHOT_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (Secret("Moonshot API key", "MOONSHOT_API_KEY"),),
            ),
            Opt("cost_per_input_token", "MOONSHOT_COST_PER_INPUT_TOKEN"),
            Opt("cost_per_output_token", "MOONSHOT_COST_PER_OUTPUT_TOKEN"),
        ),
    ),
    SetProviderSpec(
        command="set-deepseek",
        provider_key=ModelKeyValues.USE_DEEPSEEK_MODEL,
        model_attr="DEEPSEEK_MODEL_NAME",
        error_subject="DeepSeek",
        success_msg=_llm_success("DeepSeek"),
        options=(
            Opt("model", "DEEPSEEK_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (Secret("DeepSeek API key", "DEEPSEEK_API_KEY"),),
            ),
            Opt("cost_per_input_token", "DEEPSEEK_COST_PER_INPUT_TOKEN"),
            Opt("cost_per_output_token", "DEEPSEEK_COST_PER_OUTPUT_TOKEN"),
        ),
    ),
    SetProviderSpec(
        command="set-local-embeddings",
        provider_key=EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
        model_attr="LOCAL_EMBEDDING_MODEL_NAME",
        error_subject="Local embedding",
        success_msg=_embedding_success("the local embedding model"),
        options=(
            Opt("model", "LOCAL_EMBEDDING_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (
                    Secret(
                        "Local Embedding Model API key",
                        "LOCAL_EMBEDDING_API_KEY",
                    ),
                ),
            ),
            Opt("base_url", "LOCAL_EMBEDDING_BASE_URL"),
        ),
    ),
    SetProviderSpec(
        command="set-gemini",
        provider_key=ModelKeyValues.USE_GEMINI_MODEL,
        model_attr="GEMINI_MODEL_NAME",
        error_subject="Gemini",
        success_msg=_llm_success("Gemini"),
        finalize=_gemini_vertexai_mode,
        options=(
            Opt("model", "GEMINI_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (Secret("Google API key", "GOOGLE_API_KEY"),),
            ),
            Opt("project", "GOOGLE_CLOUD_PROJECT"),
            Opt("location", "GOOGLE_CLOUD_LOCATION"),
            Opt(
                "service_account_file",
                "GOOGLE_SERVICE_ACCOUNT_KEY",
                transform=load_service_account_key_file,
            ),
        ),
    ),
    SetProviderSpec(
        command="set-litellm",
        provider_key=ModelKeyValues.USE_LITELLM,
        model_attr="LITELLM_MODEL_NAME",
        error_subject="LiteLLM",
        success_msg=_llm_success("LiteLLM"),
        options=(
            Opt("model", "LITELLM_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (Secret("LiteLLM API key", "LITELLM_API_KEY"),),
            ),
            Opt("base_url", "LITELLM_API_BASE"),
            SecretOpt(
                "proxy_prompt_api_key",
                (Secret("LiteLLM Proxy API key", "LITELLM_PROXY_API_KEY"),),
            ),
            Opt("proxy_base_url", "LITELLM_PROXY_API_BASE"),
        ),
    ),
    SetProviderSpec(
        command="set-portkey",
        provider_key=ModelKeyValues.USE_PORTKEY_MODEL,
        model_attr="PORTKEY_MODEL_NAME",
        error_subject="Portkey",
        success_msg=_llm_success("Portkey"),
        options=(
            Opt("model", "PORTKEY_MODEL_NAME"),
            SecretOpt(
                "prompt_api_key",
                (Secret("Portkey API key", "PORTKEY_API_KEY"),),
            ),
            Opt("base_url", "PORTKEY_BASE_URL"),
            Opt("provider", "PORTKEY_PROVIDER_NAME"),
        ),
    ),
    SetProviderSpec(
        command="set-openrouter",
        provider_key=ModelKeyValues.USE_OPENROUTER_MODEL,
        model_attr="OPENROUTER_MODEL_NAME",
        error_subject="OpenRouter",
        success_msg=_llm_success("OpenRouter"),
        options=(
            Opt(
                "model",
                "OPENROUTER_MODEL_NAME",
                help=(
                    "Model identifier to use for this provider "
                    "(e.g., `openai/gpt-4.1`)."
                ),
            ),
            SecretOpt(
                "prompt_api_key",
                (Secret("OpenRouter API key", "OPENROUTER_API_KEY"),),
            ),
            Opt(
                "base_url",
                "OPENROUTER_BASE_URL",
                help=(
                    "Override the API endpoint/base URL used by this provider "
                    "(default: https://openrouter.ai/api/v1)."
                ),
            ),
            Opt("temperature", "TEMPERATURE"),
            Opt(
                "cost_per_input_token",
                "OPENROUTER_COST_PER_INPUT_TOKEN",
                help=(
                    "USD per input token used for cost tracking. If unset and "
                    "OpenRouter does not return pricing metadata, costs will "
                    "not be calculated."
                ),
            ),
            Opt(
                "cost_per_output_token",
                "OPENROUTER_COST_PER_OUTPUT_TOKEN",
                help=(
                    "USD per output token used for cost tracking. If unset "
                    "and OpenRouter does not return pricing metadata, costs "
                    "will not be calculated."
                ),
            ),
        ),
    ),
)

SET_PROVIDERS = {spec.command: spec for spec in _SET_SPECS}


def _apply_unset(
    spec: UnsetProviderSpec,
    save: Optional[str],
    clear_secrets: bool,
    quiet: bool,
) -> None:
    """Null the provider's settings, then report what happened."""
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        for attr in spec.settings_attrs:
            setattr(settings, attr, None)
        if clear_secrets:
            for attr in spec.secret_attrs:
                setattr(settings, attr, None)

    handled, path, updates = edit_ctx.result

    if handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg=spec.updated_msg,
        tip_msg=None,
    ):
        print(
            spec.openai_active_msg
            if is_openai_configured()
            else spec.fallback_msg
        )


def register_unset_command(app: typer.Typer, command: str) -> None:
    """Register the ``unset-*`` command named ``command`` on ``app``.

    Providers without secrets get no ``--clear-secrets`` flag, so the two
    signatures are declared separately — typer builds each command's options
    from the callback signature, and an unconditional flag would show up in
    ``--help`` for providers that have nothing to clear.
    """
    spec = UNSET_PROVIDERS[command]

    if spec.secret_attrs:

        def unset_provider(
            save: Optional[str] = typer.Option(
                None, "-s", "--save", help=spec.save_help
            ),
            clear_secrets: bool = typer.Option(
                False, "-x", "--clear-secrets", help=spec.clear_secrets_help
            ),
            quiet: bool = typer.Option(
                False, "-q", "--quiet", help=_QUIET_HELP
            ),
        ):
            _apply_unset(spec, save, clear_secrets, quiet)

    else:

        def unset_provider(
            save: Optional[str] = typer.Option(
                None, "-s", "--save", help=spec.save_help
            ),
            quiet: bool = typer.Option(
                False, "-q", "--quiet", help=_QUIET_HELP
            ),
        ):
            _apply_unset(spec, save, clear_secrets=False, quiet=quiet)

    unset_provider.__name__ = command.replace("-", "_")
    unset_provider.__doc__ = spec.help_text
    app.command(name=command)(unset_provider)


def _resolve_option(opt) -> tuple:
    """Fold an Opt's overrides onto its catalogue entry."""
    defn = _OPTION_CATALOGUE[opt.param]
    if isinstance(opt, SecretOpt):
        help_text = opt.help or _prompt_api_key_help(opt.secrets[0].attr)
        return defn.flags, defn.annotation, defn.default, help_text, defn.extra
    help_text = opt.help or defn.help
    default = defn.default if opt.default is _UNSET else opt.default
    annotation = opt.annotation or defn.annotation
    return defn.flags, annotation, default, help_text, defn.extra


def _build_signature(spec: SetProviderSpec) -> inspect.Signature:
    """Synthesize the callback signature typer reads the options from.

    Declaring the options this way rather than as a fixed parameter list keeps
    each command's ``--help`` to the flags that provider actually supports.
    """
    params = []
    for opt in spec.options:
        flags, annotation, default, help_text, extra = _resolve_option(opt)
        params.append(
            inspect.Parameter(
                opt.param,
                inspect.Parameter.KEYWORD_ONLY,
                default=typer.Option(
                    default, *flags, help=help_text, **(extra or {})
                ),
                annotation=annotation,
            )
        )
    params.append(
        inspect.Parameter(
            "save",
            inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(None, "-s", "--save", help=_SET_SAVE_HELP),
            annotation=Optional[str],
        )
    )
    params.append(
        inspect.Parameter(
            "quiet",
            inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(False, "-q", "--quiet", help=_QUIET_HELP),
            annotation=bool,
        )
    )
    return inspect.Signature(params)


def _collect_assignments(
    spec: SetProviderSpec, values: Dict[str, Any]
) -> Dict[str, Any]:
    """Map the options the caller actually supplied onto settings attributes.

    Blank text input is treated as absent, secrets are prompted for only when
    their flag is passed, and options that were left out are skipped entirely
    so an unrelated ``set-*`` run never clears them.
    """
    assigned: Dict[str, Any] = {}
    for opt in spec.options:
        if isinstance(opt, SecretOpt):
            if not values.get(opt.param):
                continue
            for secret in opt.secrets:
                entered = coerce_blank_to_none(
                    typer.prompt(secret.label, hide_input=secret.hide_input)
                )
                if entered is not None:
                    assigned[secret.attr] = entered
            continue
        _, annotation, _, _, _ = _resolve_option(opt)
        value = values.get(opt.param)
        if annotation in (str, Optional[str]):
            value = coerce_blank_to_none(value)
        if value is None:
            continue
        assigned[opt.attr] = opt.transform(value) if opt.transform else value
    return assigned


def register_set_command(app: typer.Typer, command: str) -> None:
    """Register the ``set-*`` command named ``command`` on ``app``."""
    spec = SET_PROVIDERS[command]

    def set_provider(**values: Any) -> None:
        save = values.get("save")
        quiet = values.get("quiet", False)
        assigned = _collect_assignments(spec, values)

        settings = get_settings()
        with settings.edit(save=save) as edit_ctx:
            edit_ctx.switch_model_provider(spec.provider_key)
            for attr, constant in spec.constants:
                setattr(settings, attr, constant)
            for attr, value in assigned.items():
                setattr(settings, attr, value)
            if spec.finalize is not None:
                spec.finalize(settings, assigned)

        handled, path, updates = edit_ctx.result

        effective_model = getattr(settings, spec.model_attr)
        if not effective_model:
            raise typer.BadParameter(
                f"{spec.error_subject} model name is not set. Pass --model "
                f"(or set {spec.model_attr}).",
                param_hint="--model",
            )
        handle_save_result(
            handled=handled,
            path=path,
            updates=updates,
            save=save,
            quiet=quiet,
            success_msg=spec.success_msg.replace(
                "{model}", escape(effective_model)
            ),
        )

    set_provider.__signature__ = _build_signature(spec)
    set_provider.__name__ = command.replace("-", "_")
    set_provider.__doc__ = spec.help_text
    app.command(name=command)(set_provider)
