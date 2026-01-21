"""
Central config for DeepEval.

- Autoloads dotenv files into os.environ without overwriting existing vars
  (order: .env -> .env.{APP_ENV} -> .env.local).
- Defines the Pydantic `Settings` model and `get_settings()` singleton.
- Exposes an `edit()` context manager that diffs changes and persists them to
  dotenv and the legacy JSON keystore (non-secret keys only), with validators and
  type coercion.
"""

import hashlib
import json
import logging
import math
import os
import re
import threading

from contextvars import ContextVar
from pathlib import Path
from pydantic import (
    AnyUrl,
    computed_field,
    confloat,
    conint,
    Field,
    field_validator,
    model_validator,
    SecretStr,
    PositiveFloat,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    NamedTuple,
    get_args,
    get_origin,
)

from deepeval.config.utils import (
    coerce_to_list,
    constrain_between,
    dedupe_preserve_order,
    parse_bool,
    read_dotenv_file,
)
from deepeval.constants import SUPPORTED_PROVIDER_SLUGS, slugify


logger = logging.getLogger(__name__)
_SAVE_RE = re.compile(r"^(?P<scheme>dotenv)(?::(?P<path>.+))?$")

_ACTIVE_SETTINGS_EDIT_CTX: ContextVar[Optional["Settings._SettingsEditCtx"]] = (
    ContextVar("_ACTIVE_SETTINGS_EDIT_CTX", default=None)
)

# settings that were converted to computed fields with override counterparts
_DEPRECATED_TO_OVERRIDE = {
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS": "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS": "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS": "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE",
}
# Track which secrets we've warned about when loading from the legacy keyfile
_LEGACY_KEYFILE_SECRET_WARNED: set[str] = set()


def _find_legacy_enum(env_key: str):
    from deepeval.key_handler import (
        ModelKeyValues,
        EmbeddingKeyValues,
        KeyValues,
    )

    enums = (ModelKeyValues, EmbeddingKeyValues, KeyValues)

    for enum in enums:
        try:
            return getattr(enum, env_key)
        except AttributeError:
            pass

    for enum in enums:
        for member in enum:
            if member.value == env_key:
                return member
    return None


def _is_secret_key(env_key: str) -> bool:
    field = Settings.model_fields.get(env_key)
    if not field:
        return False
    if field.annotation is SecretStr:
        return True

    origin = get_origin(field.annotation)
    if origin is Union:
        return any(arg is SecretStr for arg in get_args(field.annotation))
    return False


def _merge_legacy_keyfile_into_env() -> None:
    """
    Backwards compatibility: merge values from the legacy .deepeval/.deepeval
    JSON keystore into os.environ for known Settings fields, without
    overwriting existing process env vars.

    This runs before we compute the Settings env fingerprint so that Pydantic
    can see these values on first construction.

    Precedence: process env -> dotenv -> legacy json
    """
    # if somebody really wants to skip this behavior
    if parse_bool(os.getenv("DEEPEVAL_DISABLE_LEGACY_KEYFILE"), default=False):
        return

    from deepeval.constants import HIDDEN_DIR, KEY_FILE
    from deepeval.key_handler import (
        KeyValues,
        ModelKeyValues,
        EmbeddingKeyValues,
    )

    key_path = Path(HIDDEN_DIR) / KEY_FILE

    try:
        with key_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Corrupted file -> ignore, same as KeyFileHandler
                return
    except FileNotFoundError:
        # No legacy store -> nothing to merge
        return

    if not isinstance(data, dict):
        return

    # Map JSON keys (enum .value) -> env keys (enum .name)
    mapping: Dict[str, str] = {}
    for enum in (KeyValues, ModelKeyValues, EmbeddingKeyValues):
        for member in enum:
            mapping[member.value] = member.name

    for json_key, raw in data.items():
        env_key = mapping.get(json_key)
        if not env_key:
            continue

        # Process env always wins
        if env_key in os.environ:
            continue
        if raw is None:
            continue

        # Mirror the legacy warning semantics for secrets, but only once per key
        if env_key not in _LEGACY_KEYFILE_SECRET_WARNED and _is_secret_key(
            env_key
        ):
            logger.warning(
                "Reading secret '%s' (legacy key '%s') from legacy %s/%s. "
                "Persisting API keys in plaintext is deprecated. "
                "Move this to your environment (.env / .env.local). "
                "This fallback will be removed in a future release.",
                env_key,
                json_key,
                HIDDEN_DIR,
                KEY_FILE,
            )
            _LEGACY_KEYFILE_SECRET_WARNED.add(env_key)
        # Let Settings validators coerce types; we just inject the raw string
        os.environ[env_key] = str(raw)


def _discover_app_env_from_files(env_dir: Path) -> Optional[str]:
    # prefer base .env.local, then .env for APP_ENV discovery
    for name in (".env.local", ".env"):
        v = read_dotenv_file(env_dir / name).get("APP_ENV")
        if v:
            v = str(v).strip()
            if v:
                return v
    return None


def autoload_dotenv() -> None:
    """
    Load env vars from .env files without overriding existing process env.

    Precedence (lowest -> highest): .env -> .env.{APP_ENV} -> .env.local
    Process env always wins over file values.

    Controls:
      - DEEPEVAL_DISABLE_DOTENV=1 -> skip
      - ENV_DIR_PATH -> directory containing .env files (default: CWD)
    """
    if parse_bool(os.getenv("DEEPEVAL_DISABLE_DOTENV"), default=False):
        return

    raw_dir = os.getenv("ENV_DIR_PATH")
    if raw_dir:
        env_dir = Path(os.path.expanduser(os.path.expandvars(raw_dir)))
    else:
        env_dir = Path(os.getcwd())

    # merge files in precedence order
    base = read_dotenv_file(env_dir / ".env")
    local = read_dotenv_file(env_dir / ".env.local")

    # Pick APP_ENV (process -> .env.local -> .env -> default)
    app_env = (
        os.getenv("APP_ENV") or _discover_app_env_from_files(env_dir) or None
    )
    merged: Dict[str, str] = {}
    env_specific: Dict[str, str] = {}
    if app_env is not None:
        app_env = app_env.strip()
        if app_env:
            env_specific = read_dotenv_file(env_dir / f".env.{app_env}")
            merged.setdefault("APP_ENV", app_env)

    merged.update(base)
    merged.update(env_specific)
    merged.update(local)

    # Write only keys that aren’t already in process env
    for k, v in merged.items():
        if k not in os.environ:
            os.environ[k] = v


class PersistResult(NamedTuple):
    handled: bool
    path: Optional[Path]
    updated: Dict[str, Any]  # typed, validated and changed


class Settings(BaseSettings):
    # def __init__(self):
    #     super().__init__()
    def __setattr__(self, name: str, value):
        ctx = _ACTIVE_SETTINGS_EDIT_CTX.get()
        if ctx is not None and name in type(self).model_fields:
            ctx._touched.add(name)
        return super().__setattr__(name, value)

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=True,
        validate_assignment=True,
    )

    #
    # General
    #

    APP_ENV: str = Field(
        "dev",
        description="Application environment name used for dotenv selection (loads .env.<APP_ENV> if present).",
    )
    LOG_LEVEL: Optional[int] = Field(
        None,
        description="Global logging level (e.g. DEBUG/INFO/WARNING/ERROR/CRITICAL or numeric).",
    )
    PYTHONPATH: str = Field(
        ".",
        description="Extra PYTHONPATH used by the CLI runner (default: current project '.').",
    )
    CONFIDENT_REGION: Optional[str] = Field(
        None,
        description="Optional Confident AI region hint (uppercased).",
    )
    CONFIDENT_OPEN_BROWSER: Optional[bool] = Field(
        True,
        description="Open a browser automatically for Confident AI links/flows when available.",
    )

    #
    # CLI
    #
    DEEPEVAL_DEFAULT_SAVE: Optional[str] = Field(
        None,
        description="Default persistence target for settings changes (e.g. 'dotenv' or 'dotenv:/path/to/.env.local').",
    )
    DEEPEVAL_DISABLE_DOTENV: Optional[bool] = Field(
        None,
        description="Disable dotenv autoloading (.env → .env.<APP_ENV> → .env.local). Tip: set to 1 in pytest/CI to prevent loading env files on import.",
    )
    ENV_DIR_PATH: Optional[Path] = Field(
        None,
        description="Directory containing .env files (default: current working directory).",
    )
    DEEPEVAL_FILE_SYSTEM: Optional[str] = Field(
        None,
        description="Filesystem mode for runtime/CLI (currently supports READ_ONLY).",
    )
    DEEPEVAL_IDENTIFIER: Optional[str] = Field(
        None,
        description="Identifier/tag to help identify your test run on Confident AI.",
    )

    #
    # Storage & Output
    #

    # When set, DeepEval will export a timestamped JSON of the latest test run
    # into this directory. The directory will be created on demand.
    DEEPEVAL_RESULTS_FOLDER: Optional[Path] = Field(
        None,
        description="If set, export a timestamped JSON of the latest test run into this folder (created if missing).",
    )

    # Display / Truncation
    DEEPEVAL_MAXLEN_TINY: Optional[int] = Field(
        40,
        description="Default truncation length for 'tiny' displays in logs/UI.",
    )
    DEEPEVAL_MAXLEN_SHORT: Optional[int] = Field(
        60,
        description="Default truncation length for 'short' displays in logs/UI.",
    )
    DEEPEVAL_MAXLEN_MEDIUM: Optional[int] = Field(
        120,
        description="Default truncation length for 'medium' displays in logs/UI.",
    )
    DEEPEVAL_MAXLEN_LONG: Optional[int] = Field(
        240,
        description="Default truncation length for 'long' displays in logs/UI.",
    )

    # If set, this overrides the default max_len used by deepeval/utils shorten
    # falls back to DEEPEVAL_MAXLEN_LONG when None.
    DEEPEVAL_SHORTEN_DEFAULT_MAXLEN: Optional[int] = Field(
        None,
        description="Override default max_len for deepeval.utils.shorten (falls back to DEEPEVAL_MAXLEN_LONG when unset).",
    )

    # Optional global suffix (keeps your "..." default).
    DEEPEVAL_SHORTEN_SUFFIX: Optional[str] = Field(
        "...",
        description="Suffix appended by deepeval.utils.shorten when truncating (default: '...').",
    )

    #
    # GPU and perf toggles
    #

    CUDA_LAUNCH_BLOCKING: Optional[bool] = Field(
        None,
        description="CUDA debug toggle (forces synchronous CUDA ops). Useful for debugging GPU errors.",
    )
    CUDA_VISIBLE_DEVICES: Optional[str] = Field(
        None,
        description="CUDA device visibility mask (e.g. '0' or '0,1').",
    )
    TOKENIZERS_PARALLELISM: Optional[bool] = Field(
        None,
        description="HuggingFace tokenizers parallelism toggle (set to false to reduce warnings/noise).",
    )
    TRANSFORMERS_NO_ADVISORY_WARNINGS: Optional[bool] = Field(
        None,
        description="Disable advisory warnings from transformers (reduces console noise).",
    )

    #
    # Model Keys
    #

    API_KEY: Optional[SecretStr] = Field(
        None,
        description="Alias for CONFIDENT_API_KEY (Confident AI API key).",
    )
    CONFIDENT_API_KEY: Optional[SecretStr] = Field(
        None,
        description="Confident AI API key (used for uploading results/telemetry to Confident).",
    )

    # ======
    # Base URL for Confident AI API server
    # ======
    CONFIDENT_BASE_URL: Optional[str] = Field(
        None,
        description="Base URL for Confident AI API server (set only if using a custom/hosted endpoint).",
    )

    # General
    TEMPERATURE: Optional[confloat(ge=0, le=2)] = Field(
        None,
        description="Global default model temperature (0–2). Model-specific constructors may override.",
    )

    # Anthropic
    USE_ANTHROPIC_MODEL: Optional[bool] = Field(
        None,
        description="Select Anthropic as the active LLM provider (USE_* flags are mutually exclusive in CLI helpers).",
    )
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(
        None, description="Anthropic API key."
    )
    ANTHROPIC_MODEL_NAME: Optional[str] = Field(
        None, description="Anthropic model name (e.g. 'claude-3-...')."
    )
    ANTHROPIC_COST_PER_INPUT_TOKEN: Optional[PositiveFloat] = Field(
        None,
        description="Anthropic input token cost (used for cost reporting).",
    )
    ANTHROPIC_COST_PER_OUTPUT_TOKEN: Optional[PositiveFloat] = Field(
        None,
        description="Anthropic output token cost (used for cost reporting).",
    )

    # AWS
    AWS_ACCESS_KEY_ID: Optional[SecretStr] = Field(
        None,
        description="AWS access key ID (for Bedrock or other AWS-backed integrations).",
    )
    AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = Field(
        None,
        description="AWS secret access key (for Bedrock or other AWS-backed integrations).",
    )
    # AWS Bedrock
    USE_AWS_BEDROCK_MODEL: Optional[bool] = Field(
        None, description="Select AWS Bedrock as the active LLM provider."
    )
    AWS_BEDROCK_MODEL_NAME: Optional[str] = Field(
        None, description="AWS Bedrock model identifier."
    )
    AWS_BEDROCK_REGION: Optional[str] = Field(
        None, description="AWS region for Bedrock (normalized to lowercase)."
    )
    AWS_BEDROCK_COST_PER_INPUT_TOKEN: Optional[PositiveFloat] = Field(
        None, description="Bedrock input token cost (used for cost reporting)."
    )
    AWS_BEDROCK_COST_PER_OUTPUT_TOKEN: Optional[PositiveFloat] = Field(
        None, description="Bedrock output token cost (used for cost reporting)."
    )
    # Azure Open AI
    USE_AZURE_OPENAI: Optional[bool] = Field(
        None, description="Select Azure OpenAI as the active LLM provider."
    )
    AZURE_OPENAI_API_KEY: Optional[SecretStr] = Field(
        None, description="Azure OpenAI API key."
    )
    AZURE_OPENAI_AD_TOKEN: Optional[SecretStr] = Field(
        None, description="Azure OpenAI Ad Token."
    )
    AZURE_OPENAI_ENDPOINT: Optional[AnyUrl] = Field(
        None, description="Azure OpenAI endpoint URL."
    )
    OPENAI_API_VERSION: Optional[str] = Field(
        None,
        description="Azure OpenAI API version (if required by your deployment).",
    )
    AZURE_DEPLOYMENT_NAME: Optional[str] = Field(
        None,
        description="Azure OpenAI deployment name (required for most Azure configs).",
    )
    AZURE_MODEL_NAME: Optional[str] = Field(
        None,
        description="Azure model name label (informational; may be used in reporting).",
    )
    AZURE_MODEL_VERSION: Optional[str] = Field(
        None,
        description="Azure model version label (informational; may be used in reporting).",
    )
    # DeepSeek
    USE_DEEPSEEK_MODEL: Optional[bool] = Field(
        None, description="Select DeepSeek as the active LLM provider."
    )
    DEEPSEEK_API_KEY: Optional[SecretStr] = Field(
        None, description="DeepSeek API key."
    )
    DEEPSEEK_MODEL_NAME: Optional[str] = Field(
        None, description="DeepSeek model name."
    )
    DEEPSEEK_COST_PER_INPUT_TOKEN: Optional[float] = Field(
        None, description="DeepSeek input token cost (used for cost reporting)."
    )
    DEEPSEEK_COST_PER_OUTPUT_TOKEN: Optional[float] = Field(
        None,
        description="DeepSeek output token cost (used for cost reporting).",
    )
    # Gemini
    USE_GEMINI_MODEL: Optional[bool] = Field(
        None, description="Select Google Gemini as the active LLM provider."
    )
    GOOGLE_API_KEY: Optional[SecretStr] = Field(
        None, description="Google API key for Gemini (non-Vertex usage)."
    )
    GEMINI_MODEL_NAME: Optional[str] = Field(
        None, description="Gemini model name (e.g. 'gemini-...')."
    )
    GOOGLE_GENAI_USE_VERTEXAI: Optional[bool] = Field(
        None,
        description="Use Vertex AI for Gemini requests instead of direct API key mode.",
    )
    GOOGLE_CLOUD_PROJECT: Optional[str] = Field(
        None,
        description="GCP project ID for Vertex AI (required if GOOGLE_GENAI_USE_VERTEXAI=true).",
    )
    GOOGLE_CLOUD_LOCATION: Optional[str] = Field(
        None,
        description="GCP region/location for Vertex AI (e.g. 'us-central1').",
    )
    GOOGLE_SERVICE_ACCOUNT_KEY: Optional[SecretStr] = Field(
        None,
        description="Service account JSON key for Vertex AI auth (if not using ADC).",
    )
    # Grok
    USE_GROK_MODEL: Optional[bool] = Field(
        None, description="Select Grok as the active LLM provider."
    )
    GROK_API_KEY: Optional[SecretStr] = Field(None, description="Grok API key.")
    GROK_MODEL_NAME: Optional[str] = Field(None, description="Grok model name.")
    GROK_COST_PER_INPUT_TOKEN: Optional[float] = Field(
        None, description="Grok input token cost (used for cost reporting)."
    )
    GROK_COST_PER_OUTPUT_TOKEN: Optional[float] = Field(
        None, description="Grok output token cost (used for cost reporting)."
    )
    # LiteLLM
    USE_LITELLM: Optional[bool] = Field(
        None, description="Select LiteLLM as the active LLM provider."
    )
    LITELLM_API_KEY: Optional[SecretStr] = Field(
        None,
        description="LiteLLM API key (if required by your LiteLLM deployment).",
    )
    LITELLM_MODEL_NAME: Optional[str] = Field(
        None,
        description="LiteLLM model name (as exposed by your LiteLLM endpoint).",
    )
    LITELLM_API_BASE: Optional[AnyUrl] = Field(
        None, description="LiteLLM API base URL (direct)."
    )
    LITELLM_PROXY_API_BASE: Optional[AnyUrl] = Field(
        None, description="LiteLLM proxy base URL (if using proxy mode)."
    )
    LITELLM_PROXY_API_KEY: Optional[SecretStr] = Field(
        None, description="LiteLLM proxy API key (if required)."
    )
    # LM Studio
    LM_STUDIO_API_KEY: Optional[SecretStr] = Field(
        None, description="LM Studio API key (if configured)."
    )
    LM_STUDIO_MODEL_NAME: Optional[str] = Field(
        None, description="LM Studio model name."
    )
    # Local Model
    USE_LOCAL_MODEL: Optional[bool] = Field(
        None,
        description="Select a local/self-hosted model as the active LLM provider.",
    )
    LOCAL_MODEL_API_KEY: Optional[SecretStr] = Field(
        None,
        description="API key for a local/self-hosted LLM endpoint (if required).",
    )
    LOCAL_EMBEDDING_API_KEY: Optional[SecretStr] = Field(
        None,
        description="API key for a local/self-hosted embedding endpoint (if required).",
    )
    LOCAL_MODEL_NAME: Optional[str] = Field(
        None,
        description="Local/self-hosted model name (informational / routing).",
    )
    LOCAL_MODEL_BASE_URL: Optional[AnyUrl] = Field(
        None, description="Base URL for a local/self-hosted LLM endpoint."
    )
    LOCAL_MODEL_FORMAT: Optional[str] = Field(
        None,
        description="Local model API format identifier (implementation-specific).",
    )
    # Moonshot
    USE_MOONSHOT_MODEL: Optional[bool] = Field(
        None, description="Select Moonshot as the active LLM provider."
    )
    MOONSHOT_API_KEY: Optional[SecretStr] = Field(
        None, description="Moonshot API key."
    )
    MOONSHOT_MODEL_NAME: Optional[str] = Field(
        None, description="Moonshot model name."
    )
    MOONSHOT_COST_PER_INPUT_TOKEN: Optional[float] = Field(
        None, description="Moonshot input token cost (used for cost reporting)."
    )
    MOONSHOT_COST_PER_OUTPUT_TOKEN: Optional[float] = Field(
        None,
        description="Moonshot output token cost (used for cost reporting).",
    )
    # Ollama
    OLLAMA_MODEL_NAME: Optional[str] = Field(
        None,
        description="Ollama model name (used when running via Ollama integration).",
    )
    # OpenAI
    USE_OPENAI_MODEL: Optional[bool] = Field(
        None, description="Select OpenAI as the active LLM provider."
    )
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        None, description="OpenAI API key."
    )
    OPENAI_MODEL_NAME: Optional[str] = Field(
        None, description="OpenAI model name (e.g. 'gpt-4.1')."
    )
    OPENAI_COST_PER_INPUT_TOKEN: Optional[float] = Field(
        None, description="OpenAI input token cost (used for cost reporting)."
    )
    OPENAI_COST_PER_OUTPUT_TOKEN: Optional[float] = Field(
        None, description="OpenAI output token cost (used for cost reporting)."
    )
    # PortKey
    USE_PORTKEY_MODEL: Optional[bool] = Field(
        None, description="Select Portkey as the active LLM provider."
    )
    PORTKEY_API_KEY: Optional[SecretStr] = Field(
        None, description="Portkey API key."
    )
    PORTKEY_MODEL_NAME: Optional[str] = Field(
        None, description="Portkey model name (as configured in Portkey)."
    )
    PORTKEY_BASE_URL: Optional[AnyUrl] = Field(
        None, description="Portkey base URL (if using a custom endpoint)."
    )
    PORTKEY_PROVIDER_NAME: Optional[str] = Field(
        None, description="Provider name/routing hint for Portkey."
    )
    # OpenRouter
    USE_OPENROUTER_MODEL: Optional[bool] = None
    OPENROUTER_API_KEY: Optional[SecretStr] = None
    OPENROUTER_MODEL_NAME: Optional[str] = None
    OPENROUTER_COST_PER_INPUT_TOKEN: Optional[float] = None
    OPENROUTER_COST_PER_OUTPUT_TOKEN: Optional[float] = None
    OPENROUTER_BASE_URL: Optional[AnyUrl] = Field(
        None, description="OpenRouter base URL (if using a custom endpoint)."
    )

    # Vertex AI
    VERTEX_AI_MODEL_NAME: Optional[str] = Field(
        None,
        description="Vertex AI model name (used by some Google integrations).",
    )
    # VLLM
    VLLM_API_KEY: Optional[SecretStr] = Field(
        None, description="vLLM API key (if required by your vLLM gateway)."
    )
    VLLM_MODEL_NAME: Optional[str] = Field(None, description="vLLM model name.")

    #
    # Embedding Keys
    #

    # Azure OpenAI
    USE_AZURE_OPENAI_EMBEDDING: Optional[bool] = Field(
        None, description="Use Azure OpenAI for embeddings."
    )
    AZURE_EMBEDDING_MODEL_NAME: Optional[str] = Field(
        None, description="Azure embedding model name label."
    )
    AZURE_EMBEDDING_DEPLOYMENT_NAME: Optional[str] = Field(
        None, description="Azure embedding deployment name."
    )

    # Local
    USE_LOCAL_EMBEDDINGS: Optional[bool] = Field(
        None, description="Use a local/self-hosted embeddings endpoint."
    )
    LOCAL_EMBEDDING_MODEL_NAME: Optional[str] = Field(
        None,
        description="Local embedding model name (informational / routing).",
    )
    LOCAL_EMBEDDING_BASE_URL: Optional[AnyUrl] = Field(
        None,
        description="Base URL for a local/self-hosted embeddings endpoint.",
    )

    #
    # Retry Policy
    #
    # Controls how Tenacity retries provider calls when the SDK isn't doing its own retries.
    # Key concepts:
    # - attempts count includes the first call. e.g. 1 = no retries, 2 = one retry.
    # - backoff sleeps follow exponential growth with a cap, plus jitter. Expected jitter
    #   contribution is ~ JITTER/2 per sleep.
    # - logging levels are looked up dynamically each attempt, so if you change LOG_LEVEL at runtime,
    #   the retry loggers will honor it without restart.
    DEEPEVAL_SDK_RETRY_PROVIDERS: Optional[List[str]] = Field(
        None,
        description="Providers for which retries should be delegated to the provider SDK (use ['*'] for all).",
    )
    DEEPEVAL_RETRY_BEFORE_LOG_LEVEL: Optional[int] = Field(
        None,
        description="Log level for 'before retry' logs (defaults to LOG_LEVEL if set, else INFO).",
    )
    DEEPEVAL_RETRY_AFTER_LOG_LEVEL: Optional[int] = Field(
        None,
        description="Log level for 'after retry' logs (defaults to ERROR).",
    )
    DEEPEVAL_RETRY_MAX_ATTEMPTS: conint(ge=1) = Field(
        2,
        description="Max attempts per provider call (includes the first call; 1 = no retries).",
    )
    DEEPEVAL_RETRY_INITIAL_SECONDS: confloat(ge=0) = Field(
        1.0,
        description="Initial backoff sleep (seconds) before the first retry.",
    )
    DEEPEVAL_RETRY_EXP_BASE: confloat(ge=1) = Field(
        2.0, description="Exponential backoff growth factor."
    )
    DEEPEVAL_RETRY_JITTER: confloat(ge=0) = Field(
        2.0, description="Uniform jitter added to each retry sleep (seconds)."
    )
    DEEPEVAL_RETRY_CAP_SECONDS: confloat(ge=0) = Field(
        5.0, description="Maximum backoff sleep per retry (seconds)."
    )

    #
    # Telemetry and Debug
    #
    DEEPEVAL_DEBUG_ASYNC: Optional[bool] = Field(
        None, description="Enable extra async debugging logs/behavior."
    )
    DEEPEVAL_TELEMETRY_OPT_OUT: Optional[bool] = Field(
        None,
        description="Opt out of DeepEval telemetry (OFF wins if conflicting legacy flags are set).",
    )
    DEEPEVAL_UPDATE_WARNING_OPT_IN: Optional[bool] = Field(
        None,
        description="Opt in to update warnings in the CLI/runtime when new versions are available.",
    )
    DEEPEVAL_GRPC_LOGGING: Optional[bool] = Field(
        None,
        description="Enable extra gRPC logging for Confident transport/debugging.",
    )
    GRPC_VERBOSITY: Optional[str] = Field(
        None, description="gRPC verbosity (grpc env var passthrough)."
    )
    GRPC_TRACE: Optional[str] = Field(
        None, description="gRPC trace categories (grpc env var passthrough)."
    )
    ERROR_REPORTING: Optional[bool] = Field(
        None,
        description="Enable/disable error reporting (implementation/integration dependent).",
    )
    IGNORE_DEEPEVAL_ERRORS: Optional[bool] = Field(
        None,
        description="Continue execution when DeepEval encounters certain recoverable errors.",
    )
    SKIP_DEEPEVAL_MISSING_PARAMS: Optional[bool] = Field(
        None,
        description="Skip metrics/test cases with missing required params instead of raising.",
    )
    DEEPEVAL_VERBOSE_MODE: Optional[bool] = Field(
        None, description="Enable verbose logging and additional warnings."
    )
    DEEPEVAL_LOG_STACK_TRACES: Optional[bool] = Field(
        None, description="Include stack traces in certain DeepEval error logs."
    )
    ENABLE_DEEPEVAL_CACHE: Optional[bool] = Field(
        None,
        description="Enable DeepEval caching where supported (may improve performance).",
    )

    CONFIDENT_TRACE_FLUSH: Optional[bool] = Field(
        None,
        description="Flush traces eagerly (useful for debugging; may add overhead).",
    )
    CONFIDENT_TRACE_ENVIRONMENT: Optional[str] = Field(
        "development",
        description="Trace environment label (e.g. development/staging/production).",
    )
    CONFIDENT_TRACE_VERBOSE: Optional[bool] = Field(
        True, description="Enable verbose trace logging for Confident tracing."
    )
    CONFIDENT_TRACE_SAMPLE_RATE: Optional[float] = Field(
        1.0, description="Trace sampling rate (0–1). Lower to reduce overhead."
    )

    CONFIDENT_METRIC_LOGGING_FLUSH: Optional[bool] = Field(
        None,
        description="Flush metric logs eagerly (useful for debugging; may add overhead).",
    )
    CONFIDENT_METRIC_LOGGING_VERBOSE: Optional[bool] = Field(
        True, description="Enable verbose metric logging."
    )
    CONFIDENT_METRIC_LOGGING_SAMPLE_RATE: Optional[float] = Field(
        1.0,
        description="Metric logging sampling rate (0–1). Lower to reduce overhead.",
    )
    CONFIDENT_METRIC_LOGGING_ENABLED: Optional[bool] = Field(
        True, description="Enable metric logging to Confident where supported."
    )

    CONFIDENT_OTEL_URL: Optional[AnyUrl] = Field(
        "https://otel.confident-ai.com",
        description="OpenTelemetry OTLP exporter endpoint (if using OTEL export).",
    )

    #
    # Network
    #
    MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS: float = Field(
        3.05,
        description="Connect timeout (seconds) when fetching remote images for multimodal inputs.",
    )
    MEDIA_IMAGE_READ_TIMEOUT_SECONDS: float = Field(
        10.0,
        description="Read timeout (seconds) when fetching remote images for multimodal inputs.",
    )
    DEEPEVAL_DISABLE_TIMEOUTS: Optional[bool] = Field(
        None,
        description="Disable DeepEval-enforced timeouts (per-attempt, per-task, gather). Provider SDK timeouts may still apply.",
    )
    # DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE
    # Per-attempt timeout (seconds) for provider calls used by the retry policy.
    # This is an OVERRIDE setting. The effective value you should rely on at runtime is
    # the computed property: DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS.
    #
    # If this is None or 0 the DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS is computed from either:
    #   - DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE: slice the outer budget
    #     across attempts after subtracting expected backoff and a small safety buffer
    #   - the default outer budget (180s) if no outer override is set.
    #
    # Tip: Set this OR the outer override, but generally not both
    DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE: Optional[confloat(gt=0)] = (
        Field(
            None,
            description="Override per-attempt provider call timeout (seconds). Leave unset to derive from task timeout.",
        )
    )

    #
    # Async Document Pipelines
    #

    DEEPEVAL_MAX_CONCURRENT_DOC_PROCESSING: conint(ge=1) = Field(
        2, description="Max concurrent async document processing tasks."
    )

    #
    # Async Task Configuration
    #
    DEEPEVAL_TIMEOUT_THREAD_LIMIT: conint(ge=1) = Field(
        128,
        description="Max worker threads used for timeout enforcement in async execution.",
    )
    DEEPEVAL_TIMEOUT_SEMAPHORE_WARN_AFTER_SECONDS: confloat(ge=0) = Field(
        5.0,
        description="Warn if waiting on the timeout semaphore longer than this many seconds.",
    )
    # DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE
    # Outer time budget (seconds) for a single metric/test-case, including retries and backoff.
    # This is an OVERRIDE setting. If None or 0 the DEEPEVAL_PER_TASK_TIMEOUT_SECONDS field is computed:
    #     attempts * per_attempt_timeout + expected_backoff + 1s safety
    # (When neither override is set 180s is used.)
    #
    # If > 0, we use the value exactly and log a warning if it is likely too small
    # to accommodate the configured attempts/backoff.
    #
    # usage:
    #   - set DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE along with DEEPEVAL_RETRY_MAX_ATTEMPTS, or
    #   - set DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE alone.
    DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE: Optional[confloat(ge=0)] = (
        Field(
            None,
            description="Override outer per-test-case timeout budget (seconds), including retries/backoff. Leave unset to auto-derive.",
        )
    )

    # Buffer time for gathering results from all tasks, added to the longest task duration
    # Increase if many tasks are running concurrently
    # DEEPEVAL_TASK_GATHER_BUFFER_SECONDS: confloat(ge=0) = (
    #     30  # 15s seemed like not enough. we may make this computed later.
    # )
    DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE: Optional[confloat(ge=0)] = (
        Field(
            None,
            description="Override buffer added to the longest task duration when gathering async results (seconds).",
        )
    )

    ###################
    # Computed Fields #
    ###################

    def _calc_auto_outer_timeout(self) -> float:
        """Compute outer budget from per-attempt timeout + retries/backoff.
        Never reference the computed property itself here.
        """
        attempts = self.DEEPEVAL_RETRY_MAX_ATTEMPTS or 1
        timeout_seconds = float(
            self.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE or 0
        )
        if timeout_seconds <= 0:
            # No per-attempt timeout set -> default outer budget
            return 180

        backoff = self._expected_backoff(attempts)
        safety_overhead = 1.0
        return float(
            math.ceil(attempts * timeout_seconds + backoff + safety_overhead)
        )

    @computed_field
    @property
    def DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS(self) -> float:
        over = self.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE
        if over is not None and float(over) > 0:
            return float(over)

        attempts = int(self.DEEPEVAL_RETRY_MAX_ATTEMPTS or 1)
        outer_over = self.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE

        # If the user set an outer override, slice it up
        if outer_over and float(outer_over) > 0 and attempts > 0:
            backoff = self._expected_backoff(attempts)
            safety = 1.0
            usable = max(0.0, float(outer_over) - backoff - safety)
            return 0.0 if usable <= 0 else (usable / attempts)

        # NEW: when neither override is set, derive from the default outer (180s)
        default_outer = 180.0
        backoff = self._expected_backoff(attempts)
        safety = 1.0
        usable = max(0.0, default_outer - backoff - safety)
        # Keep per-attempt sensible (cap to at least 1s)
        return 0.0 if usable <= 0 else max(1.0, usable / attempts)

    @computed_field
    @property
    def DEEPEVAL_PER_TASK_TIMEOUT_SECONDS(self) -> float:
        """If OVERRIDE is set (nonzero), return it; else return the derived budget."""
        outer = self.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE
        if outer not in (None, 0):
            # Warn if user-provided outer is likely to truncate retries
            if (self.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0) > 0:
                min_needed = self._calc_auto_outer_timeout()
                if float(outer) < min_needed:
                    if self.DEEPEVAL_VERBOSE_MODE:
                        logger.warning(
                            "Metric timeout (outer=%ss) is less than attempts × per-attempt "
                            "timeout + backoff (≈%ss). Retries may be cut short.",
                            float(outer),
                            min_needed,
                        )
            return float(outer)

        # Auto mode
        return self._calc_auto_outer_timeout()

    @computed_field
    @property
    def DEEPEVAL_TASK_GATHER_BUFFER_SECONDS(self) -> float:
        """
        Buffer time we add to the longest task’s duration to allow gather/drain
        to complete. If an override is provided, use it; otherwise derive a
        sensible default from the task-level budget:
            buffer = constrain_between(0.15 * DEEPEVAL_PER_TASK_TIMEOUT_SECONDS, 10, 60)
        """
        over = self.DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE
        if over is not None and float(over) >= 0:
            return float(over)

        outer = float(self.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS or 0.0)
        base = 0.15 * outer
        return constrain_between(base, 10.0, 60.0)

    ##############
    # Validators #
    ##############

    @field_validator(
        "CONFIDENT_METRIC_LOGGING_ENABLED",
        "CONFIDENT_METRIC_LOGGING_VERBOSE",
        "CONFIDENT_METRIC_LOGGING_FLUSH",
        "CONFIDENT_OPEN_BROWSER",
        "CONFIDENT_TRACE_FLUSH",
        "CONFIDENT_TRACE_VERBOSE",
        "CUDA_LAUNCH_BLOCKING",
        "DEEPEVAL_DEBUG_ASYNC",
        "DEEPEVAL_LOG_STACK_TRACES",
        "DEEPEVAL_DISABLE_TIMEOUTS",
        "DEEPEVAL_VERBOSE_MODE",
        "DEEPEVAL_GRPC_LOGGING",
        "DEEPEVAL_DISABLE_DOTENV",
        "DEEPEVAL_TELEMETRY_OPT_OUT",
        "DEEPEVAL_UPDATE_WARNING_OPT_IN",
        "ENABLE_DEEPEVAL_CACHE",
        "ERROR_REPORTING",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "IGNORE_DEEPEVAL_ERRORS",
        "SKIP_DEEPEVAL_MISSING_PARAMS",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
        "USE_AWS_BEDROCK_MODEL",
        "USE_OPENAI_MODEL",
        "USE_AZURE_OPENAI",
        "USE_LOCAL_MODEL",
        "USE_GEMINI_MODEL",
        "USE_MOONSHOT_MODEL",
        "USE_GROK_MODEL",
        "USE_DEEPSEEK_MODEL",
        "USE_LITELLM",
        "USE_AZURE_OPENAI_EMBEDDING",
        "USE_LOCAL_EMBEDDINGS",
        "USE_PORTKEY_MODEL",
        mode="before",
    )
    @classmethod
    def _coerce_yes_no(cls, v):
        return None if v is None else parse_bool(v, default=False)

    @field_validator("DEEPEVAL_RESULTS_FOLDER", "ENV_DIR_PATH", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        # expand ~ and env vars;
        # but don't resolve to avoid failing on non-existent paths
        return Path(os.path.expandvars(os.path.expanduser(s)))

    # Treat "", "none", "null" as None for numeric overrides
    @field_validator(
        "OPENAI_COST_PER_INPUT_TOKEN",
        "OPENAI_COST_PER_OUTPUT_TOKEN",
        "AWS_BEDROCK_COST_PER_INPUT_TOKEN",
        "AWS_BEDROCK_COST_PER_OUTPUT_TOKEN",
        "TEMPERATURE",
        "CONFIDENT_TRACE_SAMPLE_RATE",
        "CONFIDENT_METRIC_LOGGING_SAMPLE_RATE",
        mode="before",
    )
    @classmethod
    def _none_or_float(cls, v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"", "none", "null"}:
            return None
        return float(v)

    @field_validator(
        "CONFIDENT_TRACE_SAMPLE_RATE", "CONFIDENT_METRIC_LOGGING_SAMPLE_RATE"
    )
    @classmethod
    def _validate_sample_rate(cls, v):
        if v is None:
            return None
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(
                "CONFIDENT_TRACE_SAMPLE_RATE or CONFIDENT_METRIC_LOGGING_SAMPLE_RATE must be between 0 and 1"
            )
        return float(v)

    @field_validator("DEEPEVAL_DEFAULT_SAVE", mode="before")
    @classmethod
    def _validate_default_save(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        m = _SAVE_RE.match(s)
        if not m:
            raise ValueError(
                "DEEPEVAL_DEFAULT_SAVE must be 'dotenv' or 'dotenv:<path>'"
            )
        path = m.group("path")
        if path is None:
            return "dotenv"
        path = os.path.expanduser(os.path.expandvars(path))
        return f"dotenv:{path}"

    @field_validator("DEEPEVAL_FILE_SYSTEM", mode="before")
    @classmethod
    def _normalize_fs(cls, v):
        if v is None:
            return None
        s = str(v).strip().upper()

        # adds friendly aliases
        if s in {"READ_ONLY", "READ-ONLY", "READONLY", "RO"}:
            return "READ_ONLY"
        raise ValueError(
            "DEEPEVAL_FILE_SYSTEM must be READ_ONLY (case-insensitive)."
        )

    @field_validator("CONFIDENT_REGION", mode="before")
    @classmethod
    def _normalize_upper(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return s.upper()

    @field_validator("AWS_BEDROCK_REGION", mode="before")
    @classmethod
    def _normalize_lower(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return s.lower()

    @field_validator("DEEPEVAL_SDK_RETRY_PROVIDERS", mode="before")
    @classmethod
    def _coerce_to_list(cls, v):
        # works with JSON list, comma/space/semicolon separated, or real lists
        return coerce_to_list(v, lower=True)

    @field_validator("DEEPEVAL_SDK_RETRY_PROVIDERS", mode="after")
    @classmethod
    def _validate_sdk_provider_list(cls, v):
        if v is None:
            return None

        normalized: list[str] = []
        star = False

        for item in v:
            s = str(item).strip()
            if not s:
                continue
            if s == "*":
                star = True
                continue
            s = slugify(s)
            if s in SUPPORTED_PROVIDER_SLUGS:
                normalized.append(s)
            else:
                if parse_bool(
                    os.getenv("DEEPEVAL_VERBOSE_MODE"), default=False
                ):
                    logger.warning("Unknown provider slug %r dropped", item)

        if star:
            return ["*"]

        # It is important to dedup after normalization to catch variants
        normalized = dedupe_preserve_order(normalized)
        return normalized or None

    @field_validator(
        "DEEPEVAL_RETRY_BEFORE_LOG_LEVEL",
        "DEEPEVAL_RETRY_AFTER_LOG_LEVEL",
        "LOG_LEVEL",
        mode="before",
    )
    @classmethod
    def _coerce_log_level(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)

        s = str(v).strip().upper()
        if not s:
            return None

        import logging

        # Accept standard names or numeric strings
        name_to_level = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        if s in name_to_level:
            return name_to_level[s]
        raise ValueError(
            "Retry log level must be one of DEBUG, INFO, WARNING, ERROR, "
            "CRITICAL, NOTSET, or a numeric logging level."
        )

    @field_validator("DEEPEVAL_TELEMETRY_OPT_OUT", mode="before")
    @classmethod
    def _apply_telemetry_enabled_alias(cls, v):
        """
        Precedence (most secure):
        - Any OFF signal wins if both are set:
          - DEEPEVAL_TELEMETRY_OPT_OUT = truthy  -> OFF
          - DEEPEVAL_TELEMETRY_ENABLED = falsy   -> OFF
        - Else, ON signal:
          - DEEPEVAL_TELEMETRY_OPT_OUT = falsy   -> ON
          - DEEPEVAL_TELEMETRY_ENABLED = truthy  -> ON
        - Else None (unset) -> ON
        """

        def normalize(x):
            if x is None:
                return None
            s = str(x).strip()
            return None if s == "" else parse_bool(s, default=False)

        new_opt_out = normalize(v)  # True means OFF, False means ON
        legacy_enabled = normalize(
            os.getenv("DEEPEVAL_TELEMETRY_ENABLED")
        )  # True means ON, False means OFF

        off_signal = (new_opt_out is True) or (legacy_enabled is False)
        on_signal = (new_opt_out is False) or (legacy_enabled is True)

        # Conflict: simultaneous OFF and ON signals
        if off_signal and on_signal:
            # Only warn if verbose or debug
            if parse_bool(
                os.getenv("DEEPEVAL_VERBOSE_MODE"), default=False
            ) or logger.isEnabledFor(logging.DEBUG):
                logger.warning(
                    "Conflicting telemetry flags detected: DEEPEVAL_TELEMETRY_OPT_OUT=%r, "
                    "DEEPEVAL_TELEMETRY_ENABLED=%r. Defaulting to OFF.",
                    new_opt_out,
                    legacy_enabled,
                )
            return True  # OFF wins

        # Clear winner
        if off_signal:
            return True  # OFF
        if on_signal:
            return False  # ON

        # Unset means ON
        return False

    @model_validator(mode="after")
    def _apply_deprecated_computed_env_aliases(self):
        """
        Backwards compatibility courtesy:
        - If users still set a deprecated computed field in the environment,
          emit a deprecation warning and mirror its value into the matching
          *_OVERRIDE field (unless the override is already set).
        - Override always wins if both are present.
        """
        for old_key, override_key in _DEPRECATED_TO_OVERRIDE.items():
            raw = os.getenv(old_key)
            if raw is None or str(raw).strip() == "":
                continue

            # if override already set, ignore the deprecated one but log a warning
            if getattr(self, override_key) is not None:
                logger.warning(
                    "Config deprecation: %s is deprecated and was ignored because %s "
                    "is already set. Please remove %s and use %s going forward.",
                    old_key,
                    override_key,
                    old_key,
                    override_key,
                )
                continue

            # apply the deprecated value into the override field.
            try:
                # let pydantic coerce the string to the target type on assignment
                setattr(self, override_key, raw)
                logger.warning(
                    "Config deprecation: %s is deprecated. Its value (%r) was applied to %s. "
                    "Please migrate to %s and remove %s from your environment.",
                    old_key,
                    raw,
                    override_key,
                    override_key,
                    old_key,
                )
            except Exception as e:
                # do not let exception bubble up, just warn
                logger.warning(
                    "Config deprecation: %s is deprecated and could not be applied to %s "
                    "(value=%r): %s",
                    old_key,
                    override_key,
                    raw,
                    e,
                )
        return self

    #######################
    # Persistence support #
    #######################
    class _SettingsEditCtx:
        # TODO: will generate this list in future PR
        COMPUTED_FIELDS: frozenset[str] = frozenset(
            {
                "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS",
                "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS",
                "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS",
            }
        )

        def __init__(
            self,
            settings: "Settings",
            save: Optional[str],
            persist: Optional[bool],
        ):
            self._s = settings
            self._save = save
            self._persist = persist
            self._before: Dict[str, Any] = {}
            self._touched: set[str] = set()
            self.result: Optional[PersistResult] = None

        @property
        def s(self) -> "Settings":
            return self._s

        def __enter__(self) -> "Settings._SettingsEditCtx":
            # snapshot current state
            self._token = _ACTIVE_SETTINGS_EDIT_CTX.set(self)
            self._before = {
                k: getattr(self._s, k) for k in type(self._s).model_fields
            }
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc_type is not None:
                    return False  # don’t persist on error

                from deepeval.config.settings_manager import (
                    update_settings_and_persist,
                    _normalize_for_env,
                    _resolve_save_path,
                )

                # lazy import legacy JSON store deps
                from deepeval.key_handler import KEY_FILE_HANDLER

                model_fields = type(self._s).model_fields
                # Exclude computed fields from persistence

                # compute diff of changed fields
                after = {k: getattr(self._s, k) for k in model_fields}

                before_norm = {
                    k: _normalize_for_env(v) for k, v in self._before.items()
                }
                after_norm = {
                    k: _normalize_for_env(v) for k, v in after.items()
                }

                changed_keys = {
                    k for k in after_norm if after_norm[k] != before_norm.get(k)
                }
                changed_keys -= self.COMPUTED_FIELDS
                touched_keys = set(self._touched) - self.COMPUTED_FIELDS

                # dotenv should persist union(changed, touched)
                persist_dotenv = self._persist is not False
                ok, resolved_path = _resolve_save_path(self._save)

                existing_dotenv = {}
                if persist_dotenv and ok and resolved_path is not None:
                    existing_dotenv = read_dotenv_file(resolved_path)

                candidate_keys_for_dotenv = (
                    changed_keys | touched_keys
                ) - self.COMPUTED_FIELDS

                keys_for_dotenv: set[str] = set()
                for key in candidate_keys_for_dotenv:
                    desired = after_norm.get(key)  # normalized string or None
                    if desired is None:
                        # only need to unset if it's actually present in dotenv
                        # if key in existing_dotenv:
                        #     keys_for_dotenv.add(key)
                        keys_for_dotenv.add(key)
                    else:
                        if existing_dotenv.get(key) != desired:
                            keys_for_dotenv.add(key)

                updates_for_dotenv = {
                    key: after[key] for key in keys_for_dotenv
                }

                if not changed_keys and not updates_for_dotenv:
                    if self._persist is False:
                        # we report handled so that the cli does not mistakenly report invalid save option
                        self.result = PersistResult(True, None, {})
                        return False

                    ok, resolved_path = _resolve_save_path(self._save)
                    self.result = PersistResult(ok, resolved_path, {})
                    return False

                updates = {k: after[k] for k in changed_keys}

                if "LOG_LEVEL" in updates:
                    from deepeval.config.logging import (
                        apply_deepeval_log_level,
                    )

                    apply_deepeval_log_level()

                #
                # .deepeval JSON support
                #

                if self._persist is not False:
                    for k in changed_keys:
                        legacy_member = _find_legacy_enum(k)
                        if legacy_member is None:
                            continue  # skip if not a defined as legacy field

                        val = updates[k]
                        # Remove from JSON if unset
                        if val is None:
                            KEY_FILE_HANDLER.remove_key(legacy_member)
                            continue

                        # Never store secrets in the JSON keystore
                        if _is_secret_key(k):
                            continue

                        # For booleans, the legacy store expects "YES"/"NO"
                        if isinstance(val, bool):
                            KEY_FILE_HANDLER.write_key(
                                legacy_member, "YES" if val else "NO"
                            )
                        else:
                            # store as string
                            KEY_FILE_HANDLER.write_key(legacy_member, str(val))

                #
                # dotenv store
                #

                # defer import to avoid cyclics
                handled, path = update_settings_and_persist(
                    updates_for_dotenv,
                    save=self._save,
                    persist_dotenv=persist_dotenv,
                )
                self.result = PersistResult(handled, path, updates_for_dotenv)
                return False
            finally:
                if self._token is not None:
                    _ACTIVE_SETTINGS_EDIT_CTX.reset(self._token)

        def switch_model_provider(self, target) -> None:
            """
            Flip USE_* settings within the target's provider family (LLM vs embeddings).
            """
            from deepeval.key_handler import KEY_FILE_HANDLER

            target_key = getattr(target, "value", str(target))

            def _is_embedding_flag(k: str) -> bool:
                return "EMBEDDING" in k

            target_is_embedding = _is_embedding_flag(target_key)

            use_fields = [
                field
                for field in type(self._s).model_fields
                if field.startswith("USE_")
                and _is_embedding_flag(field) == target_is_embedding
            ]

            if target_key not in use_fields:
                raise ValueError(
                    f"{target_key} is not a recognized USE_* field"
                )

            for field in use_fields:
                on = field == target_key
                setattr(self._s, field, on)

                if self._persist is not False:
                    legacy_member = _find_legacy_enum(field)
                    if legacy_member is not None:
                        KEY_FILE_HANDLER.write_key(
                            legacy_member, "YES" if on else "NO"
                        )

    def edit(
        self, *, save: Optional[str] = None, persist: Optional[bool] = None
    ):
        """Context manager for atomic, persisted updates.

        Args:
            save: 'dotenv[:path]' to explicitly write to a dotenv file.
                  None (default) respects DEEPEVAL_DEFAULT_SAVE if set.
            persist: If False, do not write (dotenv, JSON), update runtime only.
                     If True or None, normal persistence rules apply.
        """
        return self._SettingsEditCtx(self, save, persist)

    def set_model_provider(self, target, *, save: Optional[str] = None):
        """
        Convenience wrapper to switch providers outside of an existing edit() block.
        Returns the PersistResult.
        """
        with self.edit(save=save) as ctx:
            ctx.switch_model_provider(target)
        return ctx.result

    def _expected_backoff(self, attempts: int) -> float:
        """Sum of expected sleeps for (attempts-1) retries, including jitter expectation."""
        sleeps = max(0, attempts - 1)
        cur = float(self.DEEPEVAL_RETRY_INITIAL_SECONDS)
        cap = float(self.DEEPEVAL_RETRY_CAP_SECONDS)
        base = float(self.DEEPEVAL_RETRY_EXP_BASE)
        jitter = float(self.DEEPEVAL_RETRY_JITTER)

        backoff = 0.0
        for _ in range(sleeps):
            backoff += min(cap, cur)
            cur *= base
        backoff += sleeps * (jitter / 2.0)  # expected jitter
        return backoff

    def _constrain_between(self, value: float, lo: float, hi: float) -> float:
        """Return value constrained to the inclusive range [lo, hi]."""
        return min(max(value, lo), hi)


_settings_singleton: Optional[Settings] = None
_settings_env_fingerprint: Optional[str] = None
_settings_lock = threading.RLock()


def _calc_env_fingerprint() -> str:
    # Pull legacy .deepeval JSON-based settings into the process env before hashing
    _merge_legacy_keyfile_into_env()

    env = os.environ.copy()
    # must hash in a stable order.
    keys = sorted(
        key
        for key in Settings.model_fields.keys()
        if key != "_DEPRECATED_TELEMETRY_ENABLED"  # exclude deprecated
    )
    # encode as triples: (key, present?, value)
    items = [(k, k in env, env.get(k)) for k in keys]
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_settings() -> Settings:
    global _settings_singleton, _settings_env_fingerprint
    fingerprint = _calc_env_fingerprint()

    with _settings_lock:
        if (
            _settings_singleton is None
            or _settings_env_fingerprint != fingerprint
        ):
            _settings_singleton = Settings()
            _settings_env_fingerprint = fingerprint
            from deepeval.config.logging import apply_deepeval_log_level

            apply_deepeval_log_level()
        return _settings_singleton


def reset_settings(*, reload_dotenv: bool = False) -> Settings:
    """
    Drop the cached Settings singleton and rebuild it from the current process
    environment.

    Args:
        reload_dotenv: When True, call `autoload_dotenv()` before re-instantiating,
                       which merges .env values into os.environ (never overwriting
                       existing process env vars).

    Returns:
        The fresh Settings instance.
    """
    global _settings_singleton, _settings_env_fingerprint
    with _settings_lock:
        if reload_dotenv:
            autoload_dotenv()
        _settings_singleton = None
        _settings_env_fingerprint = None
    return get_settings()
