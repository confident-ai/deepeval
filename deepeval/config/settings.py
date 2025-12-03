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

from dotenv import dotenv_values
from pathlib import Path
from pydantic import (
    AnyUrl,
    computed_field,
    confloat,
    conint,
    field_validator,
    model_validator,
    SecretStr,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, List, Optional, NamedTuple

from deepeval.config.utils import (
    parse_bool,
    coerce_to_list,
    constrain_between,
    dedupe_preserve_order,
)
from deepeval.constants import SUPPORTED_PROVIDER_SLUGS, slugify


logger = logging.getLogger(__name__)
_SAVE_RE = re.compile(r"^(?P<scheme>dotenv)(?::(?P<path>.+))?$")

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


def _is_secret_key(settings: "Settings", env_key: str) -> bool:
    field = type(settings).model_fields.get(env_key)
    if not field:
        return False
    if field.annotation is SecretStr:
        return True
    # Optional[SecretStr] etc.
    from typing import get_origin, get_args, Union

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
        SECRET_KEYS,
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
        if (
            json_key in SECRET_KEYS
            and json_key not in _LEGACY_KEYFILE_SECRET_WARNED
        ):
            logger.warning(
                "Reading secret '%s' from legacy %s/%s. "
                "Persisting API keys in plaintext is deprecated. "
                "Move this to your environment (.env / .env.local). "
                "This fallback will be removed in a future release.",
                json_key,
                HIDDEN_DIR,
                KEY_FILE,
            )
            _LEGACY_KEYFILE_SECRET_WARNED.add(json_key)

        # Let Settings validators coerce types; we just inject the raw string
        os.environ[env_key] = str(raw)


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        # filter out None to avoid writing "None" later
        return {
            k: v for k, v in dotenv_values(str(path)).items() if v is not None
        }
    except Exception:
        return {}


def _discover_app_env_from_files(env_dir: Path) -> Optional[str]:
    # prefer base .env.local, then .env for APP_ENV discovery
    for name in (".env.local", ".env"):
        v = _read_env_file(env_dir / name).get("APP_ENV")
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
    base = _read_env_file(env_dir / ".env")
    local = _read_env_file(env_dir / ".env.local")

    # Pick APP_ENV (process -> .env.local -> .env -> default)
    app_env = (
        os.getenv("APP_ENV") or _discover_app_env_from_files(env_dir) or None
    )
    merged: Dict[str, str] = {}
    env_specific: Dict[str, str] = {}
    if app_env is not None:
        app_env = app_env.strip()
        if app_env:
            env_specific = _read_env_file(env_dir / f".env.{app_env}")
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
    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=True,
        validate_assignment=True,
    )

    #
    # General
    #

    APP_ENV: str = "dev"
    LOG_LEVEL: Optional[int] = None
    PYTHONPATH: str = "."
    CONFIDENT_REGION: Optional[str] = None
    CONFIDENT_OPEN_BROWSER: Optional[bool] = True

    #
    # CLI
    #

    DEEPEVAL_DEFAULT_SAVE: Optional[str] = None
    DEEPEVAL_DISABLE_DOTENV: Optional[bool] = None
    ENV_DIR_PATH: Optional[Path] = (
        None  # where .env files live (CWD if not set)
    )
    DEEPEVAL_FILE_SYSTEM: Optional[str] = None
    DEEPEVAL_IDENTIFIER: Optional[str] = None

    #
    # Storage & Output
    #

    # When set, DeepEval will export a timestamped JSON of the latest test run
    # into this directory. The directory will be created on demand.
    DEEPEVAL_RESULTS_FOLDER: Optional[Path] = None

    # Display / Truncation
    DEEPEVAL_MAXLEN_TINY: Optional[int] = 40
    DEEPEVAL_MAXLEN_SHORT: Optional[int] = 60
    DEEPEVAL_MAXLEN_MEDIUM: Optional[int] = 120
    DEEPEVAL_MAXLEN_LONG: Optional[int] = 240

    # If set, this overrides the default max_len used by deepeval/utils shorten
    # falls back to DEEPEVAL_MAXLEN_LONG when None.
    DEEPEVAL_SHORTEN_DEFAULT_MAXLEN: Optional[int] = None

    # Optional global suffix (keeps your "..." default).
    DEEPEVAL_SHORTEN_SUFFIX: Optional[str] = "..."

    #
    # GPU and perf toggles
    #

    CUDA_LAUNCH_BLOCKING: Optional[bool] = None
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    TOKENIZERS_PARALLELISM: Optional[bool] = None
    TRANSFORMERS_NO_ADVISORY_WARNINGS: Optional[bool] = None

    #
    # Model Keys
    #

    API_KEY: Optional[SecretStr] = None
    CONFIDENT_API_KEY: Optional[SecretStr] = None

    # ======
    # Base URL for Confident AI API server
    # ======
    CONFIDENT_BASE_URL: Optional[str] = None

    # General
    TEMPERATURE: Optional[confloat(ge=0, le=2)] = None

    # Anthropic
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    # Azure Open AI
    AZURE_OPENAI_API_KEY: Optional[SecretStr] = None
    AZURE_OPENAI_ENDPOINT: Optional[AnyUrl] = None
    OPENAI_API_VERSION: Optional[str] = None
    AZURE_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_MODEL_NAME: Optional[str] = None
    AZURE_MODEL_VERSION: Optional[str] = None
    USE_AZURE_OPENAI: Optional[bool] = None
    # DeepSeek
    USE_DEEPSEEK_MODEL: Optional[bool] = None
    DEEPSEEK_API_KEY: Optional[SecretStr] = None
    DEEPSEEK_MODEL_NAME: Optional[str] = None
    # Gemini
    USE_GEMINI_MODEL: Optional[bool] = None
    GOOGLE_API_KEY: Optional[SecretStr] = None
    GEMINI_MODEL_NAME: Optional[str] = None
    GOOGLE_GENAI_USE_VERTEXAI: Optional[bool] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_CLOUD_LOCATION: Optional[str] = None
    GOOGLE_SERVICE_ACCOUNT_KEY: Optional[str] = None
    # Grok
    USE_GROK_MODEL: Optional[bool] = None
    GROK_API_KEY: Optional[SecretStr] = None
    GROK_MODEL_NAME: Optional[str] = None
    # LiteLLM
    USE_LITELLM: Optional[bool] = None
    LITELLM_API_KEY: Optional[SecretStr] = None
    LITELLM_MODEL_NAME: Optional[str] = None
    LITELLM_API_BASE: Optional[AnyUrl] = None
    LITELLM_PROXY_API_BASE: Optional[AnyUrl] = None
    LITELLM_PROXY_API_KEY: Optional[SecretStr] = None
    # LM Studio
    LM_STUDIO_API_KEY: Optional[SecretStr] = None
    LM_STUDIO_MODEL_NAME: Optional[str] = None
    # Local Model
    USE_LOCAL_MODEL: Optional[bool] = None
    LOCAL_MODEL_API_KEY: Optional[SecretStr] = None
    LOCAL_EMBEDDING_API_KEY: Optional[SecretStr] = None
    LOCAL_MODEL_NAME: Optional[str] = None
    LOCAL_MODEL_BASE_URL: Optional[AnyUrl] = None
    LOCAL_MODEL_FORMAT: Optional[str] = None
    # Moonshot
    USE_MOONSHOT_MODEL: Optional[bool] = None
    MOONSHOT_API_KEY: Optional[SecretStr] = None
    MOONSHOT_MODEL_NAME: Optional[str] = None
    # Ollama
    OLLAMA_MODEL_NAME: Optional[str] = None
    # OpenAI
    USE_OPENAI_MODEL: Optional[bool] = None
    OPENAI_API_KEY: Optional[SecretStr] = None
    OPENAI_MODEL_NAME: Optional[str] = None
    OPENAI_COST_PER_INPUT_TOKEN: Optional[float] = None
    OPENAI_COST_PER_OUTPUT_TOKEN: Optional[float] = None
    # PortKey
    USE_PORTKEY_MODEL: Optional[bool] = None
    PORTKEY_API_KEY: Optional[SecretStr] = None
    PORTKEY_MODEL_NAME: Optional[str] = None
    PORTKEY_BASE_URL: Optional[AnyUrl] = None
    PORTKEY_PROVIDER_NAME: Optional[str] = None
    # Vertex AI
    VERTEX_AI_MODEL_NAME: Optional[str] = None
    # VLLM
    VLLM_API_KEY: Optional[SecretStr] = None
    VLLM_MODEL_NAME: Optional[str] = None

    #
    # Embedding Keys
    #

    # Azure OpenAI
    USE_AZURE_OPENAI_EMBEDDING: Optional[bool] = None
    AZURE_EMBEDDING_DEPLOYMENT_NAME: Optional[str] = None
    # Local
    USE_LOCAL_EMBEDDINGS: Optional[bool] = None
    LOCAL_EMBEDDING_MODEL_NAME: Optional[str] = None
    LOCAL_EMBEDDING_BASE_URL: Optional[AnyUrl] = None

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
    DEEPEVAL_SDK_RETRY_PROVIDERS: Optional[List[str]] = (
        None  # ["*"] to delegate all retries to SDKs
    )
    DEEPEVAL_RETRY_BEFORE_LOG_LEVEL: Optional[int] = (
        None  # default is LOG_LEVEL if set, else INFO
    )
    DEEPEVAL_RETRY_AFTER_LOG_LEVEL: Optional[int] = None  # default -> ERROR
    DEEPEVAL_RETRY_MAX_ATTEMPTS: conint(ge=1) = (
        2  # attempts = first try + retries
    )
    DEEPEVAL_RETRY_INITIAL_SECONDS: confloat(ge=0) = (
        1.0  # first sleep before retry, if any
    )
    DEEPEVAL_RETRY_EXP_BASE: confloat(ge=1) = (
        2.0  # exponential growth factor for sleeps
    )
    DEEPEVAL_RETRY_JITTER: confloat(ge=0) = 2.0  # uniform jitter
    DEEPEVAL_RETRY_CAP_SECONDS: confloat(ge=0) = (
        5.0  # cap for each backoff sleep
    )

    #
    # Telemetry and Debug
    #
    DEEPEVAL_DEBUG_ASYNC: Optional[bool] = None
    DEEPEVAL_TELEMETRY_OPT_OUT: Optional[bool] = None
    DEEPEVAL_UPDATE_WARNING_OPT_IN: Optional[bool] = None
    DEEPEVAL_GRPC_LOGGING: Optional[bool] = None
    GRPC_VERBOSITY: Optional[str] = None
    GRPC_TRACE: Optional[str] = None
    ERROR_REPORTING: Optional[bool] = None
    IGNORE_DEEPEVAL_ERRORS: Optional[bool] = None
    SKIP_DEEPEVAL_MISSING_PARAMS: Optional[bool] = None
    DEEPEVAL_VERBOSE_MODE: Optional[bool] = None
    DEEPEVAL_LOG_STACK_TRACES: Optional[bool] = None
    ENABLE_DEEPEVAL_CACHE: Optional[bool] = None

    CONFIDENT_TRACE_FLUSH: Optional[bool] = None
    CONFIDENT_TRACE_ENVIRONMENT: Optional[str] = "development"
    CONFIDENT_TRACE_VERBOSE: Optional[bool] = True
    CONFIDENT_TRACE_SAMPLE_RATE: Optional[float] = 1.0

    CONFIDENT_METRIC_LOGGING_FLUSH: Optional[bool] = None
    CONFIDENT_METRIC_LOGGING_VERBOSE: Optional[bool] = True
    CONFIDENT_METRIC_LOGGING_SAMPLE_RATE: Optional[float] = 1.0
    CONFIDENT_METRIC_LOGGING_ENABLED: Optional[bool] = True

    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[AnyUrl] = None

    #
    # Network
    #
    MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS: float = 3.05
    MEDIA_IMAGE_READ_TIMEOUT_SECONDS: float = 10.0
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
        None
    )

    #
    # Async Document Pipelines
    #

    DEEPEVAL_MAX_CONCURRENT_DOC_PROCESSING: conint(ge=1) = 2

    #
    # Async Task Configuration
    #
    DEEPEVAL_TIMEOUT_THREAD_LIMIT: conint(ge=1) = 128
    DEEPEVAL_TIMEOUT_SEMAPHORE_WARN_AFTER_SECONDS: confloat(ge=0) = 5.0
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
    DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE: Optional[confloat(ge=0)] = None

    # Buffer time for gathering results from all tasks, added to the longest task duration
    # Increase if many tasks are running concurrently
    # DEEPEVAL_TASK_GATHER_BUFFER_SECONDS: confloat(ge=0) = (
    #     30  # 15s seemed like not enough. we may make this computed later.
    # )
    DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE: Optional[confloat(ge=0)] = (
        None
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
        "CONFIDENT_OPEN_BROWSER",
        "CONFIDENT_TRACE_FLUSH",
        "CONFIDENT_TRACE_VERBOSE",
        "CUDA_LAUNCH_BLOCKING",
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
            self.result: Optional[PersistResult] = None

        @property
        def s(self) -> "Settings":
            return self._s

        def __enter__(self) -> "Settings._SettingsEditCtx":
            # snapshot current state
            self._before = {
                k: getattr(self._s, k) for k in type(self._s).model_fields
            }
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is not None:
                return False  # don’t persist on error

            from deepeval.config.settings_manager import (
                update_settings_and_persist,
                _normalize_for_env,
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
            after_norm = {k: _normalize_for_env(v) for k, v in after.items()}

            changed_keys = {
                k for k in after_norm if after_norm[k] != before_norm.get(k)
            }
            changed_keys -= self.COMPUTED_FIELDS

            if not changed_keys:
                self.result = PersistResult(False, None, {})
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
                    if _is_secret_key(self._s, k):
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
                updates,
                save=self._save,
                persist_dotenv=(False if self._persist is False else True),
            )
            self.result = PersistResult(handled, path, updates)
            return False

        def switch_model_provider(self, target) -> None:
            """
            Flip all USE_* toggles so that the one matching the target is True and the rest are False.
            Also,  mirror this change into the legacy JSON keystore as "YES"/"NO".

            `target` may be an Enum with `.value`, such as ModelKeyValues.USE_OPENAI_MODEL
            or a plain string like "USE_OPENAI_MODEL".
            """
            from deepeval.key_handler import KEY_FILE_HANDLER

            # Target key is the env style string, such as "USE_OPENAI_MODEL"
            target_key = getattr(target, "value", str(target))

            use_fields = [
                k for k in type(self._s).model_fields if k.startswith("USE_")
            ]
            if target_key not in use_fields:
                raise ValueError(
                    f"{target_key} is not a recognized USE_* field"
                )

            for k in use_fields:
                on = k == target_key
                # dotenv persistence will serialize to "1"/"0"
                setattr(self._s, k, on)
                if self._persist is not False:
                    # legacy json persistence will serialize to "YES"/"NO"
                    legacy_member = _find_legacy_enum(k)
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
_settings_env_fingerprint: "str | None" = None
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
