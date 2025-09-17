"""
Central config for DeepEval.

- Autoloads dotenv files into os.environ without overwriting existing vars
  (order: .env -> .env.{APP_ENV} -> .env.local).
- Defines the Pydantic `Settings` model and `get_settings()` singleton.
- Exposes an `edit()` context manager that diffs changes and persists them to
  dotenv and the legacy JSON keystore (non-secret keys only), with validators and
  type coercion.
"""

import logging
import os
import re

from dotenv import dotenv_values
from pathlib import Path
from pydantic import AnyUrl, SecretStr, field_validator, confloat
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, List, Optional, NamedTuple

from deepeval.config.utils import (
    parse_bool,
    coerce_to_list,
    dedupe_preserve_order,
)
from deepeval.constants import SUPPORTED_PROVIDER_SLUGS, slugify


logger = logging.getLogger(__name__)
_SAVE_RE = re.compile(r"^(?P<scheme>dotenv)(?::(?P<path>.+))?$")


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
    LOG_LEVEL: str = "info"
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
    DEEPEVAL_SDK_RETRY_PROVIDERS: Optional[List[str]] = None
    DEEPEVAL_RETRY_BEFORE_LOG_LEVEL: Optional[int] = None  # default -> INFO
    DEEPEVAL_RETRY_AFTER_LOG_LEVEL: Optional[int] = None  # default -> ERROR

    #
    # Telemetry and Debug
    #
    DEEPEVAL_TELEMETRY_OPT_OUT: Optional[bool] = None
    DEEPEVAL_UPDATE_WARNING_OPT_IN: Optional[bool] = None
    DEEPEVAL_GRPC_LOGGING: Optional[bool] = None
    GRPC_VERBOSITY: Optional[str] = None
    GRPC_TRACE: Optional[str] = None
    ERROR_REPORTING: Optional[bool] = None
    IGNORE_DEEPEVAL_ERRORS: Optional[bool] = None
    SKIP_DEEPEVAL_MISSING_PARAMS: Optional[bool] = None
    DEEPEVAL_VERBOSE_MODE: Optional[bool] = None
    ENABLE_DEEPEVAL_CACHE: Optional[bool] = None
    CONFIDENT_TRACE_FLUSH: Optional[bool] = None
    CONFIDENT_TRACE_ENVIRONMENT: Optional[str] = "development"
    CONFIDENT_TRACE_VERBOSE: Optional[bool] = True
    CONFIDENT_SAMPLE_RATE: Optional[float] = 1.0
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[AnyUrl] = None

    #
    # Network
    #
    MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS: float = 3.05
    MEDIA_IMAGE_READ_TIMEOUT_SECONDS: float = 10.0

    ##############
    # Validators #
    ##############

    @field_validator(
        "CONFIDENT_OPEN_BROWSER",
        "CONFIDENT_TRACE_FLUSH",
        "CONFIDENT_TRACE_VERBOSE",
        "USE_OPENAI_MODEL",
        "USE_AZURE_OPENAI",
        "USE_LOCAL_MODEL",
        "USE_GEMINI_MODEL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "USE_MOONSHOT_MODEL",
        "USE_GROK_MODEL",
        "USE_DEEPSEEK_MODEL",
        "USE_LITELLM",
        "USE_AZURE_OPENAI_EMBEDDING",
        "USE_LOCAL_EMBEDDINGS",
        "DEEPEVAL_GRPC_LOGGING",
        "DEEPEVAL_DISABLE_DOTENV",
        "DEEPEVAL_TELEMETRY_OPT_OUT",
        "DEEPEVAL_UPDATE_WARNING_OPT_IN",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
        "CUDA_LAUNCH_BLOCKING",
        "ERROR_REPORTING",
        "IGNORE_DEEPEVAL_ERRORS",
        "SKIP_DEEPEVAL_MISSING_PARAMS",
        "DEEPEVAL_VERBOSE_MODE",
        "ENABLE_DEEPEVAL_CACHE",
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
        "CONFIDENT_SAMPLE_RATE",
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

    @field_validator("CONFIDENT_SAMPLE_RATE")
    @classmethod
    def _validate_sample_rate(cls, v):
        if v is None:
            return None
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("CONFIDENT_SAMPLE_RATE must be between 0 and 1")
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
                if cls.DEEPEVAL_VERBOSE_MODE:
                    logger.warning("Unknown provider slug %r dropped", item)

        if star:
            return ["*"]

        # It is important to dedup after normalization to catch variants
        normalized = dedupe_preserve_order(normalized)
        return normalized or None

    @field_validator(
        "DEEPEVAL_RETRY_BEFORE_LOG_LEVEL",
        "DEEPEVAL_RETRY_AFTER_LOG_LEVEL",
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

    #######################
    # Persistence support #
    #######################
    class _SettingsEditCtx:
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

            # compute diff of changed fields
            after = {k: getattr(self._s, k) for k in type(self._s).model_fields}

            before_norm = {
                k: _normalize_for_env(v) for k, v in self._before.items()
            }
            after_norm = {k: _normalize_for_env(v) for k, v in after.items()}

            changed_keys = {
                k for k in after_norm if after_norm[k] != before_norm.get(k)
            }
            if not changed_keys:
                self.result = PersistResult(False, None, {})
                return False

            updates = {k: after[k] for k in changed_keys}

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


_settings_singleton: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()
    return _settings_singleton
