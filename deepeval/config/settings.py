import os
import re

from dotenv import dotenv_values
from pathlib import Path
from pydantic import AnyUrl, SecretStr, field_validator, confloat
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Literal, Optional

from deepeval.utils import parse_bool


_SAVE_RE = re.compile(r"^(?P<scheme>dotenv)(?::(?P<path>.+))?$")


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
    # prefer base .env, then .env.local for APP_ENV discovery
    for name in (".env", ".env.local"):
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

    # Pick APP_ENV (process -> .env -> .env.local -> default)
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True)

    #
    # General
    #

    APP_ENV: str = "dev"
    LOG_LEVEL: str = "info"
    PYTHONPATH: str = "."

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
    # Telemetry and Debug
    #
    DEEPEVAL_TELEMETRY_OPT_OUT: Optional[bool] = None
    DEEPEVAL_UPDATE_WARNING_OPT_IN: Optional[bool] = None
    DEEPEVAL_GRPC_LOGGING: Optional[bool] = None
    GRPC_VERBOSITY: Optional[str] = None
    GRPC_TRACE: Optional[str] = None
    ERROR_REPORTING: Optional[bool] = None
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[AnyUrl] = None

    @field_validator(
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
        # don't resolve to avoid failing on non-existent paths
        return Path(os.path.expandvars(os.path.expanduser(s)))

    # Treat "", "none", "null" as None for numeric overrides
    @field_validator(
        "OPENAI_COST_PER_INPUT_TOKEN",
        "OPENAI_COST_PER_OUTPUT_TOKEN",
        "TEMPERATURE",
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
        # allow some friendly aliases; keep or trim as you like
        if s in {"READ_ONLY", "READ-ONLY", "READONLY", "RO"}:
            return "READ_ONLY"
        raise ValueError(
            "DEEPEVAL_FILE_SYSTEM must be READ_ONLY (case-insensitive)."
        )
