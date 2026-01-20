from __future__ import annotations
import json
import os
import pyfiglet
import typer
import webbrowser

from pydantic import ValidationError
from pydantic.fields import FieldInfo
from enum import Enum
from pathlib import Path
from rich import print
from typing import (
    Any,
    Dict,
    Iterable,
    Tuple,
    Optional,
    get_args,
    get_origin,
    Union,
)
from opentelemetry.trace import Span

from deepeval.config.settings import Settings, get_settings
from deepeval.key_handler import (
    KEY_FILE_HANDLER,
    ModelKeyValues,
    EmbeddingKeyValues,
)
from deepeval.test_run.test_run import (
    global_test_run_manager,
)
from deepeval.confident.api import get_confident_api_key, set_confident_api_key
from deepeval.cli.dotenv_handler import DotenvHandler


StrOrEnum = Union[str, "Enum"]
PROD = "https://app.confident-ai.com"
# List all mutually exclusive USE_* keys
USE_LLM_KEYS = [
    key
    for key in Settings.model_fields
    if key.startswith("USE_") and key in ModelKeyValues.__members__
]
USE_EMBED_KEYS = [
    key
    for key in Settings.model_fields
    if key.startswith("USE_") and key in EmbeddingKeyValues.__members__
]


def render_login_message():
    print(
        "🥳 Welcome to [rgb(106,0,255)]Confident AI[/rgb(106,0,255)], the evals cloud platform 🏡❤️"
    )
    print("")
    print(pyfiglet.Figlet(font="big_money-ne").renderText("Confident AI"))


def upload_and_open_link(_span: Span):
    last_test_run_data = global_test_run_manager.get_latest_test_run_data()
    if last_test_run_data:
        confident_api_key = get_confident_api_key()
        if confident_api_key == "" or confident_api_key is None:
            render_login_message()

            print(
                f"🔑 You'll need to get an API key at [link={PROD}]{PROD}[/link] to view your results (free)"
            )
            webbrowser.open(PROD)
            while True:
                confident_api_key = input("🔐 Enter your API Key: ").strip()
                if confident_api_key:
                    set_confident_api_key(confident_api_key)
                    print(
                        "\n🎉🥳 Congratulations! You've successfully logged in! :raising_hands: "
                    )
                    _span.set_attribute("completed", True)
                    break
                else:
                    print("❌ API Key cannot be empty. Please try again.\n")

        print("📤 Uploading test run to Confident AI...")
        global_test_run_manager.post_test_run(last_test_run_data)
    else:
        print(
            "❌ No test run found in cache. Run 'deepeval login' + an evaluation to get started 🚀."
        )


def clear_evaluation_model_keys():
    for key in ModelKeyValues:
        KEY_FILE_HANDLER.remove_key(key)


def clear_embedding_model_keys():
    for key in EmbeddingKeyValues:
        KEY_FILE_HANDLER.remove_key(key)


def _to_str_key(k: StrOrEnum) -> str:
    return k.name if hasattr(k, "name") else str(k)


def _normalize_kv(updates: Dict[StrOrEnum, str]) -> Dict[str, str]:
    return {_to_str_key(k): v for k, v in updates.items()}


def _normalize_keys(keys: Iterable[StrOrEnum]) -> list[str]:
    return [_to_str_key(k) for k in keys]


def _normalize_setting_key(raw_key: str) -> str:
    """Normalize CLI keys like 'log-level' / 'LOG_LEVEL' to model field names."""
    return raw_key.strip().lower().replace("-", "_")


def _parse_save_option(
    save_opt: Optional[str] = None, default_path: str = ".env.local"
) -> Tuple[bool, Optional[str]]:
    if not save_opt:
        return False, None
    kind, *rest = save_opt.split(":", 1)
    if kind != "dotenv":
        return False, None
    path = rest[0] if rest else default_path
    return True, path


def resolve_save_target(save_opt: Optional[str]) -> Optional[str]:
    """
    Returns a normalized save target string like 'dotenv:.env.local' or None.
    Precedence:
      1) --save=...
      2) DEEPEVAL_DEFAULT_SAVE (opt-in project default)
      3) None (no save)
    """
    if save_opt:
        return save_opt

    env_default = os.getenv("DEEPEVAL_DEFAULT_SAVE")
    if env_default and env_default.strip():
        return env_default.strip()

    return None


def save_environ_to_store(
    updates: Dict[StrOrEnum, str], save_opt: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Save 'updates' into the selected store (currently only dotenv). Idempotent upsert.
    Returns (handled, path).
    """
    ok, path = _parse_save_option(save_opt)
    if not ok:
        return False, None
    if updates:
        DotenvHandler(path).upsert(_normalize_kv(updates))
    return True, path


def unset_environ_in_store(
    keys: Iterable[StrOrEnum], save_opt: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Remove keys from the selected store (currently only dotenv).
    Returns (handled, path).
    """
    ok, path = _parse_save_option(save_opt)
    if not ok:
        return False, None
    norm = _normalize_keys(keys)
    if norm:
        DotenvHandler(path).unset(norm)
    return True, path


def _as_legacy_use_key(
    k: str,
) -> Union[ModelKeyValues, EmbeddingKeyValues, None]:
    if k in ModelKeyValues.__members__:
        return ModelKeyValues[k]
    if k in EmbeddingKeyValues.__members__:
        return EmbeddingKeyValues[k]
    return None


def switch_model_provider(
    target: Union[ModelKeyValues, EmbeddingKeyValues],
    save: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Ensure exactly one USE_* flag is enabled.
    We *unset* all other USE_* keys (instead of writing explicit "NO") to:
      - keep dotenv clean
      - preserve Optional[bool] semantics (unset vs explicit false)
    """
    keys_to_clear = (
        USE_LLM_KEYS if isinstance(target, ModelKeyValues) else USE_EMBED_KEYS
    )
    target_key = target.name  # or _to_str_key(target)

    if target_key not in keys_to_clear:
        raise ValueError(f"{target} is not a recognized USE_* model key")

    # Clear legacy JSON store entries
    for k in keys_to_clear:
        legacy = _as_legacy_use_key(k)
        if legacy is not None:
            KEY_FILE_HANDLER.remove_key(legacy)

    KEY_FILE_HANDLER.write_key(target, "YES")

    if not save:
        return True, None

    handled, path = unset_environ_in_store(keys_to_clear, save)
    if not handled:
        return False, None
    return save_environ_to_store({target: "true"}, save)


def coerce_blank_to_none(value: Optional[str]) -> Optional[str]:
    """Return None if value is None/blank/whitespace; otherwise return stripped string."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_service_account_key_file(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise typer.BadParameter(
            f"Could not read service account file: {path}",
            param_hint="--service-account-file",
        ) from e

    if not raw:
        raise typer.BadParameter(
            f"Service account file is empty: {path}",
            param_hint="--service-account-file",
        )

    # Validate it's JSON and normalize to a single-line string for dotenv.
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(
            f"Service account file does not contain valid JSON: {path}",
            param_hint="--service-account-file",
        ) from e

    return json.dumps(obj, separators=(",", ":"))


def unwrap_optional(annotation: Any) -> Any:
    """
    If `annotation` is Optional[T] (i.e. Union[T, None]), return T.
    Otherwise return `annotation` unchanged.

    Note: If it's a Union with multiple non-None members, we leave it unchanged.
    """
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def looks_like_json_container_literal(raw_value: str) -> bool:
    setting = raw_value.strip()
    return (setting.startswith("{") and setting.endswith("}")) or (
        setting.startswith("[") and setting.endswith("]")
    )


def should_parse_json_for_field(field_info: FieldInfo) -> bool:
    annotation = unwrap_optional(field_info.annotation)
    origin = get_origin(annotation) or annotation
    return origin in (list, dict, tuple, set)


def maybe_parse_json_literal(raw_value: str, field_info) -> object:
    if not isinstance(raw_value, str):
        return raw_value
    if not looks_like_json_container_literal(raw_value):
        return raw_value
    if not should_parse_json_for_field(field_info):
        return raw_value
    try:
        return json.loads(raw_value)
    except Exception as e:
        raise typer.BadParameter(f"Invalid JSON for {field_info}: {e}") from e


def resolve_field_names(settings, query: str) -> list[str]:
    """Return matching Settings fields for a case-insensitive partial query."""
    fields = type(settings).model_fields
    query = _normalize_setting_key(query)

    # exact match (case-insensitive) first
    exact = [
        name for name in fields.keys() if _normalize_setting_key(name) == query
    ]
    if exact:
        return exact

    # substring matches
    return [
        name for name in fields.keys() if query in _normalize_setting_key(name)
    ]


def is_optional(annotation) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def parse_and_validate(field_name: str, field_info, raw: str):
    """
    Validate and coerce a CLI value by delegating to the Settings model.

    Field validators like LOG_LEVEL coercion (e.g. 'error' -> numeric log level)
    are applied.
    """
    settings = get_settings()
    value: object = maybe_parse_json_literal(raw, field_info)
    payload = settings.model_dump(mode="python")
    payload[field_name] = value

    try:
        validated = type(settings).model_validate(payload)
    except ValidationError as e:
        # Surface field-specific error(s) if possible
        field_errors: list[str] = []
        for err in e.errors():
            loc = err.get("loc") or ()
            if loc and loc[0] == field_name:
                field_errors.append(err.get("msg") or str(err))

        detail = "; ".join(field_errors) if field_errors else str(e)
        raise typer.BadParameter(
            f"Invalid value for {field_name}: {raw!r}. {detail}"
        ) from e

    return getattr(validated, field_name)
