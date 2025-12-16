from __future__ import annotations

import os
import webbrowser
import pyfiglet

from enum import Enum
from rich import print
from typing import Optional, Dict, Iterable, Tuple, Union
from opentelemetry.trace import Span

from deepeval.config.settings import Settings
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
        "🥳 Welcome to [rgb(106,0,255)]Confident AI[/rgb(106,0,255)], the DeepEval cloud platform 🏡❤️"
    )
    print("")
    print(pyfiglet.Figlet(font="big_money-ne").renderText("DeepEval Cloud"))


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
