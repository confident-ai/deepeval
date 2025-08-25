from __future__ import annotations

import os
import webbrowser
import pyfiglet

from enum import Enum
from pathlib import Path
from rich import print
from typing import Optional, Dict, Iterable, List, Tuple, Union
from opentelemetry.trace import Span

from deepeval.key_handler import (
    KEY_FILE_HANDLER,
    KeyValues,
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
USE_MODEL_KEYS: List[ModelKeyValues | EmbeddingKeyValues] = [
    ModelKeyValues.USE_OPENAI_MODEL,
    ModelKeyValues.USE_AZURE_OPENAI,
    ModelKeyValues.USE_LOCAL_MODEL,
    ModelKeyValues.USE_GROK_MODEL,
    ModelKeyValues.USE_MOONSHOT_MODEL,
    ModelKeyValues.USE_DEEPSEEK_MODEL,
    ModelKeyValues.USE_GEMINI_MODEL,
    ModelKeyValues.USE_LITELLM,
    EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING,
    EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS,
    # MAINTENANCE: add more if new USE_* keys appear
]


def render_login_message():
    print(
        f"🥳 Welcome to [rgb(106,0,255)]Confident AI[/rgb(106,0,255)], the DeepEval cloud platform 🏡❤️"
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

        print(f"📤 Uploading test run to Confident AI...")
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
    return k.value if hasattr(k, "value") else str(k)


def _normalize_kv(updates: Dict[StrOrEnum, str]) -> Dict[str, str]:
    return {_to_str_key(k): v for k, v in updates.items()}


def _normalize_keys(keys: Iterable[StrOrEnum]) -> list[str]:
    return [_to_str_key(k) for k in keys]


def _parse_save_option(
    save_opt: str | None, default_path: str = ".env.local"
) -> Tuple[bool, str | None]:
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
    save_opt: str | None, updates: Dict[StrOrEnum, str]
) -> Tuple[bool, str | None]:
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
    save_opt: str | None, keys: Iterable[StrOrEnum]
) -> Tuple[bool, str | None]:
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


def switch_model_provider(target: ModelKeyValues, save: str = None) -> None:
    """
    Ensure exactly one USE_* model flag is set to "YES" and the rest to "NO",
    both in the .deepeval json store and in a dotenv file (if save is provided).
    """
    if target not in USE_MODEL_KEYS:
        raise ValueError(f"{target} is not a recognized USE_* model key")

    for key in USE_MODEL_KEYS:
        value = "YES" if key == target else "NO"
        KEY_FILE_HANDLER.write_key(key, value)

        if save:
            handled, path = save_environ_to_store(save, {key: value})
            if not handled:
                print("Unsupported --save option. Use --save=dotenv[:path].")
