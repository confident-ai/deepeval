"""
Applies CLI driven updates to the live Settings and optionally persists them to a
dotenv file. Also syncs os.environ, handles unsets, and warns on unknown fields.
Primary entrypoint: update_settings_and_persist.
"""

import json
import logging
import os

from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union
from enum import Enum

from pydantic import SecretStr
from deepeval.config.settings import get_settings, _SAVE_RE
from deepeval.cli.dotenv_handler import DotenvHandler
from deepeval.config.utils import bool_to_env_str

logger = logging.getLogger(__name__)
StrOrEnum = Union[str, Enum]


def _env_key(k: StrOrEnum) -> str:
    return k.value if isinstance(k, Enum) else str(k)


def _normalize_for_env(val: Any) -> Optional[str]:
    """Convert typed value to string for dotenv + os.environ; None -> unset."""
    if val is None:
        return None
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    if isinstance(val, bool):
        return bool_to_env_str(val)
    # encode sequences as JSON so Settings can parse them back reliably.
    if isinstance(val, (list, tuple, set)):
        return json.dumps(list(val))
    return str(val)


def _resolve_save_path(save_opt: Optional[str]) -> Tuple[bool, Optional[Path]]:
    """
    Returns (ok, path).
      - ok=False -> invalid save option format
      - ok=True, path=None -> no persistence requested
      - ok=True, path=Path -> persist to that file
    """
    raw = (
        save_opt if save_opt is not None else os.getenv("DEEPEVAL_DEFAULT_SAVE")
    )
    if not raw:
        return True, None
    m = _SAVE_RE.match(raw.strip())
    if not m:
        return False, None
    path = m.group("path") or ".env.local"
    path = Path(os.path.expanduser(os.path.expandvars(path)))
    return True, path


def update_settings_and_persist(
    updates: Mapping[StrOrEnum, Any],
    *,
    save: Optional[str] = None,
    unset: Iterable[StrOrEnum] = (),
    persist_dotenv: bool = True,
) -> Tuple[bool, Optional[Path]]:
    """
    Write and update:
      - validate + assign into live Settings()
      - update os.environ
      - persist to dotenv, if `save` or DEEPEVAL_DEFAULT_SAVE provided
      - unset keys where value is None or explicitly in `unset`
    Returns (handled, path_to_dotenv_if_any).
    """
    settings = get_settings()

    # validate + assign into settings.
    # validation is handled in Settings as long as validate_assignment=True
    typed: Dict[str, Any] = {}
    for key, value in updates.items():
        k = _env_key(key)
        if k not in type(settings).model_fields:
            suggestion = get_close_matches(
                k, type(settings).model_fields.keys(), n=1
            )
            if suggestion:
                logger.warning(
                    "Unknown settings field '%s'; did you mean '%s'? Ignoring.",
                    k,
                    suggestion[0],
                    stacklevel=2,
                )
            else:
                logger.warning(
                    "Unknown settings field '%s'; ignoring.", k, stacklevel=2
                )
            continue

        setattr(settings, k, value)
        # coercion is handled in Settings
        typed[k] = getattr(settings, k)

    # build env maps
    to_write: Dict[str, str] = {}
    to_unset: set[str] = set(_env_key(k) for k in unset)

    for k, v in typed.items():
        env_val = _normalize_for_env(v)
        if env_val is None:
            to_unset.add(k)
        else:
            to_write[k] = env_val

    # update process env so that it is effective immediately
    for k, v in to_write.items():
        os.environ[k] = v
    for k in to_unset:
        os.environ.pop(k, None)

    if not persist_dotenv:
        return True, None

    # persist to dotenv if save is ok
    ok, path = _resolve_save_path(save)
    if not ok:
        return False, None  # unsupported --save
    if path:
        h = DotenvHandler(path)
        if to_write:
            h.upsert(to_write)
        if to_unset:
            h.unset(to_unset)
        return True, path
    return True, None
