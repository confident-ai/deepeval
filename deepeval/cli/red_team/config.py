from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

CONFIG_FILE = Path.home() / ".deepteam" / "config.txt"


def _read() -> Dict[str, str]:
    if CONFIG_FILE.exists():
        data: Dict[str, str] = {}
        for line in CONFIG_FILE.read_text().splitlines():
            key, _, value = line.partition("=")
            if key:
                data[key] = value
        return data
    return {}


def _write(data: Dict[str, str]):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text("\n".join(f"{k}={v}" for k, v in data.items()))


def set_key(key: str, value: str):
    data = _read()
    data[key] = value
    _write(data)


def get_key(key: str) -> str | None:
    return _read().get(key)


def remove_key(key: str):
    data = _read()
    if key in data:
        del data[key]
        _write(data)


def apply_env():
    """Apply stored keys to os.environ."""
    for k, v in _read().items():
        os.environ.setdefault(k, v)
