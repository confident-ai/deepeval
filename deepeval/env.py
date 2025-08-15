from __future__ import annotations
import os

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except Exception:
    load_dotenv = None
    find_dotenv = None


def autoload_dotenv() -> None:
    """
    Autoload environment variables for DeepEval at import time.

    Precedence from highest -> lowest:
      1) Existing process environment variables
      2) .env.local (from current working directory)
      3) .env (from current working directory)

    Behavior:
      - Loads .env.local then .env if present, without overriding existing vars.
      - Opt-out by setting DEEPEVAL_DISABLE_DOTENV=1.
      - Soft-fails cleanly if python-dotenv is not installed.
    """
    if os.getenv("DEEPEVAL_DISABLE_DOTENV") == "1":
        return

    if not (load_dotenv and find_dotenv):
        return

    for name in (".env.local", ".env"):
        path = find_dotenv(name, usecwd=True)
        if path:
            # Don't override previously set values
            load_dotenv(path, override=False)
