"""
Minimal logging configuration helpers for DeepEval.

This module centralizes how the library-level logger ("deepeval") is configured. We
intentionally keep configuration lightweight so application code retains control
over handlers and formatters.
"""

import logging
from deepeval.config.settings import get_settings


def apply_deepeval_log_level() -> None:
    """
    Apply DeepEval's current log level to the package logger.

    This function reads `LOG_LEVEL` from `deepeval.config.settings.get_settings()`
    and sets the level of the `"deepeval"` logger accordingly. If `LOG_LEVEL` is
    unset (None), INFO is used as a default. The logger's `propagate` flag is set
    to True so records bubble up to the application's handlers. DeepEval does not
    install its own handlers here (a NullHandler is attached in `__init__.py`).

    The function is idempotent and safe to call multiple times. It is invoked
    automatically when settings are first constructed and whenever `LOG_LEVEL`
    is changed via `settings.edit`.
    """
    settings = get_settings()
    log_level = settings.LOG_LEVEL
    logging.getLogger("deepeval").setLevel(
        log_level if log_level is not None else logging.INFO
    )
    # ensure we bubble up to app handlers
    logging.getLogger("deepeval").propagate = True
