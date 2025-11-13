"""Small runtime helpers for tagging traces with environment metadata."""

import sys
from typing import Optional
from importlib import metadata


def py_version_str() -> str:
    """Return the current Python version as 'MAJOR.MINOR.PATCH'.

    Useful for embedding into trace metadata without importing `platform`.
    """
    return ".".join(map(str, sys.version_info[:3]))


def pkg_version(pkg: str) -> Optional[str]:
    """Return the installed version string for `pkg`, or None if not
    found.
    """
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        # no point in breaking things if the package being queried doesn't exist
        # this allows for safe exploration.
        return None
