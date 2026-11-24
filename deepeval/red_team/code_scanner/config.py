from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from .schema import Severity

CONFIG_FILENAMES = (".deepteam-code-scan.yaml",)


@dataclass
class CodeScanConfig:
    """
    Settings for a code scan, loaded from a repo-root YAML file.

    The instruction field is passed straight through to the scanner's
    `instruction` parameter when the CLI constructs it.
    """

    min_severity: Severity = "low"
    instruction: Optional[str] = None
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    vulnerabilities: Optional[List[str]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    diffs_only: bool = False


def find_config_file(directory: str = ".") -> Optional[str]:
    """
    Return the first recognized config file in `directory`, or None.
    """
    for name in CONFIG_FILENAMES:
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    return None


def load_config(
    path: Optional[str] = None, directory: str = "."
) -> CodeScanConfig:
    """
    Load scan config from an explicit `path`, or auto-discover one of
    CONFIG_FILENAMES in `directory`. Returns defaults if none is found.
    """
    config_path = path or find_config_file(directory)
    if not config_path or not Path(config_path).is_file():
        return CodeScanConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return CodeScanConfig(
        min_severity=raw.get("min_severity", "low"),
        instruction=raw.get("instruction"),
        include=raw.get("include"),
        exclude=raw.get("exclude"),
        vulnerabilities=raw.get("vulnerabilities"),
        model=raw.get("model"),
        provider=raw.get("provider"),
        diffs_only=bool(raw.get("diffs_only", False)),
    )
