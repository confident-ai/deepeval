"""Builds the GitHub Actions workflow `deepeval link-ci` adds to a repository."""

import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List

WORKFLOW_PATH = ".github/workflows/deepeval.yml"


class PackageManager(Enum):
    UV = "uv"
    POETRY = "poetry"
    PIP_REQUIREMENTS = "pip-requirements"
    PIP = "pip"


@dataclass(frozen=True)
class WorkflowConfig:
    default_branch: str
    python_version: str
    package_manager: PackageManager
    test_run_args: List[str]
    secret_names: List[str]
    working_directory: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    requirements_mention_deepeval: bool = False


def detect_package_manager(directory: Path) -> PackageManager:
    if (directory / "uv.lock").is_file():
        return PackageManager.UV
    if (directory / "poetry.lock").is_file():
        return PackageManager.POETRY
    if (directory / "requirements.txt").is_file():
        return PackageManager.PIP_REQUIREMENTS
    return PackageManager.PIP


def requirements_mention_deepeval(directory: Path) -> bool:
    try:
        text = (directory / "requirements.txt").read_text(encoding="utf-8")
    except OSError:
        return False
    return re.search(r"^\s*deepeval\b", text, re.MULTILINE) is not None


def detect_python_version(directory: Path, repository_root: Path) -> str:
    for candidate in (directory, repository_root):
        try:
            version = (candidate / ".python-version").read_text().strip()
        except OSError:
            continue
        if re.fullmatch(r"\d+\.\d+(\.\d+)?", version):
            return version
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _install_steps(config: WorkflowConfig) -> List[str]:
    manager = config.package_manager
    if manager is PackageManager.UV:
        return [
            "      - uses: astral-sh/setup-uv@v5",
            "      - name: Install dependencies",
            "        run: uv sync",
        ]
    if manager is PackageManager.POETRY:
        return [
            "      - name: Install dependencies",
            "        run: |",
            "          pipx install poetry",
            "          poetry install",
        ]
    if manager is PackageManager.PIP_REQUIREMENTS:
        install = "pip install -r requirements.txt"
        if not config.requirements_mention_deepeval:
            install += " deepeval"
        return ["      - name: Install dependencies", f"        run: {install}"]
    return [
        "      - name: Install dependencies",
        "        run: pip install deepeval",
    ]


def _run_prefix(manager: PackageManager) -> str:
    if manager is PackageManager.UV:
        return "uv run "
    if manager is PackageManager.POETRY:
        return "poetry run "
    return ""


def build_workflow(config: WorkflowConfig) -> str:
    env_lines = [
        f"      {name}: ${{{{ secrets.{name} }}}}"
        for name in config.secret_names
    ] + [
        f"      {name}: {json.dumps(value)}"
        for name, value in sorted(config.env.items())
    ]
    lines = [
        "name: DeepEval",
        "",
        "on:",
        "  pull_request:",
        "  push:",
        f"    branches: [{json.dumps(config.default_branch)}]",
        "",
        "permissions:",
        "  contents: read",
        "",
        "jobs:",
        "  deepeval:",
        "    name: deepeval test run",
        "    runs-on: ubuntu-latest",
    ]
    if env_lines:
        lines += ["    env:", *env_lines]
    if config.working_directory:
        lines += [
            "    defaults:",
            "      run:",
            f"        working-directory: {json.dumps(config.working_directory)}",
        ]
    lines += [
        "    steps:",
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        "        with:",
        f"          python-version: {json.dumps(config.python_version)}",
        *_install_steps(config),
        "      - name: Run deepeval tests",
        "        run: |",
        "          "
        + _run_prefix(config.package_manager)
        + shlex.join(["deepeval", "test", "run", *config.test_run_args]),
    ]
    return "\n".join(lines) + "\n"
