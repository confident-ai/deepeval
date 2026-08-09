#!/usr/bin/env python3
"""Bump both SDK versions everywhere they are declared, then build and publish.

    python .scripts/release.py
    python .scripts/release.py --dry-run

Every run releases both SDKs and always asks for both versions, one prompt each,
showing what that SDK is on now. Press enter to take the patch bump it offers,
or type `major`/`minor`/`patch` or an explicit version like `4.2.0`. Type `skip`
to leave one SDK where it is.

The version is declared in more places than the one you publish from. Python has
four (`pyproject.toml`, `deepeval/_version.py`, `CITATION.cff`, and the docs'
committed `sdk-versions.json`); TypeScript has two. `docs/scripts/
generate-sdk-versions.mjs` fails the docs build when the Python three disagree,
so a partial bump breaks docs rather than shipping quietly.

Tokens come from the environment, never from arguments:

    PYPI_TOKEN or POETRY_PYPI_TOKEN_PYPI   PyPI (a `pypi-` prefixed API token)
    NPM_TOKEN or NODE_AUTH_TOKEN           npm
    NPM_OTP                                npm one-time code, if 2FA is on

A successful publish is committed for you as "new release". The tree is expected
to be clean when you start, so the bump is the whole commit.

This never creates a git tag. `.github/workflows/release.yml` tags `python-v*` /
`typescript-v*` when a version file lands on `main`, and it skips a tag that
already exists — tagging locally means the workflow finds nothing new and drafts
no release. Push, and let it do that half.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_VERSIONS_JSON = "docs/lib/generated/sdk-versions.json"
COMMIT_MESSAGE = "new release"
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-.]?[0-9A-Za-z.-]+)?$")


@dataclass(frozen=True)
class Edit:
    """One capture group in one file, rewritten in place."""

    file: str
    pattern: str
    value: Callable[[str], str]  # new version -> replacement text
    is_version: bool = True  # False for fields that travel with a release, e.g. a date


def _version_edit(file: str, pattern: str) -> Edit:
    return Edit(file=file, pattern=pattern, value=lambda version: version)


@dataclass(frozen=True)
class Target:
    name: str
    package: str
    edits: List[Edit]
    json_key: str  # key in sdk-versions.json
    publish: Callable[["Target", str], None]
    registry_versions: Callable[[str], Optional[set]]


def pypi_versions(package: str) -> Optional[set]:
    data = _get_json(f"https://pypi.org/pypi/{package}/json")
    return set(data["releases"]) if data else None


def npm_versions(package: str) -> Optional[set]:
    data = _get_json(f"https://registry.npmjs.org/{package}")
    return set(data["versions"]) if data else None


def _get_json(url: str) -> Optional[dict]:
    """None on any network trouble — a pre-flight check is not worth failing on."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


# --------------------------------------------------------------------------- #
# file editing
# --------------------------------------------------------------------------- #


def read_version(edit: Edit) -> str:
    path = REPO_ROOT / edit.file
    if not path.exists():
        fail(f"{edit.file} is missing — run this from a full checkout.")
    match = re.search(edit.pattern, path.read_text(), re.MULTILINE)
    if not match:
        fail(f"could not find a version in {edit.file} (pattern: {edit.pattern})")
    return match.group(1)


def apply_edit(edit: Edit, version: str) -> None:
    path = REPO_ROOT / edit.file
    text = path.read_text()
    match = re.search(edit.pattern, text, re.MULTILINE)
    if not match:
        fail(f"could not find {edit.pattern} in {edit.file}")
    start, end = match.span(1)
    path.write_text(text[:start] + edit.value(version) + text[end:])


def read_json_version(key: str) -> Optional[str]:
    path = REPO_ROOT / SDK_VERSIONS_JSON
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(key)


def write_json_version(key: str, version: str) -> None:
    path = REPO_ROOT / SDK_VERSIONS_JSON
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data[key] = version
    path.write_text(f"{json.dumps(data, indent=2)}\n")


# --------------------------------------------------------------------------- #
# publishing
# --------------------------------------------------------------------------- #


def publish_python(target: Target, version: str) -> None:
    require_tool("poetry")
    token = first_env("PYPI_TOKEN", "POETRY_PYPI_TOKEN_PYPI", "PYPI_API_TOKEN")
    if not token:
        fail("set PYPI_TOKEN (or POETRY_PYPI_TOKEN_PYPI) to publish to PyPI.")

    # `poetry publish` uploads everything sitting in dist/, including artifacts
    # from older releases, so a stale dist/ turns one release into many.
    dist = REPO_ROOT / "dist"
    if dist.exists():
        stale = [p for p in dist.iterdir() if p.is_file()]
        for path in stale:
            path.unlink()
        if stale:
            say(f"cleared {len(stale)} file(s) from dist/")

    run(["poetry", "build"])
    # Passed by env, not argv: arguments are visible to anything reading `ps`.
    run(["poetry", "publish"], env={"POETRY_PYPI_TOKEN_PYPI": token})


def publish_typescript(target: Target, version: str) -> None:
    require_tool("npm")
    token = first_env("NPM_TOKEN", "NODE_AUTH_TOKEN")
    if not token:
        fail("set NPM_TOKEN (or NODE_AUTH_TOKEN) to publish to npm.")

    cwd = REPO_ROOT / "typescript"
    command = ["npm", "publish"]
    otp = os.environ.get("NPM_OTP")
    if otp:
        command.append(f"--otp={otp}")

    # A throwaway npmrc keeps the token out of the user's real ~/.npmrc.
    with tempfile.NamedTemporaryFile("w", suffix=".npmrc", delete=False) as handle:
        handle.write(f"//registry.npmjs.org/:_authToken={token}\n")
        npmrc = handle.name
    try:
        # `prepublishOnly` runs the build, so a broken build fails the publish.
        run(command, cwd=cwd, env={"npm_config_userconfig": npmrc})
    finally:
        os.unlink(npmrc)


TARGETS: Dict[str, Target] = {
    "python": Target(
        name="python",
        package="deepeval",
        edits=[
            _version_edit("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
            _version_edit("deepeval/_version.py", r'__version__[^=]*=\s*"([^"]+)"'),
            _version_edit("CITATION.cff", r"^version:\s*(\S+)"),
            Edit(
                file="CITATION.cff",
                pattern=r'^date-released:\s*"([^"]+)"',
                value=lambda _: date.today().isoformat(),
                is_version=False,
            ),
        ],
        json_key="python",
        publish=publish_python,
        registry_versions=pypi_versions,
    ),
    "typescript": Target(
        name="typescript",
        package="deepeval",
        edits=[_version_edit("typescript/package.json", r'"version":\s*"([^"]+)"')],
        json_key="typescript",
        publish=publish_typescript,
        registry_versions=npm_versions,
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def say(message: str) -> None:
    print(f"[release] {message}")


def fail(message: str) -> None:
    print(f"[release] error: {message}", file=sys.stderr)
    raise SystemExit(1)


def first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        fail(f"{tool} is not on PATH.")


def run(
    command: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    say(f"$ {' '.join(command)}")
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(command, cwd=cwd or REPO_ROOT, env=merged)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def current_version(target: Target) -> str:
    """Every declaration must already agree, or the bump would paper over drift."""
    found = [
        (edit.file, read_version(edit)) for edit in target.edits if edit.is_version
    ]
    versions = {version for _, version in found}
    if len(versions) > 1:
        detail = ", ".join(f"{file} says {version}" for file, version in found)
        fail(f"{target.name} version declarations disagree — {detail}")
    return found[0][1]


def next_version(current: str, bump: str) -> str:
    """Raises ValueError rather than exiting: the prompt re-asks instead of dying."""
    if bump in {"major", "minor", "patch"}:
        core = re.match(r"^(\d+)\.(\d+)\.(\d+)", current)
        if not core:
            raise ValueError(
                f"cannot {bump}-bump a non-semver version ({current}) — "
                "type one explicitly."
            )
        major, minor, patch = (int(part) for part in core.groups())
        if bump == "major":
            return f"{major + 1}.0.0"
        if bump == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"

    if not SEMVER.match(bump):
        raise ValueError(f"'{bump}' is neither major/minor/patch nor a version number.")
    return bump


def prompt_version(target: Target, current: str) -> Optional[str]:
    """Ask what to release this SDK as. None means leave it alone."""
    try:
        default = next_version(current, "patch")
    except ValueError:
        default = None  # non-semver current: no bump to offer, so require an answer

    suggestion = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"  {target.name}: {current} ->{suggestion} ").strip()
        except EOFError:
            raise SystemExit(1)

        if not answer:
            if default:
                return default
            print(f"    {current} is not semver — type a version.")
            continue
        if answer.lower() in {"skip", "s"}:
            return None

        try:
            version = next_version(current, answer)
        except ValueError as error:
            print(f"    {error}")
            continue
        if version == current:
            print(f"    {target.name} is already at {version} — pick a higher one.")
            continue
        return version


def check_clean_tree() -> None:
    if git("status", "--porcelain"):
        fail(
            "working tree is dirty. The bump is the whole commit — "
            "commit or stash what you have first."
        )


def check_not_published(target: Target, version: str) -> None:
    published = target.registry_versions(target.package)
    if published is None:
        say("could not reach the registry; skipping the already-published check")
        return
    if version in published:
        fail(f"{target.package} {version} is already published — pick a higher version.")


def check_tag_free(target: Target, version: str) -> None:
    tag = f"{target.name}-v{version}"
    try:
        git("rev-parse", "-q", "--verify", f"refs/tags/{tag}")
    except subprocess.CalledProcessError:
        return
    fail(
        f"tag {tag} already exists. release.yml skips existing tags, so pushing "
        "this bump would draft no release."
    )


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bump both SDK versions everywhere and publish them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the bump and change nothing",
    )
    return parser.parse_args()


def release_one(target: Target, current: str, version: str) -> None:
    """Rewrite this SDK's version files and publish it, or leave neither done."""
    originals = {
        edit.file: (REPO_ROOT / edit.file).read_text() for edit in target.edits
    }
    for edit in target.edits:
        apply_edit(edit, version)
    write_json_version(target.json_key, version)

    try:
        target.publish(target, version)
    except SystemExit:
        # A failed publish leaves nothing shipped, so leave nothing bumped.
        for file, text in originals.items():
            (REPO_ROOT / file).write_text(text)
        write_json_version(target.json_key, current)
        say(f"{target.name} publish failed — reverted its version files to {current}.")
        raise


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        fail("release.py asks for both versions — run it from a terminal.")

    currents = {name: current_version(target) for name, target in TARGETS.items()}

    say("version to release (enter for the patch bump, or major/minor/patch/skip):")
    plan: Dict[str, str] = {}
    for name, target in TARGETS.items():
        chosen = prompt_version(target, currents[name])
        if chosen is not None:
            plan[name] = chosen
    print()

    if not plan:
        say("both skipped — nothing to release.")
        return

    files: List[str] = []
    for name, version in plan.items():
        target = TARGETS[name]
        say(f"{name}: {currents[name]} -> {version}")
        for file in sorted({edit.file for edit in target.edits}):
            say(f"  {file}")
            files.append(file)
        if read_json_version(target.json_key) is not None:
            say(f"  {SDK_VERSIONS_JSON}")
            files.append(SDK_VERSIONS_JSON)

    if args.dry_run:
        say("dry run — nothing written.")
        return

    # Check both before publishing either: a bad TypeScript version should not
    # surface only once Python is already on PyPI, which cannot be taken back.
    check_clean_tree()
    for name, version in plan.items():
        check_tag_free(TARGETS[name], version)
        check_not_published(TARGETS[name], version)

    done: List[str] = []
    for name, version in plan.items():
        try:
            release_one(TARGETS[name], currents[name], version)
        except SystemExit:
            if done:
                say(f"already published this run: {', '.join(done)} — leave them bumped.")
            raise
        done.append(f"{name} v{version}")

    summary = ", ".join(done)
    files = sorted(set(files))
    git("add", *files)
    git("commit", "-m", COMMIT_MESSAGE)
    say(f"committed {summary} (not pushed, not tagged)")

    print()
    say("next:")
    print("  git push origin main")
    tags = " and ".join(f"{name}-v{version}" for name, version in plan.items())
    print(f"  release.yml then tags {tags} and drafts the release — do not tag it yourself.")


if __name__ == "__main__":
    main()
