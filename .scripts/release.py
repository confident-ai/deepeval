#!/usr/bin/env python3
"""Bump both SDK versions everywhere they are declared, and commit the bump.

    python .scripts/release.py
    python .scripts/release.py --dry-run

It asks for both versions, one prompt each, showing what that SDK is on now.
Press enter to take the next version that can actually be tagged — after the
highest existing `python-v*` / `typescript-v*` tag, skipping any number that
is already taken. Or type `major`/`minor`/`patch` or an explicit version like
`4.2.0`. Type `skip` to leave one SDK where it is.

The two SDKs are versioned independently: separate tags, separate registries,
separate release notes. A shared number would mean nothing to either audience,
so an SDK whose own files have not changed since its own last tag defaults to
`skip` rather than to a bump. Enter is safe on the SDK you did not touch.

Python components are single digits (`x.y.z`, each 0-9), so a patch bump after
`4.1.9` is `4.2.0` and after `4.9.9` it is `5.0.0`. TypeScript uses plain
semver instead — npm reads the major as a break signal and `0.x` as unstable,
so `0.9.12` goes to `0.9.13`, not `1.0.0`.

This does not publish. It rewrites the version files, commits them as
"new release", and prints the publish commands for you to run.

The version is declared in more places than the one you publish from. Python has
four (`pyproject.toml`, `deepeval/_version.py`, `CITATION.cff`, and the docs'
committed `sdk-versions.json`); TypeScript has two. `docs/scripts/
generate-sdk-versions.mjs` fails the docs build when the Python ones disagree,
so a partial bump breaks docs rather than shipping quietly.

This never creates a git tag. `.github/workflows/release.yml` tags `python-v*` /
`typescript-v*` when a version file lands on `main`, and it skips a tag that
already exists — tagging locally means the workflow finds nothing new and drafts
no release. Push, and let it do that half.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_VERSIONS_JSON = "docs/lib/generated/sdk-versions.json"
COMMIT_MESSAGE = "new release"
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-.]?[0-9A-Za-z.-]+)?$")
MAX_COMPONENT = 9  # each of x.y.z stays a single digit; overflow carries left


@dataclass(frozen=True)
class Edit:
    """One capture group in one file, rewritten in place."""

    file: str
    pattern: str
    value: Callable[[str], str]  # new version -> replacement text
    is_version: bool = (
        True  # False for fields that travel with a release, e.g. a date
    )


def _version_edit(file: str, pattern: str) -> Edit:
    return Edit(file=file, pattern=pattern, value=lambda version: version)


@dataclass(frozen=True)
class Target:
    name: str
    edits: List[Edit]
    json_key: str  # key in sdk-versions.json
    publish_commands: List[str]  # printed at the end, never run
    paths: List[str]  # what shipping this SDK covers, for "did it change?"
    single_digit: bool  # cap x.y.z at 9 and carry left, rather than plain semver


TARGETS: Dict[str, Target] = {
    "python": Target(
        name="python",
        edits=[
            _version_edit("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
            _version_edit(
                "deepeval/_version.py", r'__version__[^=]*=\s*"([^"]+)"'
            ),
            _version_edit("CITATION.cff", r"^version:\s*(\S+)"),
            Edit(
                file="CITATION.cff",
                pattern=r'^date-released:\s*"([^"]+)"',
                value=lambda _: date.today().isoformat(),
                is_version=False,
            ),
        ],
        json_key="python",
        publish_commands=["poetry build", "poetry publish"],
        paths=["deepeval", "pyproject.toml"],
        single_digit=True,
    ),
    "typescript": Target(
        name="typescript",
        edits=[
            _version_edit("typescript/package.json", r'"version":\s*"([^"]+)"')
        ],
        json_key="typescript",
        publish_commands=["cd typescript && npm publish"],
        paths=["typescript"],
        # npm consumers read the major as a break signal and 0.x as unstable,
        # so this one follows plain semver: 0.9.12 -> 0.9.13, not 1.0.0.
        single_digit=False,
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


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# file editing
# --------------------------------------------------------------------------- #


def read_version(edit: Edit) -> str:
    path = REPO_ROOT / edit.file
    if not path.exists():
        fail(f"{edit.file} is missing — run this from a full checkout.")
    match = re.search(edit.pattern, path.read_text(), re.MULTILINE)
    if not match:
        fail(
            f"could not find a version in {edit.file} (pattern: {edit.pattern})"
        )
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
# versions
# --------------------------------------------------------------------------- #


def current_version(target: Target) -> str:
    """Every declaration must already agree, or the bump would paper over drift."""
    found = [
        (edit.file, read_version(edit))
        for edit in target.edits
        if edit.is_version
    ]
    versions = {version for _, version in found}
    if len(versions) > 1:
        detail = ", ".join(f"{file} says {version}" for file, version in found)
        fail(f"{target.name} version declarations disagree — {detail}")
    return found[0][1]


def _core_parts(version: str) -> Optional[tuple]:
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        return None
    return tuple(int(part) for part in core.groups())


def _require_single_digit_version(version: str) -> None:
    """New releases are x.y.z with each component 0-9. Current may already be past that."""
    parts = _core_parts(version)
    if parts is None:
        return
    if any(part > MAX_COMPONENT for part in parts):
        raise ValueError(
            f"'{version}' has a double-digit component — "
            "releases are [0-9].[0-9].[0-9] (e.g. 4.2.0, not 4.1.10)."
        )


def _join(major: int, minor: int, patch: int, single_digit: bool) -> str:
    """Under single_digit, roll patch>9 into minor and minor>9 into major."""
    if not single_digit:
        return f"{major}.{minor}.{patch}"
    if patch > MAX_COMPONENT:
        patch = 0
        minor += 1
    if minor > MAX_COMPONENT:
        minor = 0
        major += 1
    if major > MAX_COMPONENT:
        raise ValueError("cannot bump past 9.9.9 — type a version explicitly.")
    return f"{major}.{minor}.{patch}"


def next_version(current: str, bump: str, single_digit: bool) -> str:
    """Raises ValueError rather than exiting: the prompt re-asks instead of dying."""
    if bump in {"major", "minor", "patch"}:
        parts = _core_parts(current)
        if not parts:
            raise ValueError(
                f"cannot {bump}-bump a non-semver version ({current}) — "
                "type one explicitly."
            )
        major, minor, patch = parts
        if bump == "major":
            return _join(major + 1, 0, 0, single_digit)
        if bump == "minor":
            return _join(major, minor + 1, 0, single_digit)
        return _join(major, minor, patch + 1, single_digit)

    if not SEMVER.match(bump):
        raise ValueError(
            f"'{bump}' is neither major/minor/patch nor a version number."
        )
    if single_digit:
        _require_single_digit_version(bump)
    return bump


def _version_key(version: str) -> tuple:
    return _core_parts(version) or (-1, -1, -1)


def next_taggable(target: Target, current: str, bump: str) -> str:
    """Next bump that is not already a local or origin tag.

    Starts from whichever is higher: the version in the files, or the highest
    existing tag. Then keeps bumping until release.yml would have something new
    to create.
    """
    tagged = tag_versions(target)
    highest = highest_tag_version(target)
    start = current
    if highest and _version_key(highest) > _version_key(current):
        start = highest

    version = next_version(start, bump, target.single_digit)
    seen = {start}
    while version in tagged:
        if version in seen:
            raise ValueError(
                f"cannot find a free {target.name} tag after {start}"
            )
        seen.add(version)
        version = next_version(version, bump, target.single_digit)
    return version


def prompt_version(target: Target, current: str) -> Optional[str]:
    """Ask what to release this SDK as. None means leave it alone."""
    unchanged = not has_unreleased_changes(target)
    default: Optional[str] = None
    if not unchanged:
        try:
            default = next_taggable(target, current, "patch")
        except ValueError:
            # non-semver current: no bump to offer, so require an answer
            default = None

    if default:
        suggestion = f" [{default}]"
    elif unchanged:
        suggestion = " [skip]"
    else:
        suggestion = ""

    if unchanged:
        print(
            f"    nothing under {'/, '.join(target.paths)}/ changed since "
            f"{target.name}-v{highest_tag_version(target)}"
        )

    while True:
        try:
            answer = input(
                f"  {target.name}: {current} ->{suggestion} "
            ).strip()
        except EOFError:
            raise SystemExit(1)

        if not answer:
            if default:
                return default
            if unchanged:
                return None
            print(f"    {current} is not semver — type a version.")
            continue
        if answer.lower() in {"skip", "s"}:
            return None

        try:
            if answer.lower() in {"major", "minor", "patch"}:
                version = next_taggable(target, current, answer.lower())
            else:
                version = next_version(current, answer, target.single_digit)
        except ValueError as error:
            print(f"    {error}")
            continue
        if version == current:
            print(
                f"    {target.name} is already at {version} — pick a higher one."
            )
            continue
        existing = existing_tag(target, version)
        if existing:
            print(
                f"    tag {existing} already exists — that version has been released."
            )
            continue
        return version


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_clean_tree() -> None:
    if git("status", "--porcelain"):
        fail(
            "working tree is dirty. The bump is the whole commit — "
            "commit or stash what you have first."
        )


_TAG_VERSIONS: Optional[Dict[str, set]] = None


def _load_tag_versions() -> Dict[str, set]:
    """Every python-v* / typescript-v* version, local and on origin, in one pass."""
    by_target: Dict[str, set] = {name: set() for name in TARGETS}
    for name in TARGETS:
        prefix = f"{name}-v"
        try:
            for line in git("tag", "-l", f"{prefix}*").splitlines():
                if line.startswith(prefix):
                    by_target[name].add(line[len(prefix) :])
        except subprocess.CalledProcessError:
            pass
    try:
        for line in git("ls-remote", "--tags", "origin").splitlines():
            if not line.strip():
                continue
            ref = line.split()[-1]
            if ref.endswith("^{}"):
                continue
            tag = ref.rsplit("/", 1)[-1]
            for name in TARGETS:
                prefix = f"{name}-v"
                if tag.startswith(prefix):
                    by_target[name].add(tag[len(prefix) :])
    except subprocess.CalledProcessError:
        pass
    return by_target


def tag_versions(target: Target) -> set:
    global _TAG_VERSIONS
    if _TAG_VERSIONS is None:
        _TAG_VERSIONS = _load_tag_versions()
    return _TAG_VERSIONS[target.name]


def existing_tag(target: Target, version: str) -> Optional[str]:
    """A tag that already exists locally or on origin still blocks a re-release."""
    if version in tag_versions(target):
        return f"{target.name}-v{version}"
    return None


def highest_tag_version(target: Target) -> Optional[str]:
    cores = [
        version for version in tag_versions(target) if _core_parts(version)
    ]
    return max(cores, key=_version_key) if cores else None


def has_unreleased_changes(target: Target) -> bool:
    """Did this SDK's own files move since its own last tag?

    The two SDKs ship to different registries off different tags, so releasing
    one because the other changed publishes a version with nothing in it. When
    the answer can't be determined (no tag yet, or the tag is only on origin),
    say yes and let the prompt decide.
    """
    version = highest_tag_version(target)
    if version is None:
        return True
    try:
        changed = git(
            "diff",
            "--name-only",
            f"{target.name}-v{version}..HEAD",
            "--",
            *target.paths,
        )
    except subprocess.CalledProcessError:
        return True
    return bool(changed.strip())


def check_tag_free(target: Target, version: str) -> None:
    tag = existing_tag(target, version)
    if tag is None:
        return
    fail(
        f"tag {tag} already exists — that version has been released. "
        "release.yml skips existing tags, so pushing this would draft no release."
    )


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bump both SDK versions everywhere and commit the bump.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the bump and change nothing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        fail("release.py asks for both versions — run it from a terminal.")

    if not args.dry_run:
        check_clean_tree()
    currents = {
        name: current_version(target) for name, target in TARGETS.items()
    }

    say(
        "version to release (enter for the next free tag, or "
        "major/minor/patch/skip):"
    )
    plan: Dict[str, str] = {}
    for name, target in TARGETS.items():
        chosen = prompt_version(target, currents[name])
        if chosen is not None:
            plan[name] = chosen
    print()

    if not plan:
        say("both skipped — nothing to bump.")
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

    for name, version in plan.items():
        check_tag_free(TARGETS[name], version)

    for name, version in plan.items():
        for edit in TARGETS[name].edits:
            apply_edit(edit, version)
        write_json_version(TARGETS[name].json_key, version)

    git("add", *sorted(set(files)))
    git("commit", "-m", COMMIT_MESSAGE)
    summary = ", ".join(f"{name} v{version}" for name, version in plan.items())
    say(f"committed {summary} (not pushed, not tagged)")

    print()
    say("now publish:")
    print("  git push origin main")
    for name in plan:
        for command in TARGETS[name].publish_commands:
            print(f"  {command}")
    tags = " and ".join(f"{name}-v{version}" for name, version in plan.items())
    print(f"\n  the push makes release.yml tag {tags} and draft the release.")


if __name__ == "__main__":
    main()
