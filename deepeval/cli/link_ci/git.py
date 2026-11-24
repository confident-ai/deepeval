"""The git and GitHub CLI calls `deepeval link-ci` makes. Every repository
change runs with the user's own git and `gh` credentials; nothing here sends a
GitHub token anywhere."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class LinkCiError(Exception):
    pass


@dataclass(frozen=True)
class GitHubRepository:
    host: str
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    def compare_url(self, base: str, head: str) -> str:
        return (
            f"https://{self.host}/{self.full_name}/compare/"
            f"{base}...{head}?expand=1"
        )

    @property
    def secrets_url(self) -> str:
        return f"https://{self.host}/{self.full_name}/settings/secrets/actions"


_REMOTE_PATTERNS = [
    re.compile(r"^git@(?P<host>[^:]+):(?P<path>.+)$"),
    re.compile(
        r"^(?:https?|ssh|git)://(?:[^@/]+@)?(?P<host>[^/:]+)(?::\d+)?/(?P<path>.+)$"
    ),
]


def parse_github_remote(url: str) -> Optional[GitHubRepository]:
    for pattern in _REMOTE_PATTERNS:
        match = pattern.match(url.strip())
        if not match:
            continue
        host = match.group("host").lower()
        path = match.group("path").rstrip("/")
        if path.endswith(".git"):
            path = path[: -len(".git")]
        parts = path.split("/")
        if "github" not in host or len(parts) != 2 or not all(parts):
            return None
        return GitHubRepository(host=host, owner=parts[0], name=parts[1])
    return None


def run_git(args: List[str], cwd: Path, input: Optional[str] = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            input=input,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise LinkCiError("git isn't installed or isn't on your PATH.") from e
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise LinkCiError(f"`git {' '.join(args)}` failed: {detail}")
    return result.stdout.strip()


def find_repository_root(cwd: Path) -> Optional[Path]:
    try:
        return Path(run_git(["rev-parse", "--show-toplevel"], cwd)).resolve()
    except LinkCiError:
        return None


def find_github_repository(root: Path) -> Optional[GitHubRepository]:
    # The configured URL, not `git remote get-url`, which expands insteadOf.
    try:
        url = run_git(["config", "--get", "remote.origin.url"], root)
    except LinkCiError:
        return None
    return parse_github_remote(url)


def find_default_branch(root: Path) -> str:
    try:
        ref = run_git(
            ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], root
        )
        if ref.startswith("origin/"):
            return ref[len("origin/") :]
    except LinkCiError:
        pass
    output = run_git(["ls-remote", "--symref", "origin", "HEAD"], root)
    for line in output.splitlines():
        if line.startswith("ref: refs/heads/") and line.endswith("\tHEAD"):
            return line[len("ref: refs/heads/") : -len("\tHEAD")]
    raise LinkCiError("Couldn't find the default branch of the origin remote.")


def exists_on_ref(root: Path, ref: str, path: str) -> bool:
    try:
        run_git(["cat-file", "-e", f"{ref}:{path}"], root)
    except LinkCiError:
        return False
    return True


def find_workflows_on_ref_containing(
    root: Path, ref: str, text: str
) -> List[str]:
    try:
        names = run_git(
            ["ls-tree", "--name-only", f"{ref}:.github/workflows"], root
        )
    except LinkCiError:
        return []
    matches = []
    for name in names.splitlines():
        if not name.endswith((".yml", ".yaml")):
            continue
        path = f".github/workflows/{name}"
        try:
            content = run_git(["show", f"{ref}:{path}"], root)
        except LinkCiError:
            continue
        if text in content:
            matches.append(path)
    return matches


def remote_branch_exists(root: Path, branch: str) -> bool:
    return bool(
        run_git(
            ["ls-remote", "--heads", "origin", f"refs/heads/{branch}"], root
        )
    )


def find_gh() -> Optional[str]:
    return shutil.which("gh")


def run_gh(
    gh: str, args: List[str], input: Optional[str] = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [gh, *args],
        input=input,
        capture_output=True,
        text=True,
        check=False,
    )


def find_authenticated_gh(repository: GitHubRepository) -> Optional[str]:
    gh = find_gh()
    if gh is None:
        return None
    status = run_gh(gh, ["auth", "status", "--hostname", repository.host])
    return gh if status.returncode == 0 else None
