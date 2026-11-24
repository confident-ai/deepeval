"""`deepeval link-ci`: open a pull request that runs `deepeval test run` with
GitHub Actions on every pull request, so each run is linked to the repository
on Confident AI's CI/CD page."""

import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import typer
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table

from deepeval.cli.auth.flow import CONFIDENT_PURPLE, prompt_confirm
from deepeval.cli.link_ci.git import (
    GitHubRepository,
    LinkCiError,
    exists_on_ref,
    find_authenticated_gh,
    find_default_branch,
    find_github_repository,
    find_repository_root,
    find_workflows_on_ref_containing,
    remote_branch_exists,
    run_gh,
    run_git,
)
from deepeval.cli.link_ci.invocation import load_test_run_args
from deepeval.cli.link_ci.workflow import (
    WORKFLOW_PATH,
    WorkflowConfig,
    build_workflow,
    detect_package_manager,
    detect_python_version,
    requirements_mention_deepeval,
)
from deepeval.config.settings import get_settings
from deepeval.utils import is_in_ci_env

LINK_CI_HELP = (
    "Open a pull request that runs your last `deepeval test run` with GitHub "
    "Actions on every pull request and push to the default branch. Each run "
    "is linked to the repository on Confident AI's CI/CD page. Uses your own "
    "git and GitHub CLI (`gh`) login; no GitHub token is sent to Confident AI."
)

DEFAULT_BRANCH_NAME = "deepeval/link-ci-cd"
COMMIT_MESSAGE = "Run deepeval tests on every pull request"
CONFIDENT_API_KEY_SECRET = "CONFIDENT_API_KEY"
DEFAULT_JUDGE_SECRET = "OPENAI_API_KEY"
FORWARDED_SETTINGS = ("CONFIDENT_BASE_URL", "CONFIDENT_REGION")
TEST_RUN_COMMAND = "deepeval test run"

DONE = "[rgb(5,245,141)]✓[/rgb(5,245,141)]"
WARNING = "[bold yellow]⚠[/bold yellow]"
INDENT = "   "

console = Console(highlight=False, emoji=False)
error_console = Console(stderr=True, highlight=False, emoji=False)


@dataclass(frozen=True)
class ExistingWorkflow:
    path: str
    default_branch: str


@dataclass(frozen=True)
class LinkCiPlan:
    root: Path
    repository: GitHubRepository
    default_branch: str
    workflow: str
    test_run_args: List[str]
    test_files: List[str]
    secret_names: List[str]
    secret_values: Dict[str, str]


def _local_api_keys() -> Dict[str, str]:
    settings = get_settings()
    keys: Dict[str, str] = {}
    for name in type(settings).model_fields:
        if not name.endswith("_API_KEY"):
            continue
        value = getattr(settings, name, None)
        if value is None:
            continue
        secret = (
            value.get_secret_value()
            if hasattr(value, "get_secret_value")
            else str(value)
        )
        if secret:
            keys[name] = secret
    return keys


def _forwarded_env() -> Dict[str, str]:
    settings = get_settings()
    env: Dict[str, str] = {}
    for name in FORWARDED_SETTINGS:
        value = getattr(settings, name, None)
        if value:
            env[name] = str(value)
    return env


def _secret_names(local_keys: Dict[str, str]) -> List[str]:
    judge_keys = sorted(
        name for name in local_keys if name != CONFIDENT_API_KEY_SECRET
    )
    return [CONFIDENT_API_KEY_SECRET, *(judge_keys or [DEFAULT_JUDGE_SECRET])]


def _uncommitted_test_files(
    root: Path, cwd: Path, default_ref: str, test_path: str
) -> List[str]:
    absolute = (cwd / test_path).resolve()
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as e:
        raise LinkCiError(
            f"{test_path} is outside the repository at {root}."
        ) from e
    if exists_on_ref(root, default_ref, relative):
        return []
    if absolute.is_file():
        return [relative]
    listed = run_git(
        [
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            relative,
        ],
        root,
    )
    return [line for line in listed.splitlines() if line]


def _plan(
    cwd: Path, test_file_or_directory: Optional[str], branch: str
) -> Union[LinkCiPlan, ExistingWorkflow]:
    root = find_repository_root(cwd)
    if root is None:
        raise LinkCiError(
            "Run `deepeval link-ci` inside the git repository your tests live in."
        )
    repository = find_github_repository(root)
    if repository is None:
        raise LinkCiError(
            "The `origin` remote of this repository isn't on GitHub. "
            "`deepeval link-ci` only sets up GitHub Actions for now."
        )

    test_run_args = (
        [test_file_or_directory]
        if test_file_or_directory
        else load_test_run_args()
    )
    if not test_run_args:
        raise LinkCiError(
            "No test run found in this directory. Run "
            "`deepeval test run <path>` first, or pass the path: "
            "`deepeval link-ci <path>`."
        )
    test_path = test_run_args[0].split("::", 1)[0]
    if not (cwd / test_path).exists():
        raise LinkCiError(f"{test_path} doesn't exist in {cwd}.")

    default_branch = find_default_branch(root)
    run_git(["fetch", "origin", default_branch], root)
    default_ref = f"origin/{default_branch}"
    existing_workflows = find_workflows_on_ref_containing(
        root, default_ref, TEST_RUN_COMMAND
    )
    if exists_on_ref(root, default_ref, WORKFLOW_PATH):
        existing_workflows = [WORKFLOW_PATH]
    if existing_workflows:
        return ExistingWorkflow(
            path=existing_workflows[0], default_branch=default_branch
        )
    if remote_branch_exists(root, branch):
        raise LinkCiError(
            f"The branch {branch} already exists on origin. Merge or delete "
            "its pull request, or pass another name with --branch."
        )

    local_keys = _local_api_keys()
    secret_names = _secret_names(local_keys)
    working_directory = (
        ""
        if cwd.resolve() == root
        else cwd.resolve().relative_to(root).as_posix()
    )
    package_manager = detect_package_manager(cwd)
    workflow = build_workflow(
        WorkflowConfig(
            default_branch=default_branch,
            python_version=detect_python_version(cwd, root),
            package_manager=package_manager,
            test_run_args=test_run_args,
            secret_names=secret_names,
            working_directory=working_directory,
            env=_forwarded_env(),
            requirements_mention_deepeval=requirements_mention_deepeval(cwd),
        )
    )
    return LinkCiPlan(
        root=root,
        repository=repository,
        default_branch=default_branch,
        workflow=workflow,
        test_run_args=test_run_args,
        test_files=_uncommitted_test_files(root, cwd, default_ref, test_path),
        secret_names=secret_names,
        secret_values={
            name: local_keys[name]
            for name in secret_names
            if name in local_keys
        },
    )


def _confirm(message: str, yes: bool) -> bool:
    return yes or prompt_confirm(message)


def _join(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _link(url: str) -> str:
    return f"[link={url}]{escape(url)}[/link]"


def _test_run_command(plan: LinkCiPlan) -> str:
    return shlex.join(["deepeval", "test", "run", *plan.test_run_args])


def _status(message: str):
    return console.status(
        message, spinner="dots", spinner_style=CONFIDENT_PURPLE
    )


def _print_plan(plan: LinkCiPlan) -> None:
    default_branch = plan.default_branch
    console.print(
        "\n🔗 Setting up GitHub Actions for "
        f"[bold]{escape(plan.repository.full_name)}[/bold]\n"
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("Runs", f"[bold cyan]{escape(_test_run_command(plan))}[/]")
    grid.add_row(
        "On",
        f"every pull request into {escape(default_branch)} and every push "
        f"to {escape(default_branch)}",
    )
    grid.add_row("Workflow", escape(WORKFLOW_PATH))
    grid.add_row("Secrets", escape(", ".join(plan.secret_names)))
    console.print(Padding(grid, (0, 0, 0, len(INDENT)), expand=False))
    console.print()


def _confirm_test_files(plan: LinkCiPlan, yes: bool) -> List[str]:
    test_files = plan.test_files
    if not test_files:
        return []
    files = _join(test_files)
    one = len(test_files) == 1
    pronoun = "it" if one else "them"
    if _confirm(
        f"{files} {'is' if one else 'are'} not on {plan.default_branch} yet, "
        f"so CI can't run {pronoun}. Add {pronoun} to the pull request?",
        yes,
    ):
        return test_files
    console.print(
        f"{WARNING} Leaving {escape(files)} out. CI fails until "
        f"{pronoun} {'is' if one else 'are'} on "
        f"{escape(plan.default_branch)}.",
        soft_wrap=True,
    )
    return []


def _share_secrets(gh: Optional[str], plan: LinkCiPlan, yes: bool) -> List[str]:
    if gh is None:
        return list(plan.secret_names)
    missing = [
        name for name in plan.secret_names if name not in plan.secret_values
    ]
    if not plan.secret_values:
        return missing
    if not _confirm(
        f"Copy your local {_join(list(plan.secret_values))} to the "
        f"repository secrets of {plan.repository.full_name} with gh?",
        yes,
    ):
        return list(plan.secret_names)
    with _status("Setting repository secrets..."):
        failed = _set_secrets(gh, plan)
    shared = [name for name in plan.secret_values if name not in failed]
    if shared:
        console.print(
            f"{DONE} Set {escape(_join(shared))} on "
            f"{escape(plan.repository.full_name)}",
            soft_wrap=True,
        )
    return missing + failed


def _open_pull_request(gh: str, plan: LinkCiPlan, branch: str) -> Optional[str]:
    result = run_gh(
        gh,
        [
            "pr",
            "create",
            "--repo",
            f"{plan.repository.host}/{plan.repository.full_name}",
            "--base",
            plan.default_branch,
            "--head",
            branch,
            "--title",
            COMMIT_MESSAGE,
            "--body",
            _pull_request_body(plan),
        ],
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    return lines[-1] if lines else None


def _pull_request_label(url: str) -> str:
    number = url.rstrip("/").rsplit("/", 1)[-1]
    return f"pull request #{number}" if number.isdigit() else "pull request"


def _set_secrets(gh: str, plan: LinkCiPlan) -> List[str]:
    repository = f"{plan.repository.host}/{plan.repository.full_name}"
    failed = []
    for name, value in plan.secret_values.items():
        result = run_gh(
            gh, ["secret", "set", name, "--repo", repository], input=value
        )
        if result.returncode != 0:
            failed.append(name)
    return failed


def _push_branch(plan: LinkCiPlan, branch: str, test_files: List[str]) -> None:
    parent = Path(tempfile.mkdtemp(prefix="deepeval-link-ci-"))
    worktree = parent / "worktree"
    created = False
    try:
        run_git(
            [
                "worktree",
                "add",
                "-b",
                branch,
                str(worktree),
                f"origin/{plan.default_branch}",
            ],
            plan.root,
        )
        created = True
        workflow_file = worktree / WORKFLOW_PATH
        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        workflow_file.write_text(plan.workflow, encoding="utf-8")
        for relative in test_files:
            destination = worktree / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan.root / relative, destination)
        run_git(["add", "--", WORKFLOW_PATH, *test_files], worktree)
        run_git(["commit", "-m", COMMIT_MESSAGE], worktree)
        run_git(["push", "--set-upstream", "origin", branch], worktree)
    finally:
        if created:
            try:
                run_git(
                    ["worktree", "remove", "--force", str(worktree)], plan.root
                )
                run_git(["branch", "-D", branch], plan.root)
            except LinkCiError:
                pass
        shutil.rmtree(parent, ignore_errors=True)


def _pull_request_body(plan: LinkCiPlan) -> str:
    command = _test_run_command(plan)
    return "\n".join(
        [
            f"Runs `{command}` with GitHub Actions on every pull request and "
            f"on every push to `{plan.default_branch}`. A failing test fails "
            "the check, and each run is listed under this repository on "
            "Confident AI's CI/CD page.",
            "",
            "Before merging:",
            "- Make sure the tests call your app to produce `actual_output`. "
            "A hard-coded `actual_output` is scored the same way on every "
            "run, so it can't catch a regression.",
            "- Add any other keys your app needs as repository secrets and "
            "to the workflow's `env`.",
            "- Check that the install step installs your app's dependencies.",
            "",
            "Pull requests from forks don't get repository secrets, so their "
            "runs aren't uploaded to Confident AI.",
        ]
    )


def link_ci_command(
    test_file_or_directory: Optional[str] = typer.Argument(
        None,
        help="The tests to run in CI. Defaults to the tests of your last "
        "`deepeval test run` in this directory, with the same options.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Answer yes to every prompt: include uncommitted test files and "
        "set repository secrets with gh.",
    ),
    branch: str = typer.Option(
        DEFAULT_BRANCH_NAME,
        "--branch",
        help="The branch to open the pull request from.",
    ),
) -> None:
    cwd = Path.cwd()
    try:
        with _status("Checking your repository..."):
            plan = _plan(cwd, test_file_or_directory, branch)
        if isinstance(plan, ExistingWorkflow):
            console.print(
                f"{DONE} {escape(plan.path)} on "
                f"{escape(plan.default_branch)} already runs "
                f"[bold cyan]{TEST_RUN_COMMAND}[/bold cyan] in GitHub "
                "Actions, so there's nothing to set up.",
                soft_wrap=True,
            )
            return

        _print_plan(plan)
        test_files = _confirm_test_files(plan, yes)
        gh = find_authenticated_gh(plan.repository)
        missing_secrets = _share_secrets(gh, plan, yes)

        with _status(f"Pushing {escape(branch)}..."):
            _push_branch(plan, branch, test_files)
        console.print(
            f"{DONE} Pushed {escape(_join([WORKFLOW_PATH, *test_files]))} "
            f"to {escape(branch)}",
            soft_wrap=True,
        )

        pull_request_url = None
        if gh:
            with _status("Opening the pull request..."):
                pull_request_url = _open_pull_request(gh, plan, branch)
    except LinkCiError as e:
        error_console.print(f"❌ {escape(str(e))}", soft_wrap=True)
        raise typer.Exit(code=1)
    except typer.Abort:
        console.print("\nCancelled. Nothing was pushed.")
        raise typer.Exit(code=1)

    if pull_request_url:
        console.print(
            f"{DONE} Opened {_pull_request_label(pull_request_url)}",
            soft_wrap=True,
        )
    if missing_secrets:
        console.print(
            f"\n{WARNING} Add {escape(', '.join(missing_secrets))} as "
            "repository secrets before the first run:\n"
            f"{INDENT}{_link(plan.repository.secrets_url)}",
            soft_wrap=True,
        )
    next_step, url = (
        ("merge the pull request", pull_request_url)
        if pull_request_url
        else (
            "open the pull request and merge it",
            plan.repository.compare_url(plan.default_branch, branch),
        )
    )
    console.print(
        f"\n🎉 One step to go: {next_step}.\n"
        f"{INDENT}{_link(url)}\n"
        f"{INDENT}Then every pull request runs your tests, and each run "
        "shows on Confident AI's CI/CD page.",
        soft_wrap=True,
    )


def _is_interactive_terminal() -> bool:
    return sys.stdout.isatty() and sys.stdin.isatty()


def _working_tree_runs_deepeval_in_ci(root: Path) -> bool:
    if (root / WORKFLOW_PATH).exists():
        return True
    workflows = root / ".github" / "workflows"
    for path in [*workflows.glob("*.yml"), *workflows.glob("*.yaml")]:
        try:
            if TEST_RUN_COMMAND in path.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def print_link_ci_hint() -> None:
    """One line after a local `deepeval test run`, until a workflow in the
    repository runs `deepeval test run`. Silent in CI, in scripts, and
    outside a GitHub repository."""
    try:
        if not _is_interactive_terminal() or is_in_ci_env():
            return
        root = find_repository_root(Path.cwd())
        if root is None or _working_tree_runs_deepeval_in_ci(root):
            return
        if find_github_repository(root) is None:
            return
    except Exception:
        return
    console.print(
        "\n» Run [bold cyan]deepeval link-ci[/bold cyan] to run these tests "
        "on every pull request with GitHub Actions.",
        soft_wrap=True,
    )
