import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
from typer.testing import CliRunner

from deepeval.cli import link_ci
from deepeval.cli.link_ci import command as link_ci_command_module
from deepeval.cli.link_ci import git as link_ci_git
from deepeval.cli.link_ci.git import parse_github_remote
from deepeval.cli.link_ci.invocation import (
    load_test_run_args,
    save_test_run_args,
)
from deepeval.cli.link_ci.workflow import (
    WORKFLOW_PATH,
    PackageManager,
    WorkflowConfig,
    build_workflow,
    detect_package_manager,
)
from deepeval.cli.main import app
from deepeval.config.settings import Settings

GITHUB_URL = "https://github.com/acme/support-bot.git"
BRANCH = "deepeval/link-ci"

runner = CliRunner()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(autouse=True)
def _never_call_the_real_gh(monkeypatch):
    monkeypatch.setattr(link_ci_git, "find_gh", lambda: None)
    monkeypatch.setattr(
        link_ci_command_module, "find_authenticated_gh", lambda _: None
    )


@pytest.fixture(autouse=True)
def _no_local_api_keys(monkeypatch):
    for name in Settings.model_fields:
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git_config = tmp_path / "gitconfig"
    git_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-b", "main")
    _git(work, "config", "user.name", "Test")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "commit.gpgsign", "false")
    _git(work, "remote", "add", "origin", GITHUB_URL)
    _git(work, "config", f"url.{remote}.insteadOf", GITHUB_URL)
    (work / "README.md").write_text("support bot\n")
    _git(work, "add", "README.md")
    _git(work, "commit", "-m", "init")
    _git(work, "push", "-u", "origin", "main")
    monkeypatch.chdir(work)
    return SimpleNamespace(work=work, remote=remote)


def _remote_file(repo, ref: str, path: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "--git-dir", str(repo.remote), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _write_test_file(repo) -> None:
    (repo.work / "tests").mkdir()
    (repo.work / "tests" / "test_example.py").write_text("def test_x(): pass\n")


class FakeGh:
    def __init__(self):
        self.calls: List[SimpleNamespace] = []

    def __call__(self, gh, args, input=None):
        self.calls.append(SimpleNamespace(args=args, input=input))
        stdout = (
            "https://github.com/acme/support-bot/pull/1\n"
            if args[:2] == ["pr", "create"]
            else ""
        )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


def test_pushes_a_branch_with_the_workflow_and_the_uncommitted_test_file_and_leaves_the_working_tree_alone(
    repo,
):
    _write_test_file(repo)
    save_test_run_args(["tests/test_example.py", "-n", "2"])
    status_before = _git(repo.work, "status", "--porcelain")

    result = runner.invoke(app, ["link-ci", "--yes"])

    assert result.exit_code == 0, result.output
    workflow = _remote_file(repo, BRANCH, WORKFLOW_PATH)
    assert "deepeval test run tests/test_example.py -n 2" in workflow
    assert 'branches: ["main"]' in workflow
    assert "CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}" in workflow
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in workflow
    assert _remote_file(repo, BRANCH, "tests/test_example.py") is not None
    assert _remote_file(repo, "main", WORKFLOW_PATH) is None

    assert _git(repo.work, "branch", "--show-current") == "main"
    assert _git(repo.work, "status", "--porcelain") == status_before
    assert _git(repo.work, "branch", "--list", BRANCH) == ""
    assert len(_git(repo.work, "worktree", "list").splitlines()) == 1

    assert (
        "https://github.com/acme/support-bot/compare/main...deepeval/link-ci?expand=1"
        in result.output
    )
    assert "Add CONFIDENT_API_KEY, OPENAI_API_KEY as repository secrets" in (
        result.output
    )


def test_with_gh_sets_secrets_from_local_keys_before_opening_the_pull_request(
    repo, monkeypatch
):
    _write_test_file(repo)
    _git(repo.work, "add", "tests")
    _git(repo.work, "commit", "-m", "tests")
    _git(repo.work, "push")
    monkeypatch.setenv("CONFIDENT_API_KEY", "confident-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    fake_gh = FakeGh()
    monkeypatch.setattr(
        link_ci_command_module, "find_authenticated_gh", lambda _: "gh"
    )
    monkeypatch.setattr(link_ci_command_module, "run_gh", fake_gh)

    result = runner.invoke(app, ["link-ci", "tests/test_example.py", "--yes"])

    assert result.exit_code == 0, result.output
    assert [call.args[:3] for call in fake_gh.calls] == [
        ["secret", "set", "CONFIDENT_API_KEY"],
        ["secret", "set", "ANTHROPIC_API_KEY"],
        ["pr", "create", "--repo"],
    ]
    secrets = {call.args[2]: call.input for call in fake_gh.calls[:2]}
    assert secrets == {
        "ANTHROPIC_API_KEY": "anthropic-key",
        "CONFIDENT_API_KEY": "confident-key",
    }
    assert all(
        "anthropic-key" not in arg for c in fake_gh.calls for arg in c.args
    )
    pr_args = fake_gh.calls[2].args
    assert pr_args[pr_args.index("--base") + 1] == "main"
    assert pr_args[pr_args.index("--head") + 1] == BRANCH
    assert "Opened https://github.com/acme/support-bot/pull/1" in result.output
    assert "repository secrets" not in result.output
    workflow = _remote_file(repo, BRANCH, WORKFLOW_PATH)
    assert "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" in workflow
    assert "OPENAI_API_KEY" not in workflow


def test_exits_early_when_the_workflow_is_already_on_the_default_branch(repo):
    _write_test_file(repo)
    (repo.work / ".github" / "workflows").mkdir(parents=True)
    (repo.work / WORKFLOW_PATH).write_text("name: DeepEval\n")
    _git(repo.work, "add", ".github")
    _git(repo.work, "commit", "-m", "ci")
    _git(repo.work, "push")

    result = runner.invoke(app, ["link-ci", "tests/test_example.py", "--yes"])

    assert result.exit_code == 0, result.output
    assert f"{WORKFLOW_PATH} on main already runs" in result.output
    assert _remote_file(repo, BRANCH, WORKFLOW_PATH) is None


def test_exits_early_when_another_workflow_already_runs_deepeval_tests(repo):
    _write_test_file(repo)
    (repo.work / ".github" / "workflows").mkdir(parents=True)
    (repo.work / ".github" / "workflows" / "ci.yaml").write_text(
        "jobs:\n  test:\n    steps:\n      - run: deepeval test run tests\n"
    )
    _git(repo.work, "add", ".github")
    _git(repo.work, "commit", "-m", "ci")
    _git(repo.work, "push")

    result = runner.invoke(app, ["link-ci", "tests/test_example.py", "--yes"])

    assert result.exit_code == 0, result.output
    assert ".github/workflows/ci.yaml on main already runs" in result.output
    assert _remote_file(repo, BRANCH, WORKFLOW_PATH) is None


def test_refuses_to_reuse_a_branch_that_already_exists_on_origin(repo):
    _write_test_file(repo)
    _git(repo.work, "push", "origin", f"main:refs/heads/{BRANCH}")

    result = runner.invoke(app, ["link-ci", "tests/test_example.py", "--yes"])

    assert result.exit_code == 1
    assert f"The branch {BRANCH} already exists on origin" in result.output


def test_fails_outside_a_github_repository(repo):
    _git(repo.work, "remote", "set-url", "origin", "https://gitlab.com/a/b.git")
    _write_test_file(repo)

    result = runner.invoke(app, ["link-ci", "tests/test_example.py"])

    assert result.exit_code == 1
    assert "isn't on GitHub" in result.output


def test_fails_when_no_test_run_was_recorded_and_no_path_is_given(repo):
    result = runner.invoke(app, ["link-ci"])

    assert result.exit_code == 1
    assert "Run `deepeval test run <path>` first" in result.output


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "git@github.com:acme/support-bot.git",
            ("github.com", "acme", "support-bot"),
        ),
        (
            "https://github.com/acme/support-bot",
            ("github.com", "acme", "support-bot"),
        ),
        (
            "https://token@github.com/acme/support-bot.git",
            ("github.com", "acme", "support-bot"),
        ),
        (
            "ssh://git@github.example.com:22/acme/support-bot.git",
            ("github.example.com", "acme", "support-bot"),
        ),
        ("https://gitlab.com/acme/support-bot.git", None),
        ("https://github.com/acme", None),
    ],
)
def test_parse_github_remote(url, expected):
    repository = parse_github_remote(url)
    assert (
        (repository.host, repository.owner, repository.name)
        if repository
        else None
    ) == expected


def _config(**overrides) -> WorkflowConfig:
    values = dict(
        default_branch="main",
        python_version="3.12",
        package_manager=PackageManager.PIP,
        test_run_args=["tests/test_example.py"],
        secret_names=["CONFIDENT_API_KEY", "OPENAI_API_KEY"],
    )
    values.update(overrides)
    return WorkflowConfig(**values)


def test_workflow_is_valid_yaml_with_quoted_values():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        build_workflow(
            _config(
                working_directory="services/bot",
                env={"CONFIDENT_BASE_URL": "https://confident.internal:8443"},
                test_run_args=[
                    "tests/test_a.py::test_x",
                    "-m",
                    "not slow: yet",
                ],
            )
        )
    )

    job = workflow["jobs"]["deepeval"]
    assert workflow[True]["push"]["branches"] == ["main"]
    assert job["defaults"]["run"]["working-directory"] == "services/bot"
    assert job["env"]["CONFIDENT_BASE_URL"] == "https://confident.internal:8443"
    assert job["steps"][-1]["run"].strip() == (
        "deepeval test run tests/test_a.py::test_x -m 'not slow: yet'"
    )


@pytest.mark.parametrize(
    "lockfile, install, run_prefix",
    [
        ("uv.lock", "uv sync", "uv run deepeval"),
        ("poetry.lock", "poetry install", "poetry run deepeval"),
        (
            "requirements.txt",
            "pip install -r requirements.txt deepeval",
            "deepeval",
        ),
        (None, "pip install deepeval", "deepeval"),
    ],
)
def test_install_step_follows_the_lockfile(
    tmp_path, lockfile, install, run_prefix
):
    if lockfile:
        (tmp_path / lockfile).write_text("")
    workflow = build_workflow(
        _config(package_manager=detect_package_manager(tmp_path))
    )

    assert install in workflow
    assert f"{run_prefix} test run tests/test_example.py" in workflow


def test_requirements_that_list_deepeval_are_not_extended():
    workflow = build_workflow(
        _config(
            package_manager=PackageManager.PIP_REQUIREMENTS,
            requirements_mention_deepeval=True,
        )
    )

    assert "run: pip install -r requirements.txt\n" in workflow


def test_saved_test_run_args_round_trip():
    assert load_test_run_args() is None
    save_test_run_args(["tests", "-m", "smoke"])
    assert load_test_run_args() == ["tests", "-m", "smoke"]


@pytest.fixture
def interactive_terminal(monkeypatch):
    monkeypatch.setattr(
        link_ci_command_module, "_is_interactive_terminal", lambda: True
    )
    monkeypatch.setattr(link_ci_command_module, "is_in_ci_env", lambda: False)


def test_hint_is_silent_when_output_is_not_a_terminal(
    repo, capsys, monkeypatch
):
    monkeypatch.setattr(link_ci_command_module, "is_in_ci_env", lambda: False)

    link_ci.print_link_ci_hint()

    assert capsys.readouterr().out == ""


def test_hint_shows_in_a_github_repository_without_the_workflow(
    repo, interactive_terminal, capsys
):
    link_ci.print_link_ci_hint()

    assert "Run `deepeval link-ci`" in capsys.readouterr().out


def test_hint_is_silent_once_the_workflow_exists(
    repo, interactive_terminal, capsys
):
    (repo.work / ".github" / "workflows").mkdir(parents=True)
    (repo.work / WORKFLOW_PATH).write_text("name: DeepEval\n")

    link_ci.print_link_ci_hint()

    assert capsys.readouterr().out == ""


def test_hint_is_silent_when_another_workflow_runs_deepeval_tests(
    repo, interactive_terminal, capsys
):
    (repo.work / ".github" / "workflows").mkdir(parents=True)
    (repo.work / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  test:\n    steps:\n      - run: deepeval test run tests\n"
    )

    link_ci.print_link_ci_hint()

    assert capsys.readouterr().out == ""


def test_hint_is_silent_in_ci(repo, interactive_terminal, capsys, monkeypatch):
    monkeypatch.setattr(link_ci_command_module, "is_in_ci_env", lambda: True)

    link_ci.print_link_ci_hint()

    assert capsys.readouterr().out == ""


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        return self.result


@pytest.fixture
def test_run_command(monkeypatch, tmp_path):
    from deepeval.cli.test import command as test_command_module

    (tmp_path / "test_example.py").write_text("def test_x(): pass\n")
    recorders = SimpleNamespace(
        link=_Recorder(),
        hint=_Recorder(),
        wrap_up=_Recorder(("https://app.confident-ai.com/run", "test-run-1")),
    )
    monkeypatch.setattr(test_command_module.pytest, "main", lambda args: 1)
    monkeypatch.setattr(
        test_command_module.global_test_run_manager,
        "wrap_up_test_run",
        recorders.wrap_up,
    )
    monkeypatch.setattr(
        test_command_module,
        "link_test_run_to_github_repository",
        recorders.link,
    )
    monkeypatch.setattr(
        test_command_module, "print_link_ci_hint", recorders.hint
    )
    return SimpleNamespace(module=test_command_module, **vars(recorders))


def test_test_run_links_the_uploaded_run_with_the_pytest_exit_code_and_remembers_its_args(
    test_run_command, monkeypatch
):
    monkeypatch.setattr(test_run_command.module, "is_in_ci_env", lambda: False)

    result = runner.invoke(
        app,
        [
            "test",
            "run",
            "test_example.py",
            "--identifier",
            "nightly",
            "-m",
            "smoke",
        ],
    )

    assert result.exit_code == 1, result.output
    assert test_run_command.link.calls == [("test-run-1", 1)]
    assert load_test_run_args() == [
        "test_example.py",
        "--identifier",
        "nightly",
        "-m",
        "smoke",
    ]
    assert len(test_run_command.hint.calls) == 1


def test_test_run_in_ci_neither_remembers_its_args_nor_prints_the_hint(
    test_run_command, monkeypatch
):
    monkeypatch.setattr(test_run_command.module, "is_in_ci_env", lambda: True)

    runner.invoke(app, ["test", "run", "test_example.py"])

    assert test_run_command.link.calls == [("test-run-1", 1)]
    assert load_test_run_args() is None
    assert test_run_command.hint.calls == []


def test_test_run_without_an_upload_does_not_link(
    test_run_command, monkeypatch
):
    test_run_command.wrap_up.result = None
    monkeypatch.setattr(test_run_command.module, "is_in_ci_env", lambda: True)

    runner.invoke(app, ["test", "run", "test_example.py"])

    assert test_run_command.link.calls == []
