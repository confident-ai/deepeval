import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

import pytest

from deepeval_pytest_plugin import plugin


REPO_ROOT = Path(__file__).resolve().parents[3]
_PLAIN_PYTEST_ENV_KEYS = (
    "CONFIDENT_AI_RUN_TEST_NAME",
    "DEEPEVAL",
    "DEEPEVAL_DISABLE_DOTENV",
    "DEEPEVAL_TELEMETRY_OPT_OUT",
    "DEEPEVAL_UPDATE_WARNING_OPT_IN",
    "ERROR_REPORTING",
    "PYTEST_ADDOPTS",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)


class _SkippedReport:
    skipped = True
    nodeid = "test_sample.py::test_skipped"
    longreprtext = "Skipped: explicit reason"


class _TerminalReporter:
    def __init__(self):
        self.requested_report_names = []

    def getreports(self, name):
        self.requested_report_names.append(name)
        return [_SkippedReport()]


class _Parser:
    def __init__(self):
        self.options = []

    def addoption(self, *args, **kwargs):
        self.options.append((args, kwargs))


def _run_hookwrapper(generator):
    next(generator)
    with pytest.raises(StopIteration):
        next(generator)


def _plain_pytest_env() -> dict:
    env = os.environ.copy()
    for key in _PLAIN_PYTEST_ENV_KEYS:
        env.pop(key, None)
    return env


def test_package_metadata_uses_import_safe_pytest_plugin():
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    poetry_config = pyproject["tool"]["poetry"]

    assert {"include": "deepeval_pytest_plugin"} in poetry_config["packages"]
    assert (
        poetry_config["plugins"]["pytest11"]["deepeval"]
        == "deepeval_pytest_plugin.plugin"
    )


def test_legacy_plugin_module_reexports_constants():
    from deepeval.plugins import plugin as legacy_plugin

    assert legacy_plugin.PYTEST_RUN_TEST_NAME == plugin.PYTEST_RUN_TEST_NAME
    assert (
        legacy_plugin.PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME
        == plugin.PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME
    )
    assert legacy_plugin.pytest_configure is plugin.pytest_configure


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("disabled", False),
        ("unknown", False),
        ("1", True),
        ("true", True),
        (" YES ", True),
        ('"enabled"', True),
    ],
)
def test_is_running_deepeval_uses_env_bool_semantics(
    monkeypatch, value, expected
):
    if value is None:
        monkeypatch.delenv("DEEPEVAL", raising=False)
    else:
        monkeypatch.setenv("DEEPEVAL", value)

    assert plugin.get_is_running_deepeval() is expected


def test_import_safe_pytest_plugin_does_not_import_deepeval():
    env = _plain_pytest_env()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import deepeval_pytest_plugin.plugin\n"
                "assert 'deepeval' not in sys.modules\n"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_addoption_is_gated_to_deepeval_runs(monkeypatch):
    parser = _Parser()
    monkeypatch.delenv("DEEPEVAL", raising=False)

    plugin.pytest_addoption(parser)

    assert parser.options == []

    monkeypatch.setenv("DEEPEVAL", "1")
    plugin.pytest_addoption(parser)

    assert parser.options == [
        (
            ("--identifier",),
            {
                "action": "store",
                "default": None,
                "help": "Custom identifier for the test run",
            },
        )
    ]


def test_configure_clears_stale_name_outside_deepeval_run(monkeypatch):
    monkeypatch.delenv("DEEPEVAL", raising=False)
    monkeypatch.setenv(
        plugin.PYTEST_RUN_TEST_NAME, "stale-from-prior-deepeval-run"
    )

    plugin.pytest_configure(config=SimpleNamespace())

    assert os.environ.get(plugin.PYTEST_RUN_TEST_NAME) is None


def test_installed_pytest_entrypoint_autoloads_import_safe_plugin(tmp_path):
    installed_site = tmp_path / "site-packages"
    dist_info = installed_site / "deepeval-0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: deepeval\nVersion: 0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[pytest11]\ndeepeval=deepeval_pytest_plugin.plugin\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "installed-autoload-run"
    run_dir.mkdir()
    (run_dir / "conftest.py").write_text(
        "import os\n\n"
        "def pytest_configure(config):\n"
        "    assert os.environ.get('CONFIDENT_AI_RUN_TEST_NAME') is None\n",
        encoding="utf-8",
    )
    test_file = run_dir / "test_plain_autoload.py"
    test_file.write_text(
        "import importlib.metadata as metadata\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n\n"
        "def _entry_points(group):\n"
        "    entry_points = metadata.entry_points()\n"
        "    if hasattr(entry_points, 'select'):\n"
        "        return list(entry_points.select(group=group))\n"
        "    return list(entry_points.get(group, []))\n\n"
        "def test_plain_autoload():\n"
        "    values = [\n"
        "        (entry_point.name, entry_point.value)\n"
        "        for entry_point in _entry_points('pytest11')\n"
        "        if entry_point.name == 'deepeval'\n"
        "    ]\n"
        "    assert ('deepeval', 'deepeval_pytest_plugin.plugin') in values\n"
        "    assert 'deepeval' not in sys.modules\n"
        "    assert os.environ.get('DEEPEVAL_PLUGIN_SENTINEL') is None\n"
        "    assert os.environ.get('CONFIDENT_AI_RUN_TEST_NAME') is None\n"
        "    assert not Path('.deepeval').exists()\n",
        encoding="utf-8",
    )
    (run_dir / ".env").write_text(
        "DEEPEVAL_PLUGIN_SENTINEL=loaded\n", encoding="utf-8"
    )

    env = _plain_pytest_env()
    env["DEEPEVAL"] = "false"
    env["CONFIDENT_AI_RUN_TEST_NAME"] = "stale-from-prior-run"
    env["PYTHONPATH"] = os.pathsep.join([str(installed_site), str(REPO_ROOT)])

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q"],
        cwd=run_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Running teardown with pytest sessionfinish..." not in result.stdout
    assert "was skipped. Reason:" not in result.stdout


def test_explicit_plugin_load_blocks_installed_entrypoint_duplicate(tmp_path):
    installed_site = tmp_path / "site-packages"
    dist_info = installed_site / "deepeval-0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: deepeval\nVersion: 0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[pytest11]\ndeepeval=deepeval_pytest_plugin.plugin\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "explicit-plugin-run"
    run_dir.mkdir()
    test_file = run_dir / "test_plain.py"
    test_file.write_text(
        "def test_plain():\n    assert True\n", encoding="utf-8"
    )

    env = _plain_pytest_env()
    env["PYTHONPATH"] = os.pathsep.join([str(installed_site), str(REPO_ROOT)])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:deepeval",
            "-p",
            "deepeval_pytest_plugin.plugin",
            str(test_file),
            "-q",
        ],
        cwd=run_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Plugin already registered" not in output


def test_sessionstart_creates_test_run_during_deepeval_run(monkeypatch):
    monkeypatch.setenv("DEEPEVAL", "1")
    from deepeval.test_run import global_test_run_manager

    calls = []

    def create_test_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        global_test_run_manager, "save_to_disk", False, raising=False
    )
    monkeypatch.setattr(
        global_test_run_manager, "create_test_run", create_test_run
    )
    options = {
        "identifier": "custom-run",
        "file_or_dir": ["tests/evals/test_example.py"],
    }
    session = SimpleNamespace(
        config=SimpleNamespace(
            getoption=lambda name, default=None: options.get(name, default)
        )
    )

    plugin.pytest_sessionstart(session)

    assert global_test_run_manager.save_to_disk is True
    assert calls == [
        {
            "identifier": "custom-run",
            "file_name": "tests/evals/test_example.py",
        }
    ]


def test_sessionstart_clears_stale_name_outside_deepeval_run(monkeypatch):
    monkeypatch.delenv("DEEPEVAL", raising=False)
    monkeypatch.setenv(
        plugin.PYTEST_RUN_TEST_NAME, "stale-from-prior-deepeval-run"
    )

    plugin.pytest_sessionstart(session=SimpleNamespace())

    assert os.environ.get(plugin.PYTEST_RUN_TEST_NAME) is None


def test_runtest_protocol_clears_stale_name_outside_deepeval_run(monkeypatch):
    monkeypatch.delenv("DEEPEVAL", raising=False)
    monkeypatch.setenv(
        plugin.PYTEST_RUN_TEST_NAME, "stale-from-prior-deepeval-run"
    )

    result = plugin.pytest_runtest_protocol(
        item=SimpleNamespace(nodeid="test_plain.py::test_plain"),
        nextitem=None,
    )

    assert result is None
    assert os.environ.get(plugin.PYTEST_RUN_TEST_NAME) is None


def test_runtest_protocol_sets_name_during_deepeval_run(monkeypatch):
    monkeypatch.setenv("DEEPEVAL", "1")

    result = plugin.pytest_runtest_protocol(
        item=SimpleNamespace(nodeid="test_eval.py::TestEval::test_metric"),
        nextitem=None,
    )

    assert result is None
    assert os.environ[plugin.PYTEST_RUN_TEST_NAME] == "test_metric"


def test_runtest_call_restores_eval_session_when_test_errors(monkeypatch):
    monkeypatch.setenv("DEEPEVAL", "1")
    from deepeval.tracing.tracing import trace_manager
    from deepeval.tracing.types import EvalMode, EvalSession

    previous_session = EvalSession()
    trace_manager.eval_session = previous_session

    hook = plugin.pytest_runtest_call(SimpleNamespace())
    next(hook)
    assert trace_manager.eval_session.mode == EvalMode.EVALUATE

    with pytest.raises(RuntimeError):
        hook.throw(RuntimeError("test body failed"))

    assert trace_manager.eval_session is previous_session


def test_sessionfinish_is_quiet_outside_deepeval_run(monkeypatch, capsys):
    monkeypatch.delenv("DEEPEVAL", raising=False)

    _run_hookwrapper(
        plugin.pytest_sessionfinish(
            session=SimpleNamespace(),
            exitstatus=0,
        )
    )

    assert capsys.readouterr().out == ""


def test_sessionfinish_prints_during_deepeval_run(monkeypatch, capsys):
    monkeypatch.setenv("DEEPEVAL", "1")
    monkeypatch.setenv(plugin.PYTEST_RUN_TEST_NAME, "test_metric")

    _run_hookwrapper(
        plugin.pytest_sessionfinish(
            session=SimpleNamespace(),
            exitstatus=0,
        )
    )

    assert (
        capsys.readouterr().out
        == "Running teardown with pytest sessionfinish...\n"
    )
    assert os.environ.get(plugin.PYTEST_RUN_TEST_NAME) is None


def test_terminal_summary_is_quiet_outside_deepeval_run(monkeypatch, capsys):
    monkeypatch.delenv("DEEPEVAL", raising=False)
    reporter = _TerminalReporter()

    plugin.pytest_terminal_summary(
        terminalreporter=reporter,
        exitstatus=0,
        config=SimpleNamespace(),
    )

    assert reporter.requested_report_names == []
    assert capsys.readouterr().out == ""


def test_terminal_summary_prints_skips_during_deepeval_run(monkeypatch, capsys):
    monkeypatch.setenv("DEEPEVAL", "1")
    reporter = _TerminalReporter()

    plugin.pytest_terminal_summary(
        terminalreporter=reporter,
        exitstatus=0,
        config=SimpleNamespace(),
    )

    assert reporter.requested_report_names == ["skipped"]
    assert (
        capsys.readouterr().out
        == "Test test_sample.py::test_skipped was skipped. Reason: Skipped: explicit reason\n"
    )


def test_public_pytest_plugin_is_quiet_in_plain_pytest_run(tmp_path):
    run_dir = tmp_path / "plain-run"
    run_dir.mkdir()
    test_file = run_dir / "test_plain_pytest.py"
    test_file.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import pytest\n\n"
        "import sys\n\n"
        "def test_plain_skip():\n"
        "    assert 'deepeval' not in sys.modules\n"
        "    assert os.environ.get('DEEPEVAL_PLUGIN_SENTINEL') is None\n"
        "    assert os.environ.get('CONFIDENT_AI_RUN_TEST_NAME') is None\n"
        "    assert not Path('.deepeval').exists()\n"
        "    pytest.skip('plain pytest skip')\n",
        encoding="utf-8",
    )
    (run_dir / ".env").write_text(
        "DEEPEVAL_PLUGIN_SENTINEL=loaded\n", encoding="utf-8"
    )

    env = _plain_pytest_env()
    env["CONFIDENT_AI_RUN_TEST_NAME"] = "stale-from-prior-run"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "deepeval_pytest_plugin.plugin",
            str(test_file),
            "-q",
        ],
        cwd=run_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Running teardown with pytest sessionfinish..." not in result.stdout
    assert "was skipped. Reason:" not in result.stdout
