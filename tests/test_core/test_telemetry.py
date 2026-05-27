import os
import pytest
import shutil
import subprocess
import sys
import textwrap

import deepeval.telemetry as telemetry_mod

from pathlib import Path


def _no_hidden_store_dir(base: Path):
    deepeval_path = base / ".deepeval"
    shutil.rmtree(deepeval_path, ignore_errors=True)


def test_telemetry_writes_create_dir_when_missing(tmp_path, monkeypatch):
    _no_hidden_store_dir(tmp_path)

    os.path
    # Ensure opt-out is not set
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)

    # Run from a clean CWD with no .deepeval
    monkeypatch.chdir(tmp_path)
    assert not os.path.exists(".deepeval")

    uid = telemetry_mod.get_unique_id()
    assert isinstance(uid, str) and len(uid) > 0
    assert os.path.exists(".deepeval/.deepeval_telemetry.txt")


def _run_telemetry_import_script(
    script: str, tmp_path: Path, telemetry_opt_out: str
):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        {
            "CONFIDENT_OPEN_BROWSER": "0",
            "DEEPEVAL_DISABLE_DOTENV": "1",
            "DEEPEVAL_TELEMETRY_OPT_OUT": telemetry_opt_out,
            "ERROR_REPORTING": "0",
            "PYTHONPATH": os.pathsep.join(
                [str(repo_root), env.get("PYTHONPATH", "")]
            ),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_importing_telemetry_does_not_replace_global_tracer_provider(
    tmp_path,
):
    result = _run_telemetry_import_script(
        """
        import logging
        import warnings

        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        records = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        logger = logging.getLogger("opentelemetry")
        logger.addHandler(ListHandler())
        logger.setLevel(logging.WARNING)

        import deepeval
        import deepeval.telemetry as telemetry_mod

        internal_provider = getattr(
            telemetry_mod, "_deepeval_tracer_provider", None
        )
        assert trace.get_tracer_provider() is not internal_provider

        host_provider = TracerProvider()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            trace.set_tracer_provider(host_provider)

        messages = [str(w.message) for w in captured] + records
        assert not any(
            "Overriding of current TracerProvider is not allowed" in message
            for message in messages
        )
        assert trace.get_tracer_provider() is host_provider
        host_provider.shutdown()
        """,
        tmp_path,
        telemetry_opt_out="0",
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_telemetry_opt_out_does_not_create_tracer_provider(tmp_path):
    result = _run_telemetry_import_script(
        """
        from opentelemetry import trace

        import deepeval.telemetry as telemetry_mod

        assert getattr(telemetry_mod, "_deepeval_tracer_provider", None) is None
        assert getattr(telemetry_mod, "_deepeval_tracer", None) is None
        assert type(trace.get_tracer_provider()).__name__ == "ProxyTracerProvider"
        """,
        tmp_path,
        telemetry_opt_out="1",
    )

    assert result.returncode == 0, result.stdout + result.stderr
