import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

OTEL_ROOTS = ("opentelemetry", "grpcio")
ALLOWED_UNGUARDED = {"deepeval/tracing/otel/test_exporter.py"}


def _extract_section(text: str, header: str):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == header:
            section = []
            for line in lines[i + 1 :]:
                if line.startswith("["):
                    break
                section.append(line)
            return "\n".join(section)
    return None


def test_main_dependencies_exclude_opentelemetry_and_grpcio():
    text = PYPROJECT_PATH.read_text(encoding="utf-8")

    deps = _extract_section(text, "[tool.poetry.dependencies]")
    assert deps is not None, "[tool.poetry.dependencies] section missing"
    for line in deps.splitlines():
        if "opentelemetry" in line or "grpcio" in line:
            assert "optional = true" in line, (
                f"otel/grpcio deps must be declared optional: "
                f"{line.strip()}"
            )
    assert (
        "grpcio" not in text
    ), "grpcio must not appear anywhere in pyproject.toml"

    extras = _extract_section(text, "[tool.poetry.extras]")
    assert extras is not None, "[tool.poetry.extras] section missing"

    otel_lines = []
    in_otel = False
    for line in extras.splitlines():
        if re.match(r"^otel\s*=", line):
            in_otel = True
            otel_lines.append(line)
        elif in_otel:
            if re.match(r"^\w+\s*=", line):
                break
            otel_lines.append(line)
    assert in_otel, "extras must declare an `otel` entry"
    otel_block = "\n".join(otel_lines)
    for pkg in (
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
    ):
        assert pkg in otel_block, f"otel extra must list {pkg}"


_CHILD_SCRIPT = r"""
import sys

EXPECTED_ROOT = sys.argv[1]

failures = []


def check(label, fn):
    try:
        fn()
    except Exception as exc:
        failures.append(label)
        print(f"[FAIL] {label} :: {type(exc).__name__}: {exc}")
    else:
        print(f"[PASS] {label}")


def check_worktree_deepeval():
    # Run first: pulling in deepeval may itself cache opentelemetry*
    # submodules in sys.modules, which would silently bypass any
    # meta_path blocker installed beforehand.
    import deepeval

    assert deepeval.__file__.startswith(EXPECTED_ROOT), (
        f"expected worktree deepeval under {EXPECTED_ROOT}, "
        f"got {deepeval.__file__}"
    )


check("worktree deepeval loaded", check_worktree_deepeval)

for k in list(sys.modules):
    root = k.split(".")[0]
    if root in ("opentelemetry", "grpcio", "grpc"):
        del sys.modules[k]

import importlib.abc


class _OtelBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("opentelemetry", "grpcio", "grpc"):
            raise ModuleNotFoundError(
                f"import of {fullname!r} blocked by test", name=fullname
            )
        return None


# Insert at the front: an appended finder runs after PathFinder, which
# would still resolve a disk-installed opentelemetry/grpcio.
sys.meta_path.insert(0, _OtelBlocker())


def is_actionable_otel_import_error(exc):
    msg = str(exc).lower()
    return isinstance(exc, ImportError) and (
        "deepeval[otel]" in msg or "opentelemetry" in msg
    )


def check_grpc_blocked():
    try:
        import grpc
    except ModuleNotFoundError:
        return
    raise AssertionError("grpc import was not blocked")


def make_exporter_raises_actionable(exporter):
    try:
        exporter.ConfidentSpanExporter()
    except ImportError as exc:
        assert "deepeval[otel]" in str(exc), (
            f"ImportError missing 'deepeval[otel]': {exc}"
        )
        return
    raise AssertionError("ConfidentSpanExporter() did not raise ImportError")


def call_raises_actionable(callable_obj):
    try:
        callable_obj()
    except Exception as exc:
        if not is_actionable_otel_import_error(exc):
            raise AssertionError(
                f"expected actionable otel ImportError, got "
                f"{type(exc).__name__}: {exc}"
            )
        return
    raise AssertionError(f"{callable_obj} did not raise without otel")


check("grpc import blocked (blocker sanity)", check_grpc_blocked)

import importlib

_modules = {}


def _import_to_cache(label, module_name):
    _modules[label] = importlib.import_module(module_name)


check(
    "import tracing.otel.exporter",
    lambda: _import_to_cache("exporter", "deepeval.tracing.otel.exporter"),
)
check(
    "import tracing.otel.utils",
    lambda: _import_to_cache(
        "utils", "deepeval.tracing.otel.utils"
    ),
)
check(
    "import tracing.otel.context_aware_processor",
    lambda: _import_to_cache(
        "casp", "deepeval.tracing.otel.context_aware_processor"
    ),
)

check(
    "ConfidentSpanExporter() raises deepeval[otel]",
    lambda: make_exporter_raises_actionable(_modules["exporter"]),
)

check(
    "import integrations.pydantic_ai",
    lambda: _import_to_cache(
        "pydantic_ai", "deepeval.integrations.pydantic_ai"
    ),
)
check(
    "import integrations.agentcore",
    lambda: _import_to_cache("agentcore", "deepeval.integrations.agentcore"),
)
check(
    "import integrations.strands",
    lambda: _import_to_cache("strands", "deepeval.integrations.strands"),
)

check(
    "instrument_pydantic_ai() actionable",
    lambda: call_raises_actionable(_modules["pydantic_ai"].instrument_pydantic_ai),
)
check(
    "instrument_agentcore() actionable",
    lambda: call_raises_actionable(_modules["agentcore"].instrument_agentcore),
)
check(
    "instrument_strands() actionable",
    lambda: call_raises_actionable(_modules["strands"].instrument_strands),
)

if failures:
    print(f"CHILD RESULT: {len(failures)} failure(s)")
    sys.exit(1)
print("CHILD RESULT: all checks passed")
sys.exit(0)
"""


def test_otel_surfaces_import_and_fail_actionably_without_opentelemetry():
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "").rstrip(os.pathsep)
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0 and "[FAIL]" not in result.stdout, (
        f"child rc={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "CHILD RESULT: all checks passed" in output


def _top_level_otel_violations():
    violations = []
    for path in sorted((REPO_ROOT / "deepeval").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_UNGUARDED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = [node.module]
                    if node.level:
                        names = []
            for name in names:
                if name.split(".")[0] in OTEL_ROOTS:
                    violations.append(f"{rel}:{node.lineno} ({name})")
    return violations


def test_no_new_unguarded_top_level_otel_imports():
    violations = _top_level_otel_violations()
    assert violations == [], (
        "unguarded module-level opentelemetry/grpcio imports outside "
        f"allowlist {sorted(ALLOWED_UNGUARDED)}:\n" + "\n".join(violations)
    )
