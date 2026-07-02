"""Guards that the compiled ``templates.json`` bundles stay in sync with sources.

The metric prompt templates live as ``.txt`` files under
``deepeval/metrics/**/templates/<Class>/<method>.txt`` (plus shared fragments),
and are compiled into ``templates/metrics/templates.json`` by
``scripts/compile_metric_templates.py``, which emits the bundle into BOTH the
Python package (``deepeval/templates/metrics/``) and the TypeScript package
(``typescript/src/templates/metrics/``). Those JSON files are committed build
artifacts, so they can silently drift if someone edits a ``.txt`` without
recompiling. These tests fail loudly when that happens.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile_metric_templates.py"
PY_TEMPLATES_JSON = (
    REPO_ROOT / "deepeval" / "templates" / "metrics" / "templates.json"
)
TS_TEMPLATES_JSON = (
    REPO_ROOT
    / "typescript"
    / "src"
    / "templates"
    / "metrics"
    / "templates.json"
)


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "_compile_metric_templates", COMPILE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _point_compiler_at_repo(compiler, repo_root):
    package_root = repo_root / "deepeval"
    compiler.REPO_ROOT = repo_root
    compiler.PACKAGE_ROOT = package_root
    compiler.TEMPLATES_JSON = (
        package_root / "templates" / "metrics" / "templates.json"
    )
    compiler.TS_TEMPLATES_JSON = (
        repo_root
        / "typescript"
        / "src"
        / "templates"
        / "metrics"
        / "templates.json"
    )
    compiler.FRAGMENTS_DIR = compiler.TEMPLATES_JSON.parent / "fragments"


def test_templates_json_is_up_to_date():
    compiler = _load_compiler()
    expected = compiler.render_bundle_json(compiler.build_bundle())
    for path in (PY_TEMPLATES_JSON, TS_TEMPLATES_JSON):
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path} is out of date with the template .txt files. "
            "Re-run `python scripts/compile_metric_templates.py`."
        )


def test_templates_json_is_valid_and_nonempty():
    data = json.loads(PY_TEMPLATES_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data
    # Every class entry maps method -> string template.
    for class_name, methods in data.items():
        assert isinstance(methods, dict), class_name
        for method, body in methods.items():
            assert isinstance(body, str), f"{class_name}.{method}"


def test_argument_correctness_verdict_prompt_counts_tool_calls():
    compiler = _load_compiler()
    bundle = compiler.build_bundle()
    prompt = bundle["ArgumentCorrectnessMetric"]["generate_verdicts"]

    assert "Tool Calls:" in prompt
    assert "for each tool call" in prompt
    assert "number of tool calls" in prompt
    assert "for each statement" not in prompt
    assert "`statements`" not in prompt


def test_compiler_rejects_symlinked_template_sources(tmp_path):
    compiler = _load_compiler()
    repo_root = tmp_path / "repo"
    templates_dir = repo_root / "deepeval" / "metrics" / "demo" / "templates"
    fragments_dir = (
        repo_root / "deepeval" / "templates" / "metrics" / "fragments"
    )
    templates_dir.mkdir(parents=True)
    fragments_dir.mkdir(parents=True)
    (templates_dir / "class.txt").write_text("DemoMetric", encoding="utf-8")

    outside_file = tmp_path / "runner-token.txt"
    outside_file.write_text("secret-token-value", encoding="utf-8")
    try:
        (templates_dir / "generate_verdicts.txt").symlink_to(outside_file)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are not supported in this environment: {exc}")

    _point_compiler_at_repo(compiler, repo_root)

    with pytest.raises(ValueError, match="symlinked template source"):
        compiler.build_bundle()


def test_compiler_rejects_symlinked_template_outputs(tmp_path):
    compiler = _load_compiler()
    repo_root = tmp_path / "repo"
    templates_dir = repo_root / "deepeval" / "metrics" / "demo" / "templates"
    fragments_dir = (
        repo_root / "deepeval" / "templates" / "metrics" / "fragments"
    )
    ts_templates_dir = (
        repo_root / "typescript" / "src" / "templates" / "metrics"
    )
    templates_dir.mkdir(parents=True)
    fragments_dir.mkdir(parents=True)
    ts_templates_dir.mkdir(parents=True)
    (templates_dir / "class.txt").write_text("DemoMetric", encoding="utf-8")
    (templates_dir / "generate_verdicts.txt").write_text(
        "Safe prompt text", encoding="utf-8"
    )

    outside_file = tmp_path / "runner-output.txt"
    outside_file.write_text("do not overwrite", encoding="utf-8")
    try:
        (ts_templates_dir / "templates.json").symlink_to(outside_file)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are not supported in this environment: {exc}")

    _point_compiler_at_repo(compiler, repo_root)

    with pytest.raises(ValueError, match="symlinked template output"):
        compiler.main()

    assert outside_file.read_text(encoding="utf-8") == "do not overwrite"


def test_compiler_rejects_broken_symlinked_template_outputs(tmp_path):
    compiler = _load_compiler()
    repo_root = tmp_path / "repo"
    templates_dir = repo_root / "deepeval" / "metrics" / "demo" / "templates"
    fragments_dir = (
        repo_root / "deepeval" / "templates" / "metrics" / "fragments"
    )
    ts_templates_dir = (
        repo_root / "typescript" / "src" / "templates" / "metrics"
    )
    outside_dir = tmp_path / "outside"
    templates_dir.mkdir(parents=True)
    fragments_dir.mkdir(parents=True)
    ts_templates_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (templates_dir / "class.txt").write_text("DemoMetric", encoding="utf-8")
    (templates_dir / "generate_verdicts.txt").write_text(
        "Safe prompt text", encoding="utf-8"
    )

    outside_file = outside_dir / "created-by-symlink.json"
    try:
        (ts_templates_dir / "templates.json").symlink_to(outside_file)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are not supported in this environment: {exc}")

    _point_compiler_at_repo(compiler, repo_root)

    with pytest.raises(ValueError, match="symlinked template output"):
        compiler.main()

    assert not outside_file.exists()


def test_compiler_rejects_symlinked_template_output_parents(tmp_path):
    compiler = _load_compiler()
    repo_root = tmp_path / "repo"
    templates_dir = repo_root / "deepeval" / "metrics" / "demo" / "templates"
    fragments_dir = (
        repo_root / "deepeval" / "templates" / "metrics" / "fragments"
    )
    src_templates_dir = repo_root / "typescript" / "src"
    outside_dir = tmp_path / "outside"
    templates_dir.mkdir(parents=True)
    fragments_dir.mkdir(parents=True)
    src_templates_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (templates_dir / "class.txt").write_text("DemoMetric", encoding="utf-8")
    (templates_dir / "generate_verdicts.txt").write_text(
        "Safe prompt text", encoding="utf-8"
    )

    try:
        (src_templates_dir / "templates").symlink_to(
            outside_dir, target_is_directory=True
        )
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are not supported in this environment: {exc}")

    _point_compiler_at_repo(compiler, repo_root)

    with pytest.raises(ValueError, match="symlinked template output"):
        compiler.main()

    assert not (outside_dir / "metrics").exists()


def test_compiler_validates_all_outputs_before_writing(tmp_path):
    compiler = _load_compiler()
    repo_root = tmp_path / "repo"
    templates_dir = repo_root / "deepeval" / "metrics" / "demo" / "templates"
    py_templates_dir = repo_root / "deepeval" / "templates" / "metrics"
    ts_templates_dir = (
        repo_root / "typescript" / "src" / "templates" / "metrics"
    )
    templates_dir.mkdir(parents=True)
    py_templates_dir.mkdir(parents=True)
    ts_templates_dir.mkdir(parents=True)
    (templates_dir / "class.txt").write_text("DemoMetric", encoding="utf-8")
    (templates_dir / "generate_verdicts.txt").write_text(
        "Safe prompt text", encoding="utf-8"
    )
    py_output = py_templates_dir / "templates.json"
    py_output.write_text('{"ExistingMetric": {}}\n', encoding="utf-8")

    outside_file = tmp_path / "runner-output.json"
    outside_file.write_text("do not overwrite", encoding="utf-8")
    try:
        (ts_templates_dir / "templates.json").symlink_to(outside_file)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are not supported in this environment: {exc}")

    _point_compiler_at_repo(compiler, repo_root)

    with pytest.raises(ValueError, match="symlinked template output"):
        compiler.main()

    assert py_output.read_text(encoding="utf-8") == '{"ExistingMetric": {}}\n'
    assert outside_file.read_text(encoding="utf-8") == "do not overwrite"


def test_compiler_creates_missing_template_output_directories(tmp_path):
    compiler = _load_compiler()
    repo_root = tmp_path / "repo"
    templates_dir = repo_root / "deepeval" / "metrics" / "demo" / "templates"
    templates_dir.mkdir(parents=True)
    (templates_dir / "class.txt").write_text("DemoMetric", encoding="utf-8")
    (templates_dir / "generate_verdicts.txt").write_text(
        "Safe prompt text", encoding="utf-8"
    )

    _point_compiler_at_repo(compiler, repo_root)
    compiler.main()

    assert compiler.TEMPLATES_JSON.is_file()
    assert compiler.TS_TEMPLATES_JSON.is_file()
