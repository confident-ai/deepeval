"""Guards that the TypeScript model registry stays in sync with Python.

Per-model pricing and capability data, and the default model each provider falls
back to, live in ``deepeval/models/llms/constants.py``, which is the source of
truth for both SDKs. ``scripts/compile_model_registry.py`` projects it into
``typescript/src/models/registry/models.json`` for the TypeScript package. That
JSON is a committed build artifact, so it silently drifts whenever someone edits
a registry without recompiling. These tests fail loudly when that happens.
"""

import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile_model_registry.py"
MODELS_JSON = (
    REPO_ROOT / "typescript" / "src" / "models" / "registry" / "models.json"
)

# Every field of DeepEvalModelData, in camelCase. Kept in step with FIELDS in the
# compile script and the ModelData interface on the TypeScript side.
EXPECTED_FIELDS = {
    "supportsLogProbs",
    "maxLogProbs",
    "supportsMultimodal",
    "supportsStructuredOutputs",
    "supportsJson",
    "inputPrice",
    "outputPrice",
    "supportsTemperature",
    "supportsThinking",
}


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "_compile_model_registry", COMPILE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_models_json_is_up_to_date():
    compiler = _load_compiler()
    expected = compiler.render_registry_json(compiler.build_registry())
    assert MODELS_JSON.read_text(encoding="utf-8") == expected, (
        f"{MODELS_JSON} is out of date with deepeval/models/llms/constants.py. "
        "Re-run `python scripts/compile_model_registry.py`."
    )


def test_models_json_covers_every_registry():
    compiler = _load_compiler()
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))

    assert data["_meta"]["doNotEdit"] is True
    for namespace in compiler.REGISTRIES.values():
        assert data[namespace], f"{namespace} namespace is empty"


def test_models_json_carries_every_python_default():
    """The emitted defaults must be exactly Python's ``DEFAULT_MODELS``.

    This is what stops the two SDKs from falling back to different judge models:
    the TypeScript providers read these values instead of declaring their own.
    """
    compiler = _load_compiler()
    constants = compiler._load_constants()
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))

    assert data["_defaults"] == constants.DEFAULT_MODELS, (
        "_defaults in models.json disagrees with DEFAULT_MODELS in "
        "deepeval/models/llms/constants.py. Re-run "
        "`python scripts/compile_model_registry.py`."
    )


def test_every_default_model_has_pricing_data():
    """A default missing from its registry would evaluate at zero cost."""
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))

    for namespace, model in data["_defaults"].items():
        if namespace in _load_compiler().REGISTRYLESS_DEFAULTS:
            continue
        assert (
            model in data[namespace]
        ), f"default model {model!r} is absent from the {namespace} registry"
        entry = data[namespace][model]
        assert entry.get("inputPrice") and entry.get("outputPrice"), (
            f"default model {namespace}/{model} has no token pricing, so both "
            "SDKs would report its evaluations as free"
        )


def test_provider_modules_do_not_declare_their_own_defaults():
    """Defaults belong in ``constants.py``, the one file codegen reads.

    A provider module that hardcodes its own fallback is invisible to the
    compile script, which is exactly how Python and TypeScript drifted before.
    """
    llms_dir = REPO_ROOT / "deepeval" / "models" / "llms"
    offenders = []
    for path in sorted(llms_dir.glob("*_model.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.match(r"^default_\w*model\w*\s*=\s*[\"']", stripped):
                offenders.append(f"{path.name}:{lineno}: {stripped}")

    assert not offenders, (
        "provider modules declare module-level default model names:\n  "
        + "\n  ".join(offenders)
        + "\nMove them into DEFAULT_MODELS in deepeval/models/llms/constants.py "
        "so scripts/compile_model_registry.py can project them to TypeScript."
    )


def test_models_json_entries_are_well_formed():
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))

    for namespace, models in data.items():
        # Underscored keys are metadata (`_meta`, `_defaults`), not namespaces.
        if namespace.startswith("_"):
            continue
        for model, entry in models.items():
            unknown = set(entry) - EXPECTED_FIELDS
            assert not unknown, f"{namespace}.{model} has unknown {unknown}"
            for price_field in ("inputPrice", "outputPrice"):
                price = entry.get(price_field)
                assert (
                    price is None or price >= 0
                ), f"{namespace}.{model}.{price_field} must be >= 0"
