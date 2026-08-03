"""Guards that the TypeScript model registry stays in sync with Python.

Per-model pricing and capability data lives in
``deepeval/models/llms/constants.py``, which is the source of truth for both
SDKs. ``scripts/compile_model_registry.py`` projects it into
``typescript/src/models/registry/models.json`` for the TypeScript package. That
JSON is a committed build artifact, so it silently drifts whenever someone edits
a registry without recompiling. These tests fail loudly when that happens.
"""

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile_model_registry.py"
MODELS_JSON = (
    REPO_ROOT
    / "typescript"
    / "src"
    / "models"
    / "registry"
    / "models.json"
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


def test_models_json_entries_are_well_formed():
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))

    for namespace, models in data.items():
        if namespace == "_meta":
            continue
        for model, entry in models.items():
            unknown = set(entry) - EXPECTED_FIELDS
            assert not unknown, f"{namespace}.{model} has unknown {unknown}"
            for price_field in ("inputPrice", "outputPrice"):
                price = entry.get(price_field)
                assert price is None or price >= 0, (
                    f"{namespace}.{model}.{price_field} must be >= 0"
                )
