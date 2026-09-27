"""Compile the Python model registries into the TypeScript package's models.json.

``deepeval/models/llms/constants.py`` is the single source of truth for per-model
pricing and capability data, and for the default model each provider falls back
to. This projects both into a committed artifact at
``typescript/src/models/registry/models.json``.

Unlike ``compile_metric_templates.py`` this is a ONE-WAY emit: Python keeps
importing ``constants.py`` directly, so nothing is written back.

Usage:
    python scripts/compile_model_registry.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSTANTS_PY = REPO_ROOT / "deepeval" / "models" / "llms" / "constants.py"
MODELS_JSON = (
    REPO_ROOT / "typescript" / "src" / "models" / "registry" / "models.json"
)

# Namespaces that carry a default model but no pricing registry, so
# `_build_defaults` must not demand a registry entry for them. OpenRouter model
# names are `provider/model` strings that DeepEval deliberately does not validate.
REGISTRYLESS_DEFAULTS = {"openrouter"}

# Python registry variable -> namespace key in the emitted JSON.
REGISTRIES = {
    "OPENAI_MODELS_DATA": "openai",
    "ANTHROPIC_MODELS_DATA": "anthropic",
    "GEMINI_MODELS_DATA": "gemini",
    "GROK_MODELS_DATA": "grok",
    "KIMI_MODELS_DATA": "kimi",
    "DEEPSEEK_MODELS_DATA": "deepseek",
    "OLLAMA_MODELS_DATA": "ollama",
    "BEDROCK_MODELS_DATA": "bedrock",
}

# Every field of DeepEvalModelData, with its Python default. A field is emitted
# only when it differs from the default, and the TS side re-applies the same
# defaults.
FIELDS: Dict[str, Any] = {
    "supports_log_probs": None,
    "max_log_probs": None,
    "supports_multimodal": None,
    "supports_structured_outputs": None,
    "supports_json": None,
    "input_price": None,
    "output_price": None,
    "supports_temperature": True,
    "supports_thinking": None,
}


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.capitalize() for word in rest)


class _RecordingModelData:
    """Stands in for ``DeepEvalModelData`` and captures the kwargs it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.recorded = kwargs


def _load_constants() -> types.ModuleType:
    """Execute ``constants.py`` against a stubbed ``deepeval.models.base_model``.

    Importing it normally would pull in ``deepeval/__init__`` and its
    third-party dependencies; stubbing its one deepeval import keeps this script,
    and so the CI drift check, on the standard library alone.
    """
    stub = types.ModuleType("deepeval.models.base_model")
    stub.DeepEvalModelData = _RecordingModelData

    packages = {}
    for name in ("deepeval", "deepeval.models"):
        package = types.ModuleType(name)
        package.__path__ = []  # mark as a package so submodule imports resolve
        packages[name] = package
    packages["deepeval.models"].base_model = stub

    injected = {**packages, "deepeval.models.base_model": stub}
    saved = {name: sys.modules.get(name) for name in injected}
    sys.modules.update(injected)
    try:
        spec = importlib.util.spec_from_file_location(
            "_deepeval_model_constants", CONSTANTS_PY
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {CONSTANTS_PY}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _extract_entry(namespace: str, model: str, value: Any) -> Dict[str, Any]:
    if not callable(value):
        raise TypeError(
            f"{namespace}.{model}: expected a make_model_data(...) factory, "
            f"got {type(value).__name__}. Registries must stay declarative."
        )
    data = value()
    if not isinstance(data, _RecordingModelData):
        raise TypeError(
            f"{namespace}.{model}: factory returned {type(data).__name__} "
            "instead of model data. Registries must stay declarative."
        )

    unknown = set(data.recorded) - set(FIELDS)
    if unknown:
        raise KeyError(
            f"{namespace}.{model}: unknown model data field(s) "
            f"{sorted(unknown)}. Add them to FIELDS here and to the ModelData "
            "interface in typescript/src/models/registry/index.ts."
        )

    return {
        _camel(field): data.recorded[field]
        for field, default in FIELDS.items()
        if field in data.recorded and data.recorded[field] != default
    }


def _build_defaults(
    constants: types.ModuleType, bundle: dict
) -> Dict[str, str]:
    """Project ``DEFAULT_MODELS``, checking each default has pricing data.

    A default that is missing from its own registry would evaluate at zero cost
    in both SDKs, so it is an error here rather than a silent one at runtime.
    """
    defaults = getattr(constants, "DEFAULT_MODELS", None)
    if not isinstance(defaults, dict):
        raise AttributeError(
            f"DEFAULT_MODELS is missing from {CONSTANTS_PY}, or is not a dict. "
            "Provider defaults must live there so both SDKs read one value."
        )

    for namespace, model in sorted(defaults.items()):
        if not isinstance(model, str) or not model:
            raise TypeError(
                f"DEFAULT_MODELS[{namespace!r}] must be a non-empty model name, "
                f"got {model!r}."
            )
        if namespace in REGISTRYLESS_DEFAULTS:
            continue
        if namespace not in bundle:
            raise KeyError(
                f"DEFAULT_MODELS[{namespace!r}] names a namespace with no "
                "registry. Add one to REGISTRIES, or list the namespace in "
                "REGISTRYLESS_DEFAULTS if it genuinely has no pricing data."
            )
        if model not in bundle[namespace]:
            raise KeyError(
                f"DEFAULT_MODELS[{namespace!r}] is {model!r}, which is absent "
                f"from {namespace.upper()}_MODELS_DATA. A default with no "
                "pricing entry would be billed as free by both SDKs."
            )

    return dict(sorted(defaults.items()))


def build_registry() -> dict:
    constants = _load_constants()

    bundle: dict = {
        "_meta": {
            "generatedBy": "scripts/compile_model_registry.py",
            "source": "deepeval/models/llms/constants.py",
            "doNotEdit": True,
        }
    }
    for variable, namespace in REGISTRIES.items():
        registry = getattr(constants, variable, None)
        if registry is None:
            raise AttributeError(
                f"{variable} is missing from {CONSTANTS_PY}. Update REGISTRIES."
            )
        # Iterate raw values rather than going through ModelDataRegistry.get(),
        # which would unwrap the factories before we can validate them.
        bundle[namespace] = {
            model: _extract_entry(namespace, model, value)
            for model, value in dict.items(registry)
        }
    bundle["_defaults"] = _build_defaults(constants, bundle)
    return bundle


def render_registry_json(bundle: dict) -> str:
    """Serialize exactly as written to ``models.json``, so tests can compare."""
    return json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    content = render_registry_json(build_registry())
    MODELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODELS_JSON.write_text(content, encoding="utf-8")
    print(f"Updated {MODELS_JSON}")


if __name__ == "__main__":
    main()
