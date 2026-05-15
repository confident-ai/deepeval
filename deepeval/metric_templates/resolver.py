from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import jinja2

from deepeval.constants import HIDDEN_DIR

class MetricTemplateNotFoundError(KeyError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

class MetricTemplateInterpolationError(ValueError):
    def __init__(self, message: str, unresolved: Set[str]) -> None:
        super().__init__(message)
        self.unresolved = unresolved

def _list_template_classes(bundle: Mapping[str, Any]) -> str:
    names = sorted(
        k for k, v in bundle.items() if not k.startswith("_") and isinstance(v, dict)
    )
    return ", ".join(names) if names else "(none)"

def _method_names_from_class_entry(entry: Any) -> Set[str]:
    if not isinstance(entry, dict):
        return set()
    return {k for k, v in entry.items() if not k.startswith("_") and isinstance(v, str)}

class TemplateRegistry:
    """Encapsulates template loading, caching, and Jinja environment setup."""
    def __init__(self) -> None:
        self._bundle: Optional[Dict[str, Any]] = None
        self._hidden_bundle: Optional[Dict[str, Any]] = None
        self._hidden_bundle_tried: bool = False
        
        # Cache Jinja environments
        self._jinja_strict: Optional[jinja2.Environment] = None
        self._jinja_lenient: Optional[jinja2.Environment] = None

    def clear(self) -> None:
        self._bundle = None
        self._hidden_bundle = None
        self._hidden_bundle_tried = False
        self._jinja_strict = None
        self._jinja_lenient = None

    def get_bundle(self) -> Dict[str, Any]:
        if self._bundle is None:
            self._bundle = json.loads(self._read_bundle_text())
        return self._bundle

    def get_hidden_bundle(self) -> Optional[Dict[str, Any]]:
        if not self._hidden_bundle_tried:
            self._hidden_bundle_tried = True
            self._hidden_bundle = self._try_load_hidden_bundle()
        return self._hidden_bundle

    def get_jinja_env(self, strict: bool) -> jinja2.Environment:
        if strict:
            if self._jinja_strict is None:
                self._jinja_strict = jinja2.Environment(undefined=jinja2.StrictUndefined)
            return self._jinja_strict
        else:
            if self._jinja_lenient is None:
                self._jinja_lenient = jinja2.Environment()
            return self._jinja_lenient

    @staticmethod
    def _read_bundle_text() -> str:
        try:
            ref = resources.files("deepeval.metric_templates").joinpath("templates.json")
            return ref.read_text(encoding="utf-8")
        except (ModuleNotFoundError, OSError, TypeError, FileNotFoundError):
            here = Path(__file__).resolve().parent / "templates.json"
            return here.read_text(encoding="utf-8")

    @staticmethod
    def _try_load_hidden_bundle() -> Optional[Dict[str, Any]]:
        path = Path(HIDDEN_DIR) / "templates.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

# Module-level singleton
_registry = TemplateRegistry()

# --- Public API ---

def clear_metric_template_cache() -> None:
    _registry.clear()

def list_methods(class_name: str) -> list[str]:
    names: Set[str] = set()
    hidden = _registry.get_hidden_bundle()
    if hidden is not None:
        names |= _method_names_from_class_entry(hidden.get(class_name))
    
    bundle = _registry.get_bundle()
    names |= _method_names_from_class_entry(bundle.get(class_name))
    
    if not names:
        raise MetricTemplateNotFoundError(
            f"No metric templates for class {class_name!r}. "
            f"Known classes: {_list_template_classes(bundle)}"
        )
    return sorted(names)

def get_raw_template(class_name: str, method: str) -> str:
    hidden = _registry.get_hidden_bundle()
    if hidden is not None and class_name in hidden:
        h_entry = hidden.get(class_name)
        h_body = h_entry.get(method) if isinstance(h_entry, dict) else None
        if isinstance(h_body, str):
            return h_body

    bundle = _registry.get_bundle()
    entry = bundle.get(class_name)
    body = entry.get(method) if isinstance(entry, dict) else None
    
    if not isinstance(body, str):
        hint = (
            f" Available methods for {class_name!r}: {list_methods(class_name)!r}"
            if isinstance(entry, dict) or (hidden is not None and class_name in hidden)
            else f" Known classes: {_list_template_classes(bundle)}"
        )
        raise MetricTemplateNotFoundError(f"No template for {class_name!r}.{method!r}.{hint}")
    return body

def get_bundle_only_template(class_name: str, method: str) -> str:
    """Return the template string from the shipped English bundle only."""
    bundle = _registry.get_bundle()
    entry = bundle.get(class_name)
    body = entry.get(method) if isinstance(entry, dict) else None
    if not isinstance(body, str):
        hint = ""
        if isinstance(entry, dict):
            keys = sorted(k for k, v in entry.items() if not k.startswith("_") and isinstance(v, str))
            hint = f" Available methods for {class_name!r}: {keys!r}"
        else:
            hint = f" Known classes: {_list_template_classes(bundle)}"
        raise MetricTemplateNotFoundError(f"No bundle template for {class_name!r}.{method!r}.{hint}")
    return body

def iter_bundle_template_methods(class_name: str) -> list[tuple[str, str]]:
    """Return ``(method, template)`` pairs from the shipped bundle only."""
    bundle = _registry.get_bundle()
    entry = bundle.get(class_name)
    if not isinstance(entry, dict):
        raise MetricTemplateNotFoundError(
            f"No metric templates for class {class_name!r}. "
            f"Known classes: {_list_template_classes(bundle)}"
        )
    pairs = [(k, v) for k, v in sorted(entry.items()) if not k.startswith("_") and isinstance(v, str)]
    if not pairs:
        raise MetricTemplateNotFoundError(f"No string template methods for class {class_name!r}.")
    return pairs

def resolve_template(
    class_name: str,
    method: str,
    *,
    multimodal: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> str:
    raw_template = get_raw_template(class_name, method)
    bundle = _registry.get_bundle()
    fragments = bundle.get("_fragments", {})

    env = _registry.get_jinja_env(strict=strict)
    
    try:
        template = env.from_string(raw_template)
        return template.render(
            multimodal=multimodal,
            _fragments=fragments,
            **kwargs
        )
    except jinja2.UndefinedError as e:
        # StrictUndefined catches missing variables gracefully
        raise MetricTemplateInterpolationError(
            f"Missing variable during template render: {e.message}", 
            unresolved=set()
        ) from e
    except jinja2.TemplateSyntaxError as e:
        raise MetricTemplateInterpolationError(
            f"Jinja syntax error in template: {e.message}", 
            unresolved=set()
        ) from e
