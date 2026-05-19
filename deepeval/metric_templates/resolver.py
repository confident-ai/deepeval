from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import jinja2
from rich import print

from deepeval.metric_templates.community.languages import (
    parse_language_setting,
    get_community_path,
    get_custom_path,
)

class MetricTemplateNotFoundError(KeyError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

class MetricTemplateInterpolationError(ValueError):
    def __init__(self, message: str, unresolved: Set[str]) -> None:
        super().__init__(message)
        self.unresolved = unresolved

def _read_json(path: Path) -> Dict[str, Any]:
    """Helper to cleanly load a JSON file, returning an empty dict if missing/invalid."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}

class TemplateRegistry:
    def __init__(self) -> None:
        self._base_templates: Optional[Dict[str, Any]] = None
        self._community_templates: Dict[str, Dict[str, Any]] = {}
        self._custom_templates: Dict[str, Dict[str, Any]] = {}
        
        self._warned_keys: Set[str] = set()
        self._jinja_strict: Optional[jinja2.Environment] = None
        self._jinja_lenient: Optional[jinja2.Environment] = None

    def clear(self) -> None:
        self._base_templates = None
        self._community_templates.clear()
        self._custom_templates.clear()
        self._warned_keys.clear()
        self._jinja_strict = None
        self._jinja_lenient = None

    def get_base_templates(self) -> Dict[str, Any]:
        if self._base_templates is None:
            try:
                ref = resources.files("deepeval.metric_templates").joinpath("templates.json")
                raw = ref.read_text(encoding="utf-8")
            except (ModuleNotFoundError, OSError, TypeError, FileNotFoundError):
                raw = (Path(__file__).resolve().parent / "templates.json").read_text(encoding="utf-8")
            self._base_templates = json.loads(raw)
        return self._base_templates

    def get_community_templates(self, slug: str) -> Dict[str, Any]:
        if slug not in self._community_templates:
            self._community_templates[slug] = _read_json(get_community_path(slug))
        return self._community_templates[slug]

    def get_custom_templates(self, slug: str) -> Dict[str, Any]:
        if slug not in self._custom_templates:
            self._custom_templates[slug] = _read_json(get_custom_path(slug))
        return self._custom_templates[slug]

    def get_jinja_env(self, strict: bool) -> jinja2.Environment:
        if strict:
            if self._jinja_strict is None:
                self._jinja_strict = jinja2.Environment(undefined=jinja2.StrictUndefined)
            return self._jinja_strict
        if self._jinja_lenient is None:
            self._jinja_lenient = jinja2.Environment()
        return self._jinja_lenient

    def warn_once(self, key: str, message: str) -> None:
        if key not in self._warned_keys:
            print(f"[yellow]Warning:[/yellow] {message}")
            self._warned_keys.add(key)


_registry = TemplateRegistry()

def clear_metric_template_cache() -> None:
    _registry.clear()

def get_active_metric_template_language() -> Optional[str]:
    try:
        from deepeval.config.settings import get_settings
        return parse_language_setting(get_settings().DEEPEVAL_METRIC_TEMPLATE_LANGUAGE)
    except Exception:
        return None

def get_raw_template(class_name: str, method: str) -> str:
    """Resolves a template by checking Custom -> Community -> Base (English)."""
    slug = get_active_metric_template_language()
    
    if slug is not None:
        # 1. Check user's custom localized templates (~/.deepeval/...)
        custom = _registry.get_custom_templates(slug)
        if class_name in custom and method in custom.get(class_name, {}):
            return custom[class_name][method]

        # 2. Check shipped community templates
        community = _registry.get_community_templates(slug)
        if class_name in community and method in community.get(class_name, {}):
            return community[class_name][method]

        # 3. Throw a helpful, actionable warning before falling back to English
        _registry.warn_once(
            f"lang_missing_{slug}_{class_name}",
            f"No '{slug}' translation found for [bold]{class_name}[/bold]. Defaulting to English.\n"
            f"-> Run [cyan]deepeval translate {slug} --metrics {class_name}[/cyan] to generate it."
        )

    # Fallback to base English templates
    base = _registry.get_base_templates()
    entry = base.get(class_name, {})
    body = entry.get(method)

    if not body or not isinstance(body, str):
        raise MetricTemplateNotFoundError(
            f"No template found for {class_name!r}.{method!r}. "
            f"Available classes: {', '.join(base.keys())}"
        )
    return body

def get_base_template(class_name: str, method: str) -> str:
    """Return the English template from the shipped base bundle only."""
    base = _registry.get_base_templates()
    body = base.get(class_name, {}).get(method)
    
    if not body or not isinstance(body, str):
        raise MetricTemplateNotFoundError(f"No base template found for {class_name!r}.{method!r}.")
    return body

def iter_base_template_methods(class_name: str) -> list[tuple[str, str]]:
    """Iterates over all (method, template) pairs for a given class in the base file."""
    base = _registry.get_base_templates()
    entry = base.get(class_name)
    
    if not entry or not isinstance(entry, dict):
        raise MetricTemplateNotFoundError(f"No metric templates for class {class_name!r}.")
        
    pairs = [(k, v) for k, v in sorted(entry.items()) if not k.startswith("_") and isinstance(v, str)]
    if not pairs:
        raise MetricTemplateNotFoundError(f"No string template methods for class {class_name!r}.")
    return pairs

def resolve_base_template(
    class_name: str,
    method: str,
    *,
    multimodal: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> str:
    raw_template = get_base_template(class_name, method)
    fragments = _registry.get_base_templates().get("_fragments", {})
    env = _registry.get_jinja_env(strict=strict)

    try:
        return env.from_string(raw_template).render(
            multimodal=multimodal,
            _fragments=fragments,
            **kwargs,
        )
    except jinja2.UndefinedError as e:
        raise MetricTemplateInterpolationError(
            f"Missing variable during template render: {e.message}",
            unresolved=set(),
        ) from e
    except jinja2.TemplateSyntaxError as e:
        raise MetricTemplateInterpolationError(
            f"Jinja syntax error in template: {e.message}",
            unresolved=set(),
        ) from e


def resolve_template(
    class_name: str,
    method: str,
    *,
    multimodal: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> str:
    """Fetches the best localized template and renders it via Jinja2."""
    raw_template = get_raw_template(class_name, method)
    fragments = _registry.get_base_templates().get("_fragments", {})
    env = _registry.get_jinja_env(strict=strict)

    try:
        return env.from_string(raw_template).render(
            multimodal=multimodal,
            _fragments=fragments,
            **kwargs,
        )
    except jinja2.UndefinedError as e:
        raise MetricTemplateInterpolationError(
            f"Missing variable during template render: {e.message}",
            unresolved=set(),
        ) from e
    except jinja2.TemplateSyntaxError as e:
        raise MetricTemplateInterpolationError(
            f"Jinja syntax error in template: {e.message}",
            unresolved=set(),
        ) from e
