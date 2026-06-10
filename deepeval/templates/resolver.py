from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import jinja2


class MetricTemplateNotFoundError(KeyError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MetricTemplateInterpolationError(ValueError):
    def __init__(self, message: str, unresolved: Set[str]) -> None:
        super().__init__(message)
        self.unresolved = unresolved


class TemplateRegistry:
    """Loads the shipped base (English) `templates.json` and caches the Jinja
    environments plus compiled templates. Templates are static, so each
    `(class, method, strict)` is compiled at most once."""

    def __init__(self) -> None:
        self._base_templates: Optional[Dict[str, Any]] = None
        self._jinja_strict: Optional[jinja2.Environment] = None
        self._jinja_lenient: Optional[jinja2.Environment] = None
        self._compiled: Dict[Tuple[str, str, bool], jinja2.Template] = {}

    def clear(self) -> None:
        self._base_templates = None
        self._jinja_strict = None
        self._jinja_lenient = None
        self._compiled.clear()

    def get_base_templates(self) -> Dict[str, Any]:
        if self._base_templates is None:
            try:
                ref = resources.files("deepeval.metric_templates").joinpath(
                    "templates.json"
                )
                raw = ref.read_text(encoding="utf-8")
            except (ModuleNotFoundError, OSError, TypeError, FileNotFoundError):
                raw = (
                    Path(__file__).resolve().parent / "templates.json"
                ).read_text(encoding="utf-8")
            self._base_templates = json.loads(raw)
        return self._base_templates

    def get_jinja_env(self, strict: bool) -> jinja2.Environment:
        if strict:
            if self._jinja_strict is None:
                self._jinja_strict = jinja2.Environment(
                    undefined=jinja2.StrictUndefined
                )
            return self._jinja_strict
        if self._jinja_lenient is None:
            self._jinja_lenient = jinja2.Environment()
        return self._jinja_lenient

    def get_compiled(
        self, class_name: str, method: str, strict: bool
    ) -> jinja2.Template:
        key = (class_name, method, strict)
        template = self._compiled.get(key)
        if template is None:
            raw = get_raw_template(class_name, method)
            template = self.get_jinja_env(strict).from_string(raw)
            self._compiled[key] = template
        return template


_registry = TemplateRegistry()


def clear_metric_template_cache() -> None:
    _registry.clear()


def get_raw_template(class_name: str, method: str) -> str:
    """Return the raw (un-rendered) base template string for a class/method."""
    base = _registry.get_base_templates()
    entry = base.get(class_name, {})
    body = entry.get(method)

    if not body or not isinstance(body, str):
        raise MetricTemplateNotFoundError(
            f"No template found for {class_name!r}.{method!r}. "
            f"Available classes: {', '.join(base.keys())}"
        )
    return body


def iter_base_template_methods(class_name: str) -> list[tuple[str, str]]:
    """Iterate over all `(method, template)` pairs for a class in the base bundle."""
    base = _registry.get_base_templates()
    entry = base.get(class_name)

    if not entry or not isinstance(entry, dict):
        raise MetricTemplateNotFoundError(
            f"No metric templates for class {class_name!r}."
        )

    pairs = [
        (k, v)
        for k, v in sorted(entry.items())
        if not k.startswith("_") and isinstance(v, str)
    ]
    if not pairs:
        raise MetricTemplateNotFoundError(
            f"No string template methods for class {class_name!r}."
        )
    return pairs


def resolve_template(
    class_name: str,
    method: str,
    *,
    multimodal: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> str:
    """Render a base metric template to a final prompt via Jinja2.

    `_fragments` (shared reusable snippets) and `multimodal` are always
    available to the template; everything else is passed through `kwargs`.
    """
    fragments = _registry.get_base_templates().get("_fragments", {})
    try:
        template = _registry.get_compiled(class_name, method, strict)
        return template.render(
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
