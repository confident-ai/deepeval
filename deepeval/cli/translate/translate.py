from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from deepeval.cli.translate.schema import TranslationRewriteSchema
from deepeval.metric_templates.community.languages import (
    get_community_path,
    get_custom_path,
)
from deepeval.metrics.utils import initialize_model, trimAndLoadJson

_JINJA_TOKEN_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}")

def run_translation_llm(
    *,
    prompt: str,
    model: Optional[str],
) -> str:
    llm, using_native = initialize_model(model)

    if using_native:
        result, _cost = llm.generate_with_schema(
            prompt, schema=TranslationRewriteSchema
        )
    else:
        result = llm.generate_with_schema(
            prompt, schema=TranslationRewriteSchema
        )

    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]

    if isinstance(result, TranslationRewriteSchema):
        return result.rewritten_template.strip()

    data = trimAndLoadJson(result, metric=None)
    parsed = TranslationRewriteSchema.model_validate(data)
    return parsed.rewritten_template.strip()

def find_missing_placeholders(source: str, translated: str) -> List[str]:
    """Ensures all Jinja variables and blocks from the source exist in the translation."""
    source_tokens = set(_JINJA_TOKEN_RE.findall(source))
    return sorted(t for t in source_tokens if t not in translated)

def _get_target_path(slug: str, *, contribute: bool) -> Path:
    """Routes the save/load operation to the correct directory based on the CLI flag."""
    return get_community_path(slug) if contribute else get_custom_path(slug)

def load_localized_templates(slug: str, *, contribute: bool = False) -> Dict[str, Any]:
    """Loads existing translations to avoid re-translating existing keys."""
    path = _get_target_path(slug, contribute=contribute)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}

def save_localized_templates(
    slug: str,
    updates: Mapping[str, Mapping[str, str]],
    *,
    contribute: bool = False,
) -> Path:
    """Merges new translations into the target JSON file and saves it."""
    path = _get_target_path(slug, contribute=contribute)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    existing = load_localized_templates(slug, contribute=contribute)

    for cls_name, methods in updates.items():
        if not methods:
            continue
        cur = dict(existing.get(cls_name) or {})
        cur.update(methods)
        existing[cls_name] = cur

    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
