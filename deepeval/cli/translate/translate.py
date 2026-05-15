from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from deepeval.cli.translate.schema import TranslationRewriteSchema
from deepeval.constants import HIDDEN_DIR
from deepeval.metrics.utils import initialize_model, trimAndLoadJson

# Matches both {{ variables }} and {% logic blocks %}
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


def save_translated_templates(updates: Mapping[str, Mapping[str, str]]) -> None:
    """Merges newly translated templates into the hidden overrides JSON file."""
    path = Path(HIDDEN_DIR) / "templates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = {}
            
    for cls_name, methods in updates.items():
        if not methods:
            continue
        cur = existing.get(cls_name)
        if not isinstance(cur, dict):
            cur = {}
        else:
            cur = dict(cur)
        for mk, mv in methods.items():
            cur[mk] = mv
        existing[cls_name] = cur
        
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )