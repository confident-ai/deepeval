"""Community metric template languages and path helpers."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from deepeval.constants import HIDDEN_DIR

_ENGLISH_ALIASES = frozenset({"english", "en", "eng"})
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")

class MetricTemplateLanguage(str, Enum):
    """Languages accepted by ``deepeval translate --contribute``."""
    HINDI = "hindi"
    THAI = "thai"
    SPANISH = "spanish"
    GERMAN = "german"
    JAPANESE = "japanese"
    CANTONESE = "cantonese"
    TELUGU = "telugu"

def is_english(lang: str) -> bool:
    return lang.strip().lower() in _ENGLISH_ALIASES

def parse_language_slug(lang: str) -> str:
    """Normalizes a language string into a safe file slug."""
    slug = lang.strip().lower().replace("-", "_").replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_]+", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    
    if not slug or not _SLUG_RE.match(slug):
        raise ValueError(
            "Language slug must start with a letter and use only lowercase "
            "letters, digits, and underscores (e.g., 'vietnamese')."
        )
    if slug in _ENGLISH_ALIASES:
        raise ValueError(
            "English is the default language. Please unset DEEPEVAL_METRIC_TEMPLATE_LANGUAGE "
            "instead of explicitly translating to English."
        )
    return slug

def parse_language_setting(lang: str | None) -> str | None:
    """Parses the environment variable setting, returning None if English or unset."""
    if not lang or is_english(lang):
        return None
    return parse_language_slug(lang)

def require_valid_language(slug: str) -> str:
    """Ensures a language exists in the Enum for OSS contributions."""
    slug = parse_language_slug(slug)
    if slug not in MetricTemplateLanguage._value2member_map_:
        raise ValueError(
            f"Language '{slug}' is not currently supported for community contributions.\n"
            f"To add it, please first add '{slug.upper()} = \"{slug}\"' to the MetricTemplateLanguage Enum "
            f"in `deepeval/metric_templates/community/languages.py`."
        )
    return slug

def get_community_path(slug: str) -> Path:
    """Path to the shipped OS-contributed templates."""
    return Path(__file__).resolve().parent / f"templates.{slug}.json"

def get_custom_path(slug: str) -> Path:
    """Path to the user's localized local templates."""
    return Path(HIDDEN_DIR) / f"templates.{slug}.json"