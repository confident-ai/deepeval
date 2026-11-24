import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from deepeval.classifiers.classifier.config import EvidenceConfig
from deepeval.classifiers.classifier.schema import EvidenceItem
from deepeval.classifiers.classifier.steps.truncate import serialize
from deepeval.classifiers.utils import (
    MULTI_TURN_CONTENT_PARAMS,
    SINGLE_TURN_CONTENT_PARAMS,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall
from deepeval.test_run.api import Evidence

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import ClassificationPrompt

_MIN_EVIDENCE_QUOTE_CHARS = 8
_WHITESPACE_RE = re.compile(r"\s+")
_TRUNCATION_TAIL_RE = re.compile(
    r"""…?\[truncated\][\s"'\\,}\]]*$""", re.IGNORECASE
)
_TRUNCATION_MARKER_RE = re.compile(r"…?\[truncated\]", re.IGNORECASE)
_TRUNCATION_CLAIM_RE = re.compile(
    r"truncat|cut[ -]?off|cut short|incomplete (?:output|response|content)",
    re.IGNORECASE,
)
TRUNCATION_MARKER_REASON = (
    'The only problem found was the "…[truncated]" marker added where long '
    "fields are shortened, which is not part of the application's output."
)


_TURN_FIELDS = ["content", "retrieval_context", "tools_called"]
_TRACE_FIELDS = [
    "input",
    "output",
    "error",
    "tools_called",
    "retrieval_context",
    "context",
]


def citable_fields(
    test_case: Union[LLMTestCase, ConversationalTestCase],
) -> List[str]:
    if isinstance(test_case, ConversationalTestCase):
        return _TURN_FIELDS + [
            param.value
            for param in MULTI_TURN_CONTENT_PARAMS
            if getattr(test_case, param.value, None)
        ]
    if test_case._trace_dict is not None:
        return _TRACE_FIELDS
    return [
        param.value
        for param in SINGLE_TURN_CONTENT_PARAMS
        if getattr(test_case, param.value, None)
    ]


def blames_truncation_marker(
    label: str,
    description: Optional[str],
    reason: Optional[str],
    evidence_items: List[EvidenceItem],
) -> bool:
    """Whether a label about cut-off content rests on the marker added where
    long fields are shortened, rather than on the application's own output."""
    claims_truncation = _TRUNCATION_CLAIM_RE.search(
        _TRUNCATION_MARKER_RE.sub(
            "", f"{label} {description or ''} {reason or ''}"
        )
    )
    cites_marker = bool(_TRUNCATION_MARKER_RE.search(reason or "")) or (
        bool(evidence_items)
        and all(
            _TRUNCATION_TAIL_RE.search(evidence_item.quote.strip())
            for evidence_item in evidence_items
        )
    )
    return bool(claims_truncation) and cites_marker


@dataclass
class _Source:
    field: str
    texts: List[str]
    turn_index: Optional[int] = None
    span_index: Optional[int] = None


def verify_evidence(
    evidence_items: List[EvidenceItem],
    test_case: Union[LLMTestCase, ConversationalTestCase],
    prompt: "ClassificationPrompt",
    evidence_config: EvidenceConfig,
) -> List[Evidence]:
    sources = _get_sources(test_case, prompt)
    verified_evidence = []
    for evidence_item in evidence_items:
        if len(verified_evidence) == evidence_config.max_items:
            break
        quote = _TRUNCATION_TAIL_RE.sub(
            "", evidence_item.quote.strip()
        ).rstrip()[: evidence_config.max_quote_chars]
        if not quote:
            continue

        cited_sources = [
            source
            for source in sources
            if source.field == evidence_item.field
            and source.turn_index == evidence_item.turn_index
            and source.span_index == evidence_item.span_index
        ]
        matched_quote, source = _find_source(
            quote, cited_sources + sources
        ) or (None, None)
        verified_evidence.append(
            Evidence(
                field=source.field if source else evidence_item.field,
                quote=matched_quote or quote,
                note=evidence_item.note.strip(),
                turn_index=source.turn_index if source else None,
                span_index=source.span_index if source else None,
                matched=source is not None,
            )
        )
    return verified_evidence


def _find_source(
    quote: str, sources: List[_Source]
) -> Optional[Tuple[str, _Source]]:
    if len(quote) < _MIN_EVIDENCE_QUOTE_CHARS:
        return None
    for source in sources:
        matched_quote = _find_quote_in_sources(quote, source.texts)
        if matched_quote is not None:
            return matched_quote, source
    return None


def _get_sources(
    test_case: Union[LLMTestCase, ConversationalTestCase],
    prompt: "ClassificationPrompt",
) -> List[_Source]:
    shown, fields = prompt.shown, prompt.evidence_fields
    own_fields = {field: getattr(test_case, field, None) for field in fields}
    if isinstance(test_case, ConversationalTestCase):
        sources = _sources_in(own_fields, fields)
        for turn in shown if isinstance(shown, list) else []:
            sources += _sources_in(turn, fields, turn_index=turn.get("index"))
        return sources
    if test_case._trace_dict is None:
        return _sources_in(own_fields, fields)
    if isinstance(shown, str):
        return [_Source(field=field, texts=[shown]) for field in fields]
    if not isinstance(shown, dict):
        return []
    sources = _sources_in(shown, fields)
    for span in shown.get("spans") or []:
        if isinstance(span, dict):
            sources += _sources_in(span, fields, span_index=span.get("index"))
    return sources


def _sources_in(
    values: Dict[str, Any], fields: List[str], **index: Optional[int]
) -> List[_Source]:
    return [
        _Source(field=field, texts=texts, **index)
        for field in fields
        if (texts := _to_texts(values.get(field)))
    ]


def _find_quote_in_sources(
    quote: str, source_texts: List[str]
) -> Optional[str]:
    quote_variants = [quote]
    unescaped = (
        quote.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
    )
    if unescaped != quote:
        quote_variants.append(unescaped)
    for quote_variant in quote_variants:
        if any(quote_variant in text for text in source_texts):
            return quote_variant
        quote_without_whitespace = _WHITESPACE_RE.sub("", quote_variant)
        if quote_without_whitespace and any(
            quote_without_whitespace in _WHITESPACE_RE.sub("", text)
            for text in source_texts
        ):
            return quote_variant
    return None


def _to_texts(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts = [
            repr(v) if isinstance(v, ToolCall) else v
            for v in value
            if isinstance(v, (str, ToolCall))
        ]
        return [*texts, str(value), _to_json(value), serialize(value)]
    return [str(value), _to_json(value), serialize(value)]


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
