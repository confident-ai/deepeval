"""View-model extensions over `deepeval.tracing.api`.

Adds the nested `children` / `root_spans` fields the TUI walks for tree
rendering, leaving every other field inherited from `BaseApiSpan` /
`TraceApi` so the on-disk shape stays the source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple, Union

from pydantic import Field

from deepeval.tracing.api import BaseApiSpan, MetricData, TraceApi


class BaseSpan(BaseApiSpan):
    children: List["BaseSpan"] = Field(default_factory=list)


class Trace(TraceApi):
    # The five flat-bucket fields are overridden only to add
    # `exclude=True`: the loader pops them out of the dict before
    # validation, so dumping `Trace` round-trips the nested form
    # (`rootSpans` → `children`) instead of every span twice.
    base_spans: Optional[List[BaseSpan]] = Field(
        None, alias="baseSpans", exclude=True
    )
    agent_spans: Optional[List[BaseSpan]] = Field(
        None, alias="agentSpans", exclude=True
    )
    llm_spans: Optional[List[BaseSpan]] = Field(
        None, alias="llmSpans", exclude=True
    )
    retriever_spans: Optional[List[BaseSpan]] = Field(
        None, alias="retrieverSpans", exclude=True
    )
    tool_spans: Optional[List[BaseSpan]] = Field(
        None, alias="toolSpans", exclude=True
    )
    root_spans: List[BaseSpan] = Field(default_factory=list, alias="rootSpans")


TraceOrSpan = Union[Trace, BaseSpan]


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # 3.11+ accepts trailing `Z`; swap for `+00:00` to work on 3.9/3.10.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def duration_ms(node: TraceOrSpan) -> Optional[float]:
    start = _parse_iso(node.start_time)
    end = _parse_iso(node.end_time)
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000.0


def format_duration(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def metric_counts(
    metrics: Optional[List[MetricData]],
) -> Optional[Tuple[int, int]]:
    if not metrics:
        return None
    passed = sum(1 for m in metrics if m.success)
    return passed, len(metrics) - passed


def has_failure(node: TraceOrSpan) -> bool:
    if (node.status or "").upper() == "ERRORED":
        return True
    if node.metrics_data and any(not m.success for m in node.metrics_data):
        return True
    return False


def iter_descendants(span: BaseSpan):
    for child in span.children:
        yield child
        yield from iter_descendants(child)


def all_spans(trace: Trace) -> List[BaseSpan]:
    out: List[BaseSpan] = []
    for root in trace.root_spans:
        out.append(root)
        out.extend(iter_descendants(root))
    return out
