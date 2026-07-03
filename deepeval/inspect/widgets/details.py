"""Right-pane details view.

Section order (top-down): header line · metric pill badges · meta strip
· metrics (full reasoning) · Confident AI CTA · type-specific details ·
input · output · optional payloads · raw JSON (collapsed).
"""

from __future__ import annotations

from typing import Any, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Markdown, Static

from deepeval.utils import serialize_to_json
from deepeval.inspect.types import (
    BaseSpan,
    MetricData,
    Trace,
    TraceOrSpan,
    duration_ms,
    format_duration,
)
from deepeval.inspect.widgets._styling import (
    PILL_FAIL,
    PILL_PASS,
    PILL_WARN,
    type_prefix,
)


# Matches the TRACE tag so the eye learns "cyan = structure markers".
_HEADER_ACCENT = "#8be9fd"
_CTA_ACCENT = "#bd93f9"


class DetailsPane(VerticalScroll):
    """Full rebuild on every selection — incremental updates aren't
    worth the complexity given how heterogeneous the section list is."""

    DEFAULT_CSS = """
    DetailsPane {
        width: 60%;
        background: $surface;
        padding: 0 2;
    }
    DetailsPane > .details-header {
        padding: 1 0 0 0;
    }
    DetailsPane > .details-divider {
        color: $boost;
        padding: 1 0 0 0;
    }
    DetailsPane > .details-section-label {
        color: $accent;
        text-style: bold;
    }
    DetailsPane > .details-empty {
        color: $text-muted;
        padding: 2 0;
    }
    DetailsPane Markdown {
        margin: 0 0 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Select a node in the tree to inspect.",
            classes="details-empty",
            id="details-empty",
        )

    async def show(self, node: Optional[TraceOrSpan]) -> None:
        await self.remove_children()
        if node is None:
            await self.mount(
                Static(
                    "Select a node in the tree to inspect.",
                    classes="details-empty",
                )
            )
            return

        sections: List[Any] = []
        sections.extend(_header_section(node))
        sections.extend(_metric_badges_section(node))
        sections.extend(_meta_strip_section(node))
        sections.extend(_metrics_section(node))
        sections.extend(_confident_cta_section(node))
        sections.extend(_type_specific_section(node))
        sections.extend(_input_section(node))
        sections.extend(_output_section(node))
        sections.extend(_optional_sections(node))
        sections.extend(_raw_json_section(node))

        await self.mount_all(sections)
        # Otherwise jumping from a long trace's tail to a short span
        # lands mid-scroll on empty area below the new content.
        self.scroll_home(animate=False)


def _divider(label: str) -> Static:
    """Section header rendered as `▌ LABEL`."""

    text = Text()
    text.append("▌ ", style=f"bold {_HEADER_ACCENT}")
    text.append(label.upper(), style=f"bold {_HEADER_ACCENT}")
    return Static(text, classes="details-divider")


def _header_section(node: TraceOrSpan) -> List[Any]:
    header = Text()
    name = node.name if not isinstance(node, Trace) else (node.name or "trace")
    name_style = (
        "bold red" if (node.status or "").upper() == "ERRORED" else "bold"
    )
    header.append_text(type_prefix(node))
    header.append(name or "<unnamed>", style=name_style)
    header.append("  ·  ", style="dim")
    header.append(format_duration(duration_ms(node)), style="dim")
    header.append("  ·  ", style="dim")
    status = (node.status or "").upper()
    if status == "SUCCESS":
        header.append(" SUCCESS ", style=PILL_PASS)
    elif status == "ERRORED":
        header.append(" ERRORED ", style=PILL_FAIL)
    elif status:
        header.append(f" {status} ", style=PILL_WARN)
    else:
        header.append("—", style="dim")

    if isinstance(node, BaseSpan) and node.type == "llm":
        tokens = _format_tokens(node)
        if tokens:
            header.append("  ·  ", style="dim")
            header.append(tokens, style="dim")
        cost = _estimate_cost(node)
        if cost is not None:
            header.append("  ·  ", style="dim")
            header.append(f"${cost:.4f}", style="dim")
        if node.model:
            header.append("  ·  ", style="dim")
            header.append(node.model, style="magenta")

    return [Static(header, classes="details-header")]


def _format_tokens(span: BaseSpan) -> Optional[str]:
    if span.input_token_count is None and span.output_token_count is None:
        return None
    i = int(span.input_token_count or 0)
    o = int(span.output_token_count or 0)
    return f"tokens {i} → {o}"


def _estimate_cost(span: BaseSpan) -> Optional[float]:
    """Returns `None` unless both rates AND token counts are available;
    partial breakdowns clutter the header without being useful."""

    if span.cost_per_input_token is None or span.cost_per_output_token is None:
        return None
    if span.input_token_count is None and span.output_token_count is None:
        return None
    return (span.cost_per_input_token or 0) * (span.input_token_count or 0) + (
        span.cost_per_output_token or 0
    ) * (span.output_token_count or 0)


def _metric_badges_section(node: TraceOrSpan) -> List[Any]:
    metrics = node.metrics_data or []
    if not metrics:
        return []
    badges = Text()
    for i, m in enumerate(metrics):
        if i:
            badges.append("  ")
        pill = PILL_PASS if m.success else PILL_FAIL
        glyph = "✓" if m.success else "✗"
        text = f" {glyph} {m.name}"
        if m.score is not None:
            text += f": {m.score:.2f}"
        text += " "
        badges.append(text, style=pill)
    return [Static(badges)]


def _meta_strip_section(node: TraceOrSpan) -> List[Any]:
    """Tags · UUID chips, plus a Metadata block.

    Labels (``Tags:``, ``Metadata:``) are rendered unconditionally with an
    explicit ``None`` placeholder when the field is empty — eyes scan
    detail panes faster when the schema is stable across nodes, rather
    than having sections appear and disappear based on whether a value
    happens to be set. ``Tags`` is trace-only because only ``Trace``
    carries the field; ``Metadata`` applies to every node.
    """

    out: List[Any] = []
    line = Text()
    parts: List[Text] = []

    if isinstance(node, Trace):
        chip = Text()
        chip.append("Tags: ", style="dim")
        tags = getattr(node, "tags", None)
        if tags:
            chip.append(", ".join(tags), style="cyan")
        else:
            chip.append("None", style="dim italic")
        parts.append(chip)

    if getattr(node, "uuid", None):
        chip = Text()
        chip.append("UUID: ", style="dim")
        chip.append(node.uuid, style="dim cyan")
        parts.append(chip)

    if parts:
        for i, chip in enumerate(parts):
            if i:
                line.append("  ·  ", style="dim")
            line.append_text(chip)
        out.append(Static(line))

    out.append(_kv_block("Metadata", getattr(node, "metadata", None)))

    return out


def _kv_block(label: str, data: Any) -> Static:
    text = Text()
    text.append(f"{label}: ", style="dim")
    if data:
        text.append(serialize_to_json(data, indent=2), style="dim")
    else:
        # Explicit None placeholder so the row reads as "no value" rather
        # than the label hanging with no value at all.
        text.append("None", style="dim italic")
    return Static(text)


def _confident_cta_section(node: TraceOrSpan) -> List[Any]:
    """Banner promoting Confident AI. Rendered on every node — repetition
    is fine for a one-line evergreen CTA.

    The CTA points users at ``deepeval login`` rather than a URL because
    (a) the login flow already opens the right page in the user's browser
    and (b) a typed command works in any terminal, while OSC-8 hyperlinks
    rely on terminal support.
    """

    body = Text()
    body.append("Upload traces to Confident AI for free. Run ")
    body.append("deepeval login", style=f"bold {_CTA_ACCENT}")

    return [
        _cta_divider("❤️ Save traces to the cloud"),
        Static(body, classes="details-cta"),
    ]


def _cta_divider(label: str) -> Static:
    text = Text()
    text.append("▌ ", style=f"bold {_CTA_ACCENT}")
    text.append(label.upper(), style=f"bold {_CTA_ACCENT}")
    return Static(text, classes="details-divider")


def _metrics_section(node: TraceOrSpan) -> List[Any]:
    """Always rendered — an explicit "no metrics" placeholder beats
    silently hiding the section, which users read as a bug."""

    out: List[Any] = [_divider("Metrics")]
    metrics = node.metrics_data or []
    if not metrics:
        hint = Text(
            "No metrics evaluated for this node.",
            style="dim italic",
        )
        out.append(Static(hint))
        return out
    for m in metrics:
        out.extend(_metric_block(m))
    return out


def _metric_block(metric: MetricData) -> List[Any]:
    headline = Text()
    if metric.success:
        headline.append(" PASS ", style=PILL_PASS)
    else:
        headline.append(" FAIL ", style=PILL_FAIL)
    headline.append("  ")
    headline.append(metric.name, style="bold")
    if metric.score is not None:
        score_style = "bold green" if metric.success else "bold red"
        headline.append(f"  {metric.score:.2f}", style=score_style)
    headline.append(f" / {metric.threshold:.2f}", style="dim")
    if metric.evaluation_model:
        headline.append(f"  ({metric.evaluation_model})", style="dim italic")
    out: List[Any] = [Static(headline)]
    # LLM-judge reasons commonly include headings / bullets / backticks,
    # so render as Markdown rather than plain text.
    if metric.reason:
        out.append(Markdown(metric.reason))
    if metric.error:
        err = Text()
        err.append("Error: ", style="bold red")
        err.append(metric.error)
        out.append(Static(err))
    return out


def _type_specific_section(node: TraceOrSpan) -> List[Any]:
    """Placed above Input/Output so these small fixed fields stay
    visible without scrolling past potentially huge I/O payloads."""

    if not isinstance(node, BaseSpan):
        return []
    if node.type == "llm":
        return _llm_block(node)
    if node.type == "retriever":
        return _retriever_block(node)
    if node.type == "tool":
        return _tool_block(node)
    if node.type == "agent":
        return _agent_block(node)
    return []


def _llm_block(span: BaseSpan) -> List[Any]:
    rows: List[tuple[str, Any]] = []
    if span.model:
        rows.append(("model", span.model))
    if span.provider:
        rows.append(("provider", span.provider))
    tokens = _format_tokens(span)
    if tokens:
        rows.append(("usage", tokens))
    if (
        span.cost_per_input_token is not None
        or span.cost_per_output_token is not None
    ):
        rows.append(
            (
                "rates",
                f"in ${span.cost_per_input_token or 0:.8f} · "
                f"out ${span.cost_per_output_token or 0:.8f}",
            )
        )
    cost = _estimate_cost(span)
    if cost is not None:
        rows.append(("cost", f"${cost:.6f}"))
    if not rows:
        return []
    return [_divider("LLM Details"), _kv_table(rows)]


def _retriever_block(span: BaseSpan) -> List[Any]:
    rows: List[tuple[str, Any]] = []
    if span.embedder:
        rows.append(("embedder", span.embedder))
    if span.top_k is not None:
        rows.append(("top_k", span.top_k))
    if span.chunk_size is not None:
        rows.append(("chunk_size", span.chunk_size))
    if not rows:
        return []
    return [_divider("Retriever Details"), _kv_table(rows)]


def _tool_block(span: BaseSpan) -> List[Any]:
    if not span.description:
        return []
    return [
        _divider("Tool Details"),
        _kv_table([("description", span.description)]),
    ]


def _agent_block(span: BaseSpan) -> List[Any]:
    rows: List[tuple[str, Any]] = []
    if span.available_tools:
        rows.append(("available_tools", ", ".join(span.available_tools)))
    if span.agent_handoffs:
        rows.append(("agent_handoffs", ", ".join(span.agent_handoffs)))
    if not rows:
        return []
    return [_divider("Agent Details"), _kv_table(rows)]


def _kv_table(rows: List[tuple[str, Any]]) -> Static:
    text = Text()
    for i, (k, v) in enumerate(rows):
        if i:
            text.append("\n")
        text.append(f"{k:<16}", style="dim")
        text.append(str(v))
    return Static(text)


def _input_section(node: TraceOrSpan) -> List[Any]:
    if node.input is None or node.input == "":
        return []
    return [_divider("Input"), _payload_widget(node.input)]


def _output_section(node: TraceOrSpan) -> List[Any]:
    if node.output is None or node.output == "":
        return []
    return [_divider("Output"), _payload_widget(node.output)]


def _payload_widget(value: Any) -> Static:
    if isinstance(value, str):
        content = value
    elif isinstance(value, (dict, list)):
        content = serialize_to_json(value, indent=2)
    else:
        content = repr(value)
    # Wrap in Text so Textual's Static doesn't parse `[...]` in the
    # content as Rich markup (e.g. `repr([ToolCall(...)])` would crash).
    return Static(Text(content), classes="details-payload")


def _optional_sections(node: TraceOrSpan) -> List[Any]:
    out: List[Any] = []
    rc = node.retrieval_context
    if rc:
        out.append(_divider("Retrieval Context"))
        for i, chunk in enumerate(rc):
            block = Text()
            block.append(f"[{i}] ", style="dim")
            block.append(chunk)
            out.append(Static(block))
    if node.tools_called:
        out.append(_divider("Tools Called"))
        out.append(_payload_widget(node.tools_called))
    if node.expected_output:
        out.append(_divider("Expected Output"))
        out.append(_payload_widget(node.expected_output))
    if node.expected_tools:
        out.append(_divider("Expected Tools"))
        out.append(_payload_widget(node.expected_tools))
    return out


def _raw_json_section(node: TraceOrSpan) -> List[Any]:
    try:
        body = node.model_dump_json(by_alias=True, indent=2)
    except Exception as e:
        body = f"<failed to dump JSON: {e}>"
    # Same Static-markup-parse hazard as _payload_widget — JSON arrays
    # start with `[`, which Textual would otherwise treat as a markup tag.
    return [
        Collapsible(
            Static(Text(body)),
            title="Raw JSON",
            collapsed=True,
        )
    ]
