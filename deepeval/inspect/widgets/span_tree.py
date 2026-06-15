"""Left-pane span tree. Root is the current `Trace`, children are its
`root_spans` and their nested `children`."""

from __future__ import annotations

from typing import Callable, List, Optional

from rich.style import Style
from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from deepeval.inspect.types import (
    BaseSpan,
    Trace,
    TraceOrSpan,
    duration_ms,
    format_duration,
    has_failure,
    metric_counts,
)
from deepeval.inspect.widgets._styling import (
    PILL_FAIL,
    PILL_PASS,
    type_prefix,
)


# Minimum gap (in cells) between the left content (name + metric badge +
# optional ERRORED pill) and the right-aligned duration. Below this the
# right column gives up trying to right-align and just leaves the
# duration adjacent to the badge — better than overlapping in narrow
# panes.
_MIN_DURATION_GAP = 2


def _node_depth(node: TreeNode) -> int:
    """Number of ancestors above ``node`` (root = 0).

    Some Textual releases expose ``TreeNode.depth`` directly, others
    don't. Walking ``parent`` is O(depth) and stable across versions —
    trace trees are typically <10 deep so the cost is negligible.
    """

    depth = 0
    current = getattr(node, "parent", None)
    while current is not None:
        depth += 1
        current = getattr(current, "parent", None)
    return depth


def _metric_badge(node: TraceOrSpan) -> Optional[Text]:
    counts = metric_counts(node.metrics_data)
    if counts is None:
        return None
    passed, failed = counts
    badge = Text()
    if passed:
        badge.append(f" ✓ {passed} ", style=PILL_PASS)
    if failed:
        if passed:
            badge.append(" ")
        badge.append(f" ✗ {failed} ", style=PILL_FAIL)
    return badge


def _label_for(node: TraceOrSpan) -> Text:
    """`<glyph> <TAG>  <name>  <metric-badge>  <ERRORED>?`

    Duration is intentionally not baked in here — `SpanTree.render_label`
    appends it right-aligned to the pane width using the per-render
    viewport size, so we can't know the gap until paint time.
    """

    label = Text()
    fail = has_failure(node)
    name_style = "bold red" if fail else "bold"

    label.append_text(type_prefix(node))

    name = node.name or ("trace" if isinstance(node, Trace) else "<unnamed>")
    label.append(name, style=name_style)

    badge = _metric_badge(node)
    if badge is not None:
        label.append("  ")
        label.append_text(badge)

    if not isinstance(node, Trace) and (node.status or "").upper() == "ERRORED":
        label.append("  ")
        label.append(" ERRORED ", style=PILL_FAIL)
    return label


SpanFilter = Callable[[BaseSpan], bool]


class SpanTree(Tree[TraceOrSpan]):
    DEFAULT_CSS = """
    SpanTree {
        width: 30%;
        min-width: 28;
        max-width: 60;
        background: $surface;
        border-right: solid $boost;
        padding: 0 1;
    }
    SpanTree > .tree--cursor {
        background: $boost;
    }
    """

    def __init__(self, *args, **kwargs):
        # `populate(...)` replaces this bootstrap label before first paint.
        super().__init__("trace", *args, **kwargs)
        self.show_root = True
        self.guide_depth = 3

    def render_label(
        self,
        node: TreeNode[TraceOrSpan],
        base_style: Style,
        style: Style,
    ) -> Text:
        """Compose `<label>  …  <duration>` with the duration right-aligned
        to the pane's content edge.

        We can't bake the gap into ``_label_for`` because it depends on
        the live viewport width (the pane resizes when the user toggles
        sidebars) and the per-node indent (deeper rows have less room).
        Textual calls this hook once per paint per row, so the cost is
        proportional to visible rows — cheap.
        """

        # Default Textual behavior: copy the stored label, stylize, return.
        # We then layer the right-aligned duration on top.
        base = node.label.copy()
        base.stylize(style)

        data = node.data
        if not isinstance(data, (Trace, BaseSpan)):
            return base

        # Intrinsic ``dim`` style + the row's runtime ``style`` (selection,
        # hover) layered on top via ``stylize`` — matches how Textual's
        # default ``render_label`` composes selection highlight onto a
        # label that carries its own per-segment styles.
        duration = Text(format_duration(duration_ms(data)), style="dim")
        duration.stylize(style)

        # Indent estimate: each guide level eats ``guide_depth`` cells,
        # and the row always carries a 2-cell expand caret (▼/▶ + space).
        # An over-estimate is safe — it just lets the duration float a
        # cell or two left of the true right edge, which still reads as
        # "right-aligned" but never wraps.
        indent = _node_depth(node) * self.guide_depth + 2
        avail = max(0, self.content_region.width - indent)
        gap = avail - base.cell_len - duration.cell_len
        if gap < _MIN_DURATION_GAP:
            gap = _MIN_DURATION_GAP

        return base + Text(" " * gap) + duration

    def populate(
        self,
        trace: Trace,
        span_filter: Optional[SpanFilter] = None,
    ) -> None:
        """Rebuild the tree from `trace`. With a `span_filter`, non-
        matching spans are pruned but their ancestors are kept so matches
        stay reachable from the trace root."""

        self.clear()
        root = self.root
        root.data = trace
        root.set_label(_label_for(trace))

        for span in trace.root_spans:
            self._add_span(root, span, span_filter)

        root.expand_all()

    def _add_span(
        self,
        parent: TreeNode[TraceOrSpan],
        span: BaseSpan,
        span_filter: Optional[SpanFilter],
    ) -> Optional[TreeNode[TraceOrSpan]]:
        kept_children: List[BaseSpan] = []
        if span_filter is None:
            kept_children = list(span.children)
        else:
            kept_children = [
                c for c in span.children if _subtree_matches(c, span_filter)
            ]
            if not span_filter(span) and not kept_children:
                return None

        node = parent.add(_label_for(span), data=span, expand=True)
        for child in kept_children:
            self._add_span(node, child, span_filter)
        return node


def _subtree_matches(span: BaseSpan, span_filter: SpanFilter) -> bool:
    if span_filter(span):
        return True
    return any(_subtree_matches(c, span_filter) for c in span.children)
