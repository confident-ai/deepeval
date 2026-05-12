"""Textual `App` for `deepeval inspect`.

Layout: HeaderBar · [SpanTree | DetailsPane] · SearchBar (toggle) · Footer.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer

from deepeval.inspect.loader import run_id_from_path, summarize_test_run
from deepeval.inspect.types import (
    BaseSpan,
    Trace,
    TraceOrSpan,
    all_spans,
)
from deepeval.inspect.widgets.details import DetailsPane
from deepeval.inspect.widgets.header_bar import HeaderBar
from deepeval.inspect.widgets.help_modal import HelpScreen
from deepeval.inspect.widgets.search_bar import SearchBar
from deepeval.inspect.widgets.span_tree import SpanTree


class InspectApp(App[None]):
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", show=False),
        # Tree navigation aliases. Tree binds up/down out of the box.
        Binding("j", "tree_cursor('down')", show=False),
        Binding("k", "tree_cursor('up')", show=False),
        Binding("h", "tree_cursor('left')", show=False),
        Binding("l", "tree_cursor('right')", show=False),
        # Trace cycling. `priority=True` beats the focused Tree's own
        # left/right (which would collapse/expand nodes). `check_action`
        # below makes these inert while SearchBar is focused so
        # left/right still work for in-input cursor editing.
        Binding("left", "cycle_trace(-1)", "Prev trace", priority=True),
        Binding("right", "cycle_trace(1)", "Next trace", priority=True),
        Binding("n", "cycle_trace(1)", show=False),
        Binding("p", "cycle_trace(-1)", show=False),
        Binding("slash", "toggle_search", "Search"),
        Binding("y", "yank_node", "Yank node"),
        Binding("shift+y", "yank_trace", "Yank trace"),
        Binding("question_mark", "toggle_help", "Help"),
    ]

    def __init__(
        self,
        traces: List[Trace],
        source_path: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not traces:
            raise ValueError("InspectApp requires at least one Trace.")
        self.traces = traces
        self.source_path = str(source_path)
        self.current_trace_index = 0
        self._search_filter: Optional[Callable[[BaseSpan], bool]] = None
        self._run_summary = summarize_test_run(self.source_path) or {}

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")
        with Horizontal(id="main-split"):
            yield SpanTree(id="span-tree")
            yield DetailsPane(id="details-pane")
        yield SearchBar()
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_tree()
        self._refresh_header()
        self.query_one(SpanTree).focus()

    async def action_cycle_trace(self, delta: int) -> None:
        n = len(self.traces)
        if n <= 1:
            self.bell()
            return
        self.current_trace_index = (self.current_trace_index + delta) % n
        # A search that matched spans in trace A is rarely useful on
        # trace B; clearing it avoids surprising "stickiness".
        self._search_filter = None
        search = self.query_one(SearchBar)
        search.value = ""
        search.display = False
        self._refresh_tree()
        self._refresh_header()

        # Reset view to the new trace's root row. We move the cursor via
        # `cursor_line = 0` rather than `select_node(tree.root)` because
        # `select_node` calls `action_select_cursor`, which toggles the
        # node's expanded state when `auto_expand=True` — collapsing the
        # root we just expanded. Writing `cursor_line` skips the toggle.
        # We also call `details.show()` explicitly to stay independent
        # of NodeHighlighted event timing across Textual versions.
        tree = self.query_one(SpanTree)
        if not tree.root.is_expanded:
            tree.root.expand()
        tree.cursor_line = 0
        tree.scroll_to_node(tree.root, animate=False)
        details = self.query_one(DetailsPane)
        await details.show(self._current_trace())

    def check_action(self, action: str, parameters: tuple) -> Optional[bool]:
        # Returning False makes the binding inert AND hides it from the
        # footer; the key falls through to the focused SearchBar so
        # left/right move the input cursor instead of cycling traces.
        if action == "cycle_trace" and isinstance(self.focused, SearchBar):
            return False
        return True

    def action_toggle_search(self) -> None:
        search = self.query_one(SearchBar)
        search.display = not search.display
        if search.display:
            search.focus()
        else:
            self.finish_search()

    @on(SearchBar.Changed)
    def on_search_changed(self, event: SearchBar.Changed) -> None:
        query = event.value.strip().lower()
        if not query:
            self._search_filter = None
        else:
            self._search_filter = lambda span: bool(
                span.name and query in span.name.lower()
            )
        self._refresh_tree()

    @on(SearchBar.Submitted)
    def on_search_submitted(self, _event: SearchBar.Submitted) -> None:
        # Enter keeps the filter but hides the bar and refocuses the
        # tree for hands-off navigation.
        self.query_one(SearchBar).display = False
        self.query_one(SpanTree).focus()

    def finish_search(self) -> None:
        self._search_filter = None
        self._refresh_tree()
        self.query_one(SpanTree).focus()

    def action_yank_node(self) -> None:
        node = self._selected_node()
        if node is None:
            self.notify("Nothing to yank.", severity="warning")
            return
        self._copy_to_clipboard(
            node.model_dump_json(by_alias=True, indent=2),
            label=f"{type(node).__name__} {getattr(node, 'name', None) or ''}",
        )

    def action_yank_trace(self) -> None:
        trace = self._current_trace()
        self._copy_to_clipboard(
            trace.model_dump_json(by_alias=True, indent=2),
            label=f"trace {trace.uuid[:8]}",
        )

    def _copy_to_clipboard(self, body: str, label: str) -> None:
        try:
            import pyperclip
        except ImportError:
            self.notify(
                "pyperclip not installed. Run "
                "`pip install 'deepeval[inspect]'`.",
                severity="error",
            )
            return
        try:
            pyperclip.copy(body)
        except pyperclip.PyperclipException as e:
            # Headless / SSH sessions without a clipboard provider land
            # here; surface a message instead of silently dropping.
            self.notify(f"Clipboard unavailable: {e}", severity="error")
            return
        self.notify(f"Yanked {label} ({len(body)} chars).")

    def action_toggle_help(self) -> None:
        if isinstance(self.screen, HelpScreen):
            self.pop_screen()
        else:
            self.push_screen(HelpScreen())

    def action_tree_cursor(self, direction: str) -> None:
        tree = self.query_one(SpanTree)
        mapping = {
            "down": tree.action_cursor_down,
            "up": tree.action_cursor_up,
            "left": tree.action_cursor_parent,
            "right": tree.action_select_cursor,
        }
        action = mapping.get(direction)
        if action is not None:
            action()

    @on(SpanTree.NodeHighlighted)
    async def on_tree_node_highlighted(
        self, event: SpanTree.NodeHighlighted
    ) -> None:
        details = self.query_one(DetailsPane)
        data = event.node.data
        if isinstance(data, (Trace, BaseSpan)):
            await details.show(data)
        else:
            await details.show(None)

    def _current_trace(self) -> Trace:
        return self.traces[self.current_trace_index]

    def _selected_node(self) -> Optional[TraceOrSpan]:
        tree = self.query_one(SpanTree)
        node = tree.cursor_node
        if node is None:
            return None
        if isinstance(node.data, (Trace, BaseSpan)):
            return node.data
        return None

    def _refresh_tree(self) -> None:
        tree = self.query_one(SpanTree)
        tree.populate(self._current_trace(), span_filter=self._search_filter)

    def _refresh_header(self) -> None:
        header = self.query_one(HeaderBar)
        summary = self._run_summary
        header.render_run_header(
            run_id=run_id_from_path(self.source_path),
            passed=summary.get("test_passed"),
            failed=summary.get("test_failed"),
            trace_index=self.current_trace_index,
            trace_count=len(self.traces),
            extra=f"{len(all_spans(self._current_trace()))} spans",
        )
