"""Bottom-of-screen search input, toggled by `/`."""

from __future__ import annotations

from textual.binding import Binding
from textual.widgets import Input


class SearchBar(Input):
    DEFAULT_CSS = """
    SearchBar {
        dock: bottom;
        height: 1;
        background: $surface;
        border: none;
    }
    SearchBar:focus {
        border: none;
    }
    """

    BINDINGS = [
        Binding("escape", "hide_and_clear", "Cancel search"),
    ]

    def __init__(self, **kwargs):
        super().__init__(
            placeholder="/  filter spans by name (Esc to clear)…",
            id="search-bar",
            **kwargs,
        )
        # Mounted-but-hidden so `/` toggles `display` instead of re-mounting.
        self.display = False

    def action_hide_and_clear(self) -> None:
        self.value = ""
        self.display = False
        if hasattr(self.app, "finish_search"):
            self.app.finish_search()
