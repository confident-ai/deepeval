"""Help overlay listing every keybinding. Shown by `?`, dismissed by escape."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static


_HELP_ROWS = [
    ("↑ ↓ / k j", "move selection in the tree"),
    ("h / l", "go to parent / select child in the tree"),
    ("← → / n p", "cycle to previous / next trace"),
    ("enter", "focus the details pane"),
    ("/", "filter the tree by span name"),
    ("escape", "clear the search filter"),
    ("y", "copy the selected node as JSON to clipboard"),
    ("Y", "copy the entire trace as JSON to clipboard"),
    ("?", "toggle this help"),
    ("q / ctrl+c", "quit"),
]


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
        background: $background 75%;
    }
    HelpScreen > Container {
        width: 60;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    HelpScreen .help-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    HelpScreen .help-row {
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("Keybindings", classes="help-title")
            for keys, desc in _HELP_ROWS:
                row = Text()
                row.append(f"{keys:<14}", style="bold cyan")
                row.append(desc)
                yield Static(row, classes="help-row")

    def action_dismiss(self, _result: object = None) -> None:
        self.app.pop_screen()
