"""Top header bar: `deepeval inspect · {run_id} · {passed}✓ {failed}✗ · trace i/N`."""

from __future__ import annotations

from typing import Optional

from rich.text import Text
from textual.widgets import Static


class HeaderBar(Static):
    DEFAULT_CSS = """
    HeaderBar {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $text;
    }
    """

    def render_run_header(
        self,
        run_id: str,
        passed: Optional[int],
        failed: Optional[int],
        trace_index: int,
        trace_count: int,
        extra: Optional[str] = None,
    ) -> None:
        text = Text()
        text.append("deepeval inspect", style="bold")
        text.append(" · ")
        text.append(run_id, style="dim")
        if passed is not None or failed is not None:
            text.append(" · ")
            if passed is not None:
                text.append(f"{passed}", style="bold green")
                text.append("✓ ", style="green")
            if failed is not None:
                text.append(f"{failed}", style="bold red")
                text.append("✗", style="red")
        if trace_count > 1:
            text.append(" · ")
            text.append(f"trace {trace_index + 1}/{trace_count}", style="dim")
        if extra:
            text.append(" · ")
            text.append(extra, style="dim")
        self.update(text)
