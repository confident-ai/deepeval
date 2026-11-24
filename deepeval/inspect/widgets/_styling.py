"""Shared span-type glyphs/tags/colors and pass/fail pill styles.

Both panes import from here so jumping between the tree and the details
view keeps the same visual identity for each span type.
"""

from __future__ import annotations

from typing import Tuple

from rich.text import Text

from deepeval.inspect.types import Trace, TraceOrSpan


# `(glyph, tag, rich style)` per span type. Tags are full words rather
# than abbreviations because the tree pane is wide enough to spell them
# out, and "RETRIEVER" reads instantly while "RET" trips users into
# mentally expanding it.
#
# Explicit hex values rather than named ANSI colors: named colors get
# theme-remapped and `dim` collapses to invisible on some palettes;
# truecolor hex survives every theme and degrades to the nearest 256-
# color match on older terminals.
TYPE_STYLE: dict[str, Tuple[str, str, str]] = {
    "trace": ("◆", "TRACE", "bold #8be9fd"),  # cyan
    "base": ("▪", "BASE", "#a8a8a8"),  # mid-gray
    "agent": ("◉", "AGENT", "bold #ff79c6"),  # pink
    "llm": ("✦", "LLM", "bold #f1fa8c"),  # yellow
    "retriever": ("⤓", "RETRIEVER", "bold #bd93f9"),  # purple
    "tool": ("⚒", "TOOL", "bold #50fa7b"),  # green
}


def type_style(node: TraceOrSpan) -> Tuple[str, str, str]:
    if isinstance(node, Trace):
        return TYPE_STYLE["trace"]
    return TYPE_STYLE.get(node.type, TYPE_STYLE["base"])


def type_prefix(node: TraceOrSpan) -> Text:
    """`◆ TRACE ` styled, ready to append into a Rich `Text` row."""

    glyph, tag, style = type_style(node)
    text = Text()
    text.append(f"{glyph} ", style=style)
    text.append(tag, style=style)
    text.append(" ")
    return text


# Pill foregrounds picked for luminance contrast: dark text on high-
# luminance backgrounds (green/yellow), light text on mid-luminance (red).
PILL_PASS = "bold #1a1a2e on #50fa7b"
PILL_FAIL = "bold #f8f8f2 on #ff5555"
PILL_WARN = "bold #1a1a2e on #f1fa8c"
