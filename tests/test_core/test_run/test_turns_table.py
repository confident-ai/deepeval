import re
from types import SimpleNamespace
from rich.console import Console
from rich.table import Table

from deepeval.utils import format_turn, shorten


def test_turns_table_tools_column_has_no_prefix():
    table = Table(show_header=True)
    table.add_column("#")
    table.add_column("Role")
    table.add_column("Content")
    table.add_column("Tools")

    tool_names = "a, b, c"
    table.add_row("1", "assistant", shorten("hello"), shorten(tool_names, 60))

    console = Console(record=True)
    console.print("\n")
    console.print(table)
    rendered = console.export_text()
    assert "a, b, c" in rendered
    assert " | tools:" not in rendered


def test_turns_table_no_role_or_tools_duplication_with_format_turn():
    t = SimpleNamespace(
        order=1,
        role="assistant",
        content="Listing directories under /home/app and /var.",
        user_id="user-42",
        retrieval_context=["id,title", "id,text"],
        tools_called=[
            SimpleNamespace(name="fs.list"),
            SimpleNamespace(name="fs.read"),
        ],
        additional_metadata={"session_id": "sess-9"},
        comments="planner step",
    )

    table = Table(show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Role", justify="left")
    table.add_column("Details", justify="left")
    table.add_column("Tools", justify="left", no_wrap=True)

    tool_names = ", ".join(
        getattr(tc, "name", str(tc)) for tc in (t.tools_called or [])
    )
    details = format_turn(
        t, include_tools_in_header=False, include_order_role_in_header=False
    )

    table.add_row(str(t.order), t.role, details, shorten(tool_names, 60))

    console = Console(record=True)
    console.print("\n")
    console.print(table)
    rendered = console.export_text()

    # tools appear only in Tools column
    assert " | tools:" not in rendered
    assert "fs.list" in rendered and "fs.read" in rendered

    # role and order are not duplicated inside Details
    assert not re.search(r"1\.\s*assistant\b", rendered)
