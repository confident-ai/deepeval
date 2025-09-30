from rich.console import Console
from rich.table import Table
from deepeval.utils import shorten


def test_turns_table_tools_column_has_no_prefix():
    table = Table(show_header=True)
    table.add_column("#")
    table.add_column("Role")
    table.add_column("Content")
    table.add_column("Tools")

    tool_names = "a, b, c"
    table.add_row("1", "assistant", shorten("hello"), shorten(tool_names, 60))

    console = Console(record=True)
    console.print(table)
    rendered = console.export_text()
    assert "a, b, c" in rendered
    assert " | tools:" not in rendered
