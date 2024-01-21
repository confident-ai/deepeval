from typing import Union

from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.console import Console
from rich.progress import Progress, BarColumn, SpinnerColumn, TextColumn


class RichManager:
    def __init__(self, show_table: bool, total_train_epochs: int) -> None:
        """
        Initialize RichManager.

        Args:
            show_table (bool): Flag to show or hide the table.
            total_train_epochs (int): Total number of training epochs.
        """
        self.show_table = show_table
        self.total_train_epochs = total_train_epochs
        self.console = Console()
        self.live = Live(auto_refresh=True, console=self.console)
        self.train_bar_started = False

        self.progress_bar_columns = [
            TextColumn(
                "{task.description} [progress.percentage][green][{task.percentage:>3.1f}%]:",
                justify="right",
            ),
            BarColumn(),
            TextColumn(
                "[green][ {task.completed}/{task.total} epochs ]",
                justify="right",
            ),
        ]
        self.spinner_columns = [
            TextColumn("{task.description}", justify="right"),
            SpinnerColumn(spinner_name="simpleDotsScrolling"),
        ]

        self.empty_column = Text("\n")

    def _initialize_progress_trackers(self) -> None:
        """
        Initialize progress trackers (progress and spinner columns).
        """
        self.progress = Progress(*self.progress_bar_columns, auto_refresh=False)
        self.spinner = Progress(*self.spinner_columns)

        self.progress_task = self.progress.add_task(
            "Train Progress", total=self.total_train_epochs
        )
        self.spinner_task = self.spinner.add_task("Initializing")

        column_list = [self.spinner, self.progress, self.empty_column]
        column_list.insert(0, Table()) if self.show_table else None

        column = Columns(column_list, equal=True, expand=True)
        self.live.update(column, refresh=True)

    def change_spinner_text(self, text: str) -> None:
        """
        Change the text displayed in the spinner.

        Args:
            text (str): Text to be displayed in the spinner.
        """
        self.spinner.reset(self.spinner_task, description=text)

    def stop(self) -> None:
        """Stop the live display."""
        self.live.stop()

    def start(self) -> None:
        """Start the live display and initialize progress trackers."""
        self.live.start()
        self._initialize_progress_trackers()

    def update(self, column: Columns) -> None:
        """
        Update the live display with a new column.

        Args:
            column (Columns): New column to be displayed.
        """
        self.live.update(column, refresh=True)

    def create_column(self) -> Union[Columns, Table]:
        """
        Create a new column with an optional table.

        Returns:
            Tuple[Columns, Table]: Tuple containing the new column and an optional table.
        """
        new_table = Table()

        column_list = [self.spinner, self.progress, self.empty_column]
        column_list.insert(0, new_table) if self.show_table else None

        column = Columns(column_list, equal=True, expand=True)
        return column, new_table

    def advance_progress(self) -> None:
        """Advance the progress tracker."""
        if not self.train_bar_started:
            self.progress.start()
            self.train_bar_started = True
        self.progress.update(self.progress_task, advance=1)
