"""The root Typer app, kept apart from `main` so command modules can register
onto it without importing `main` back."""

import typer

app = typer.Typer(name="deepeval", no_args_is_help=True)
