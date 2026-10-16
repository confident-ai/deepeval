"""`deepeval setup` Typer command."""

import typer

from deepeval.cli.setup.launcher import (
    SetupLauncherError,
    ensure_binary,
    run_binary,
)

SETUP_HELP = (
    "Set up an evaluation for this project with a coding agent, optionally "
    "saving results to Confident AI. Downloads the latest setup wizard "
    "(macOS and Linux), verifies its checksum, caches it, and runs it. Wizard options such as --project-dir and --proj-id are passed "
    "through; run `deepeval setup -h` to list them."
)

# Click would otherwise reject the wizard's own flags as unknown options.
SETUP_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def _status(message: str) -> None:
    typer.echo(message, err=True)


def setup_command(ctx: typer.Context) -> None:
    try:
        binary = ensure_binary(notify=_status)
        code = run_binary(binary, list(ctx.args))
    except SetupLauncherError as error:
        _status(str(error))
        raise typer.Exit(1)
    raise typer.Exit(code)
