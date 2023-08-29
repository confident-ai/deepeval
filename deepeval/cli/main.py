import typer
import os
from typing import Optional
from typing_extensions import Annotated

# Rich has a few dependency issues
try:
    from rich import print
except Exception as e:
    pass
from ..key_handler import KEY_FILE_HANDLER
from .test import app as test_app
from ..api import Api
from ..constants import IMPLEMENTATION_ID_NAME

app = typer.Typer(name="deepeval")

app.add_typer(test_app, name="test")


@app.command()
def login(
    api_key: Annotated[
        str,
        typer.Option(
            help="API key to get from https://app.confident-ai.com. Required if you want to log events to the server."
        ),
    ] = "",
    project_name: Annotated[str, typer.Option(help="The name of your project")] = "",
):
    """Login to the DeepEval platform."""
    print("Welcome to [bold]DeepEval[/bold]!")
    print("Grab your API key here: https://app.confident-ai.com/")
    if api_key == "":
        api_key = KEY_FILE_HANDLER.fetch_api_key()
        api_key = typer.prompt(
            text="Paste it here (Hit enter if default is right)",
            default=api_key,
        )
    KEY_FILE_HANDLER.write_api_key(api_key)
    client = Api(api_key=api_key)
    if project_name == "":
        project_name = KEY_FILE_HANDLER.fetch_implementation_name()
        if project_name is None or project_name == "":
            project_name = "example"
        print("What is the name of your project?")
        project_name = typer.prompt(
            text="Name (Hit enter if default is right):",
            default=project_name,
        )
    KEY_FILE_HANDLER.write_data(IMPLEMENTATION_ID_NAME, project_name)
    print("Success! :raising_hands:")
    print(
        "If you are new to DeepEval, try generate a sample test: [bold]deepeval test generate test_sample.py[/bold]"
    )
    print("Run a sample test: [bold]deepeval test run test_sample.py[/bold]")


if __name__ == "__main__":
    app()
