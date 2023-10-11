import typer
from typing_extensions import Annotated

# Rich has a few dependency issues
try:
    from rich import print
except Exception as e:
    pass
from ..api import Api
from ..constants import IMPLEMENTATION_ID_NAME
from ..key_handler import KEY_FILE_HANDLER
from .test import app as test_app

app = typer.Typer(name="deepeval")

app.add_typer(test_app, name="test")


@app.command()
def login(
    api_key: Annotated[
        str,
        typer.Option(
            help="API key to get from https://app.confident-ai.com. Required if you want to log events to the server."
        ),
    ] = ""
):
    """Login to the DeepEval platform."""
    print("Welcome to [bold]DeepEval[/bold]!")
    print(
        "Grab your API key here: [link=https://app.confident-ai.com]https://app.confident-ai.com[/link] "
    )
    if api_key == "":
        while True:
            api_key = input("Paste your API Key: ").strip()
            if api_key:
                break
            else:
                print("API Key cannot be empty. Please try again.\n")
    KEY_FILE_HANDLER.write_api_key(api_key)
    client = Api(api_key=api_key)
    print("Success! :raising_hands:")
    print(
        "If you are new to DeepEval, try generate a sample test: [bold]deepeval test generate --output-file test_sample.py[/bold]"
    )
    print("Run a sample test: [bold]deepeval test run test_sample.py[/bold]")


@app.command()
def switch(
    implementation_name: Annotated[
        str, typer.Option(help="The name of the project you want to switch to")
    ] = "",
):
    """Switch to a different project on the DeepEval platform."""
    if implementation_name == "":
        print("You must provide a project name to switch to.")
    else:
        KEY_FILE_HANDLER.write_data(IMPLEMENTATION_ID_NAME, implementation_name)
        print(f"Switched to project: {implementation_name}")


if __name__ == "__main__":
    app()
