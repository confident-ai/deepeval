import typer
import os

# Rich has a few dependency issues
try:
    from rich import print
except Exception as e:
    pass
from ..key_handler import KeyFileHandler
from .test import app as test_app
from ..api import Api
from ..constants import IMPLEMENTATION_ID_NAME

app = typer.Typer(name="deepeval")

app.add_typer(test_app, name="test")

handler = KeyFileHandler()


@app.command()
def login():
    """Login to the DeepEval platform."""
    print("Welcome to [bold]DeepEval[/bold]!")
    print("Grab your API key here: https://app.confident-ai.com/organization")
    api_key = input("Paste it here:")
    handler.write_api_key(api_key)
    client = Api(api_key=api_key)
    print("What is the name of your project?")
    project = input("Name:")
    handler.write_data(IMPLEMENTATION_ID_NAME, project)
    print("Success! :raising_hands:")
    print("Run: [bold]deepeval test run sample[/bold]")


if __name__ == "__main__":
    app()
