import typer
from typing_extensions import Annotated

# Rich has a few dependency issues
try:
    from rich import print
except Exception as e:
    pass
from deepeval.api import Api
from deepeval.key_handler import KEY_FILE_HANDLER
from deepeval.cli.test import app as test_app
import webbrowser

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
    webbrowser.open("https://app.confident-ai.com")
    if api_key == "":
        while True:
            api_key = input("Paste your API Key: ").strip()
            if api_key:
                break
            else:
                print("API Key cannot be empty. Please try again.\n")
    KEY_FILE_HANDLER.write_api_key(api_key)
    client = Api(api_key=api_key)
    print("Congratulations! Login successful :raising_hands: ")
    print(
        "If you are new to DeepEval, follow our quickstart tutorial here: [bold][link=https://docs.confident-ai.com/docs/getting-started]https://docs.confident-ai.com/docs/getting-started[/link][/bold]"
    )


if __name__ == "__main__":
    app()
