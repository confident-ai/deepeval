from typing import Optional
from rich import print
import webbrowser
import threading
import random
import string
import socket
import typer

from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
from deepeval.cli.recommend import app as recommend_app
from deepeval.telemetry import capture_login_event, get_logged_in_with
from deepeval.cli.test import app as test_app
from deepeval.cli.server import start_server


PROD = "https://app.confident-ai.com"
LOCAL = "http://localhost:3000"

app = typer.Typer(name="deepeval")
app.add_typer(test_app, name="test")
app.add_typer(recommend_app, name="recommend")


def generate_pairing_code():
    """Generate a random pairing code."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))  # Bind to port 0 to get an available port
        return s.getsockname()[1]


@app.command()
def login(
    api_key: str = typer.Option(
        "",
        help="API key to get from https://app.confident-ai.com. Required if you want to log events to the server.",
    ),
    confident_api_key: Optional[str] = typer.Option(
        None,
        "--confident-api-key",
        "-c",
        help="Optional confident API key to bypass login.",
    ),
    use_existing: Optional[bool] = typer.Option(
        False,
        "--use-existing",
        "-u",
        help="Use the existing API key stored in the key file if present.",
    ),
):
    with capture_login_event() as span:
        # Use the confident_api_key if it is provided, otherwise proceed with existing logic
        try:
            if use_existing:
                confident_api_key = KEY_FILE_HANDLER.fetch_data(
                    KeyValues.API_KEY
                )
                if confident_api_key:
                    print("Using existing API key.")

            if confident_api_key:
                api_key = confident_api_key
            else:
                """Login to the DeepEval platform."""
                print("Welcome to :sparkles:[bold]DeepEval[/bold]:sparkles:!")

                # Start the pairing server
                port = find_available_port()
                pairing_code = generate_pairing_code()
                pairing_thread = threading.Thread(
                    target=start_server,
                    args=(pairing_code, port, PROD),
                    daemon=True,
                )
                pairing_thread.start()

                # Open web url
                login_url = f"{PROD}/pair?code={pairing_code}&port={port}"
                print(
                    f"Login and grab your API key here: [link={login_url}]{login_url}[/link] "
                )
                webbrowser.open(login_url)

                if api_key == "":
                    while True:
                        api_key = input("Paste your API Key: ").strip()
                        if api_key:
                            break
                        else:
                            print(
                                "API Key cannot be empty. Please try again.\n"
                            )

            KEY_FILE_HANDLER.write_key(KeyValues.API_KEY, api_key)
            span.set_attribute("completed", False)

            print(
                "\n🎉🥳 Congratulations! You've successfully logged in! :raising_hands: "
            )
            print(
                "You're now using DeepEval with [rgb(106,0,255)]Confident AI[/rgb(106,0,255)]. Follow our quickstart tutorial here: [bold][link=https://docs.confident-ai.com/confident-ai/confident-ai-introduction]https://docs.confident-ai.com/confident-ai/confident-ai-introduction[/link][/bold]"
            )
        except:
            span.set_attribute("completed", False)


@app.command(name="set-azure-openai")
def set_azure_openai_env(
    azure_openai_api_key: str = typer.Option(
        ..., "--openai-api-key", help="Azure OpenAI API key"
    ),
    azure_openai_endpoint: str = typer.Option(
        ..., "--openai-endpoint", help="Azure OpenAI endpoint"
    ),
    openai_api_version: str = typer.Option(
        ..., "--openai-api-version", help="OpenAI API version"
    ),
    azure_deployment_name: str = typer.Option(
        ..., "--deployment-name", help="Azure deployment name"
    ),
    azure_model_version: Optional[str] = typer.Option(
        None, "--model-version", help="Azure model version (optional)"
    ),
):
    KEY_FILE_HANDLER.write_key(
        KeyValues.AZURE_OPENAI_API_KEY, azure_openai_api_key
    )
    KEY_FILE_HANDLER.write_key(
        KeyValues.AZURE_OPENAI_ENDPOINT, azure_openai_endpoint
    )
    KEY_FILE_HANDLER.write_key(KeyValues.OPENAI_API_VERSION, openai_api_version)
    KEY_FILE_HANDLER.write_key(
        KeyValues.AZURE_DEPLOYMENT_NAME, azure_deployment_name
    )

    if azure_model_version is not None:
        KEY_FILE_HANDLER.write_key(
            KeyValues.AZURE_MODEL_VERSION, azure_model_version
        )

    KEY_FILE_HANDLER.write_key(KeyValues.USE_AZURE_OPENAI, "YES")
    KEY_FILE_HANDLER.write_key(KeyValues.USE_LOCAL_MODEL, "NO")

    print(
        ":raising_hands: Congratulations! You're now using Azure OpenAI for all evals that require an LLM."
    )


@app.command(name="set-azure-openai-embedding")
def set_azure_openai_embedding_env(
    azure_embedding_deployment_name: str = typer.Option(
        ...,
        "--embedding-deployment-name",
        help="Azure embedding deployment name",
    ),
):
    KEY_FILE_HANDLER.write_key(
        KeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME,
        azure_embedding_deployment_name,
    )
    KEY_FILE_HANDLER.write_key(KeyValues.USE_AZURE_OPENAI_EMBEDDING, "YES")
    KEY_FILE_HANDLER.write_key(KeyValues.USE_LOCAL_EMBEDDINGS, "NO")

    print(
        ":raising_hands: Congratulations! You're now using Azure OpenAI Embeddings within DeepEval."
    )


@app.command(name="unset-azure-openai")
def unset_azure_openai_env():
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_OPENAI_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_OPENAI_ENDPOINT)
    KEY_FILE_HANDLER.remove_key(KeyValues.OPENAI_API_VERSION)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_DEPLOYMENT_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_MODEL_VERSION)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_AZURE_OPENAI)

    print(
        ":raising_hands: Congratulations! You're now using regular OpenAI for all evals that require an LLM."
    )


@app.command(name="unset-azure-openai-embedding")
def unset_azure_openai_embedding_env():
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_AZURE_OPENAI_EMBEDDING)

    print(
        ":raising_hands: Congratulations! You're now using regular OpenAI embeddings for all evals that require text embeddings."
    )


#############################################
# Ollama Integration ########################
#############################################


@app.command(name="set-ollama")
def set_local_model_env(
    model_name: str = typer.Argument(..., help="Name of the Ollama model"),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "-b",
        "--base-url",
        help="Base URL for the local model API",
    ),
):
    KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_MODEL_BASE_URL, base_url)
    KEY_FILE_HANDLER.write_key(KeyValues.USE_LOCAL_MODEL, "YES")
    KEY_FILE_HANDLER.write_key(KeyValues.USE_AZURE_OPENAI, "NO")
    KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_MODEL_API_KEY, "ollama")
    print(
        ":raising_hands: Congratulations! You're now using a local Ollama model for all evals that require an LLM."
    )


@app.command(name="unset-ollama")
def unset_local_model_env():
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_BASE_URL)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_LOCAL_MODEL)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_API_KEY)
    print(
        ":raising_hands: Congratulations! You're now using regular OpenAI for all evals that require an LLM."
    )


#############################################
# Local Model Integration ###################
#############################################


@app.command(name="set-local-model")
def set_local_model_env(
    model_name: str = typer.Option(
        ..., "--model-name", help="Name of the local model"
    ),
    base_url: str = typer.Option(
        ..., "--base-url", help="Base URL for the local model API"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for the local model (if required)"
    ),
    format: Optional[str] = typer.Option(
        "json",
        "--format",
        help="Format of the response from the local model (default: json)",
    ),
):
    KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_MODEL_BASE_URL, base_url)
    if api_key:
        KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_MODEL_API_KEY, api_key)
    if format:
        KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_MODEL_FORMAT, format)
    KEY_FILE_HANDLER.write_key(KeyValues.USE_LOCAL_MODEL, "YES")
    KEY_FILE_HANDLER.write_key(KeyValues.USE_AZURE_OPENAI, "NO")
    print(
        ":raising_hands: Congratulations! You're now using a local model for all evals that require an LLM."
    )


@app.command(name="unset-local-model")
def unset_local_model_env():
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_BASE_URL)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_FORMAT)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_LOCAL_MODEL)

    print(
        ":raising_hands: Congratulations! You're now using regular OpenAI for all evals that require an LLM."
    )


#############################################
# Local Embedding Model Integration #########
#############################################


@app.command(name="set-local-embeddings")
def set_local_embeddings_env(
    model_name: str = typer.Option(
        ..., "--model-name", help="Name of the local embeddings model"
    ),
    base_url: str = typer.Option(
        ..., "--base-url", help="Base URL for the local embeddings API"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for the local embeddings (if required)"
    ),
):
    KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_EMBEDDING_MODEL_NAME, model_name)
    KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_EMBEDDING_BASE_URL, base_url)
    if api_key:
        KEY_FILE_HANDLER.write_key(KeyValues.LOCAL_EMBEDDING_API_KEY, api_key)

    KEY_FILE_HANDLER.write_key(KeyValues.USE_LOCAL_EMBEDDINGS, "YES")
    KEY_FILE_HANDLER.write_key(KeyValues.USE_AZURE_OPENAI_EMBEDDING, "NO")

    print(
        ":raising_hands: Congratulations! You're now using local embeddings for all evals that require text embeddings."
    )


@app.command(name="unset-local-embeddings")
def unset_local_embeddings_env():
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_EMBEDDING_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_EMBEDDING_BASE_URL)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_EMBEDDING_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_LOCAL_EMBEDDINGS)

    print(
        ":raising_hands: Congratulations! You're now using regular OpenAI embeddings for all evals that require text embeddings."
    )


if __name__ == "__main__":
    app()
