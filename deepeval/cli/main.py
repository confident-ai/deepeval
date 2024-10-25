import typer
from typing import Optional
from rich import print
from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
from deepeval.cli.test import app as test_app
import webbrowser

app = typer.Typer(name="deepeval")
app.add_typer(test_app, name="test")


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
    # Use the confident_api_key if it is provided, otherwise proceed with existing logic
    if use_existing:
        confident_api_key = KEY_FILE_HANDLER.fetch_data(KeyValues.API_KEY)
        if confident_api_key:
            print("Using existing API key.")

    if confident_api_key:
        api_key = confident_api_key
    else:
        """Login to the DeepEval platform."""
        print("Welcome to [bold]DeepEval[/bold]!")
        print(
            "Login and grab your API key here: [link=https://app.confident-ai.com]https://app.confident-ai.com[/link] "
        )
        webbrowser.open("https://app.confident-ai.com")
        if api_key == "":
            while True:
                api_key = input("Paste your API Key: ").strip()
                if api_key:
                    break
                else:
                    print("API Key cannot be empty. Please try again.\n")

    KEY_FILE_HANDLER.write_key(KeyValues.API_KEY, api_key)
    print("Congratulations! Login successful :raising_hands: ")
    print(
        "If you are new to DeepEval, follow our quickstart tutorial here: [bold][link=https://docs.confident-ai.com/docs/getting-started]https://docs.confident-ai.com/docs/getting-started[/link][/bold]"
    )


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
