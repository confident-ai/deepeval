import os
import typer
from typing import Optional

try:
    from rich import print
except Exception as e:
    pass
from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues

app = typer.Typer(name="azure-openai")


@app.command(name="set")
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

    print(
        ":raising_hands: Congratulations! You're now using Azure OpenAI for all evals that require an LLM."
    )


@app.command(name="unset")
def unset_azure_openai_env():
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_OPENAI_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_OPENAI_ENDPOINT)
    KEY_FILE_HANDLER.remove_key(KeyValues.OPENAI_API_VERSION)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_DEPLOYMENT_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_MODEL_VERSION)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_AZURE_OPENAI)

    print(
        ":raising_hands: Congratulations! You're now using regular OpenAI for all evals that require an LLM."
    )
