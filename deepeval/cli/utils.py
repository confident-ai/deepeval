from rich import print
import webbrowser
import pyfiglet
from typing import Optional
from opentelemetry.trace import Span

from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
from deepeval.test_run.test_run import (
    global_test_run_manager,
)

PROD = "https://app.confident-ai.com"


def render_login_message():
    print(
        f"🥳 Welcome to [rgb(106,0,255)]Confident AI[/rgb(106,0,255)], the DeepEval cloud platform 🏡❤️"
    )
    print("")
    print(pyfiglet.Figlet(font="big_money-ne").renderText("DeepEval Cloud"))


def upload_and_open_link(_span: Span):
    last_test_run_data = global_test_run_manager.get_latest_test_run_data()
    if last_test_run_data:
        confident_api_key = KEY_FILE_HANDLER.fetch_data(KeyValues.API_KEY)
        if confident_api_key == "" or confident_api_key is None:
            render_login_message()

            print(
                f"🔑 You'll need to get an API key at [link={PROD}]{PROD}[/link] to view your results (free)"
            )
            webbrowser.open(PROD)
            while True:
                confident_api_key = input("🔐 Enter your API Key: ").strip()
                if confident_api_key:
                    KEY_FILE_HANDLER.write_key(
                        KeyValues.API_KEY, confident_api_key
                    )
                    print(
                        "\n🎉🥳 Congratulations! You've successfully logged in! :raising_hands: "
                    )
                    _span.set_attribute("completed", True)
                    break
                else:
                    print("❌ API Key cannot be empty. Please try again.\n")

        print(f"📤 Uploading test run to Confident AI...")
        global_test_run_manager.post_test_run(last_test_run_data)
    else:
        print(
            "❌ No test run found in cache. Run 'deepeval login' + an evaluation to get started 🚀."
        )

def clear_evaluation_model_keys():
    # Azure Open AI
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_AZURE_OPENAI)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_OPENAI_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_OPENAI_ENDPOINT)
    KEY_FILE_HANDLER.remove_key(KeyValues.OPENAI_API_VERSION)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_DEPLOYMENT_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_MODEL_VERSION)
    # Local Model
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_LOCAL_MODEL)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_BASE_URL)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_MODEL_FORMAT)
    # Gemini
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_GEMINI_MODEL)
    KEY_FILE_HANDLER.remove_key(KeyValues.GEMINI_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.GOOGLE_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.GOOGLE_GENAI_USE_VERTEXAI)
    KEY_FILE_HANDLER.remove_key(KeyValues.GOOGLE_CLOUD_PROJECT)
    KEY_FILE_HANDLER.remove_key(KeyValues.GOOGLE_CLOUD_LOCATION)
    # LiteLLM
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_LITELLM)
    KEY_FILE_HANDLER.remove_key(KeyValues.LITELLM_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.LITELLM_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.LITELLM_API_BASE)
    # OpenAI
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_OPENAI_MODEL)
    KEY_FILE_HANDLER.remove_key(KeyValues.OPENAI_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.OPENAI_COST_PER_INPUT_TOKEN)
    KEY_FILE_HANDLER.remove_key(KeyValues.OPENAI_COST_PER_OUTPUT_TOKEN)
    # Moonshot
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_MOONSHOT_MODEL)
    KEY_FILE_HANDLER.remove_key(KeyValues.MOONSHOT_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.MOONSHOT_API_KEY)
    # Grok
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_GROK_MODEL)
    KEY_FILE_HANDLER.remove_key(KeyValues.GROK_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.GROK_API_KEY)
    # DeepSeek
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_DEEPSEEK_MODEL)
    KEY_FILE_HANDLER.remove_key(KeyValues.DEEPSEEK_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.DEEPSEEK_API_KEY)

def clear_embedding_model_keys():
    # Azure Open AI
    KEY_FILE_HANDLER.remove_key(KeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_AZURE_OPENAI_EMBEDDING)
    # Local Model
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_EMBEDDING_MODEL_NAME)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_EMBEDDING_BASE_URL)
    KEY_FILE_HANDLER.remove_key(KeyValues.LOCAL_EMBEDDING_API_KEY)
    KEY_FILE_HANDLER.remove_key(KeyValues.USE_LOCAL_EMBEDDINGS)
