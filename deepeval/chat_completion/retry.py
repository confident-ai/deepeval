from typing import Callable, Any
import openai
import time


def call_openai_with_retry(
    callable: Callable[[], Any], max_retries: int = 2
) -> Any:
    if not openai.api_key:
        raise ValueError(
            "OpenAI API key is not set. Please ensure it's set in your environment variables or passed explicitly."
        )

    for _ in range(max_retries):
        try:
            response = callable()
            return response
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(2)
            continue

    raise Exception(
        "Max retries reached. Unable to make a successful API call to OpenAI."
    )
