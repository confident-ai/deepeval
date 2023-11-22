from typing import Callable, Any
import time


def call_openai_with_retry(
    callable: Callable[[], Any], max_retries: int = 2
) -> Any:
    for _ in range(max_retries):
        try:
            response = callable()
            return response
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(2)
            continue

    raise Exception(
        "Max retries reached. Unable to make a successful API call to OpenAI."
    )
