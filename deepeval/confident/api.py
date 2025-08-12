import logging
from typing import Optional
import aiohttp
import requests
from enum import Enum
import os

from tenacity import (
    retry,
    wait_exponential_jitter,
    retry_if_exception_type,
    RetryCallState,
)

from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues

CONFIDENT_API_KEY_ENV_VAR = "CONFIDENT_API_KEY"
DEEPEVAL_BASE_URL = "https://deepeval.confident-ai.com"
DEEPEVAL_BASE_URL_EU = "https://eu.deepeval.confident-ai.com"
API_BASE_URL = "https://api.confident-ai.com"
API_BASE_URL_EU = "https://eu.api.confident-ai.com"
retryable_exceptions = requests.exceptions.SSLError


def get_base_api_url():
    region = KEY_FILE_HANDLER.fetch_data(KeyValues.CONFIDENT_REGION)
    if region == "EU":
        return API_BASE_URL_EU
    else:
        return API_BASE_URL


def get_deepeval_base_url():
    region = KEY_FILE_HANDLER.fetch_data(KeyValues.CONFIDENT_REGION)
    if region == "EU":
        return DEEPEVAL_BASE_URL_EU
    else:
        return DEEPEVAL_BASE_URL


def get_confident_api_key():
    return KEY_FILE_HANDLER.fetch_data(KeyValues.API_KEY) or os.getenv(
        CONFIDENT_API_KEY_ENV_VAR
    )


def is_confident():
    confident_api_key = get_confident_api_key()
    return confident_api_key is not None


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"Confident AI Error: {exception}. Retrying: {retry_state.attempt_number} time(s)..."
    )


class HttpMethods(Enum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PUT = "PUT"


class Endpoints(Enum):
    DATASET_ENDPOINT = "/v1/datasets"
    DATASET_QUEUE_ENDPOINT = "/v1/datasets/:alias/queue"

    TEST_RUN_ENDPOINT = "/v1/test-run"
    TRACES_ENDPOINT = "/v1/traces"
    FEEDBACK_ENDPOINT = "/v1/feedback"
    PROMPT_ENDPOINT = "/v1/prompt"
    RECOMMEND_ENDPOINT = "/v1/recommend-metrics"
    EVALUATE_ENDPOINT = "/evaluate"

    EVALUATE_THREAD_ENDPOINT = "/v1/evaluate/threads/:threadId"
    EVALUATE_TRACE_ENDPOINT = "/v1/evaluate/traces/:traceUuid"
    EVALUATE_SPAN_ENDPOINT = "/v1/evaluate/spans/:spanUuid"


class Api:
    def __init__(self, api_key: Optional[str] = None, base_url=None):
        if api_key is None:
            api_key = (
                KEY_FILE_HANDLER.fetch_data(KeyValues.API_KEY)
                or get_confident_api_key()
            )

        if not api_key:
            raise ValueError(
                f"No Confident API key found. Please run `deepeval login` or set the {CONFIDENT_API_KEY_ENV_VAR} environment variable in the CLI."
            )

        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "CONFIDENT_API_KEY": api_key,
        }
        self.base_api_url = base_url or get_base_api_url()

    @staticmethod
    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def _http_request(
        method: str, url: str, headers=None, json=None, params=None
    ):
        session = requests.Session()
        return session.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            params=params,
            verify=True,  # SSL verification is always enabled
        )

    def send_request(
        self,
        method: HttpMethods,
        endpoint: Endpoints,
        body=None,
        params=None,
        url_params=None,
    ):
        url = f"{self.base_api_url}{endpoint.value}"

        # Replace URL parameters if provided
        if url_params:
            for key, value in url_params.items():
                placeholder = f":{key}"
                if placeholder in url:
                    url = url.replace(placeholder, str(value))

        res = self._http_request(
            method=method.value,
            url=url,
            headers=self._headers,
            json=body,
            params=params,
        )

        if res.status_code == 200:
            try:
                return res.json()
            except ValueError:
                return res.text
        elif res.status_code == 409 and body:
            message = res.json().get("message", "Conflict occurred.")

            # Prompt the user for action
            user_input = (
                input(
                    f"{message} Would you like to overwrite it? [y/N] or change the alias [c]: "
                )
                .strip()
                .lower()
            )

            if user_input == "y":
                body["overwrite"] = True
                return self.send_request(method, endpoint, body)
            elif user_input == "c":
                new_alias = input("Enter a new alias: ").strip()
                body["alias"] = new_alias
                return self.send_request(method, endpoint, body)
            else:
                print("Aborted.")
                return None
        else:
            raise Exception(res.json().get("error", res.text))

    async def a_send_request(
        self,
        method: HttpMethods,
        endpoint: Endpoints,
        body=None,
        params=None,
        url_params=None,
    ):
        url = f"{self.base_api_url}{endpoint.value}"

        # Replace URL parameters if provided
        if url_params:
            for key, value in url_params.items():
                placeholder = f":{key}"
                if placeholder in url:
                    url = url.replace(placeholder, str(value))

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method.value,
                url=url,
                headers=self._headers,
                json=body,
                params=params,
                ssl=True,  # SSL verification enabled
            ) as res:
                if res.status == 200:
                    try:
                        return await res.json()
                    except aiohttp.ContentTypeError:
                        return await res.text()
                elif res.status == 409 and body:
                    message = (await res.json()).get(
                        "message", "Conflict occurred."
                    )

                    user_input = (
                        input(
                            f"{message} Would you like to overwrite it? [y/N] or change the alias [c]: "
                        )
                        .strip()
                        .lower()
                    )

                    if user_input == "y":
                        body["overwrite"] = True
                        return await self.a_send_request(method, endpoint, body)
                    elif user_input == "c":
                        new_alias = input("Enter a new alias: ").strip()
                        body["alias"] = new_alias
                        return await self.a_send_request(method, endpoint, body)
                    else:
                        print("Aborted.")
                        return None
                else:
                    try:
                        error_data = await res.json()
                        error_message = error_data.get(
                            "error", await res.text()
                        )
                    except aiohttp.ContentTypeError:
                        error_message = await res.text()
                    raise Exception(error_message)
