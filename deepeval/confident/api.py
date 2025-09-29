import logging
from typing import Optional, Any, Union, Tuple
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
from pydantic import SecretStr

import deepeval
from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
from deepeval.confident.types import ApiResponse, ConfidentApiError
from deepeval.config.settings import get_settings


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


def get_confident_api_key() -> Optional[str]:
    s = get_settings()
    key: Optional[SecretStr] = s.CONFIDENT_API_KEY or s.API_KEY
    return key.get_secret_value() if key else None


def set_confident_api_key(api_key: Optional[str]) -> None:
    """
    - Always updates runtime (os.environ) via settings.edit()
    - If DEEPEVAL_DEFAULT_SAVE is set, also persists to dotenv
    - Never writes secrets to the legacy JSON keystore (your Settings logic already skips secrets)
    """
    s = get_settings()
    save = (
        s.DEEPEVAL_DEFAULT_SAVE or None
    )  # e.g. "dotenv" or "dotenv:/path/.env"

    # If you *only* want runtime changes unless a default save is present:
    if save is None:
        with s.edit(persist=False):
            s.CONFIDENT_API_KEY = SecretStr(api_key) if api_key else None
            s.API_KEY = SecretStr(api_key) if api_key else None
    else:
        # Respect default save: update runtime + write to dotenv, but not JSON
        with s.edit(save=save, persist=None):
            s.CONFIDENT_API_KEY = SecretStr(api_key) if api_key else None
            s.API_KEY = SecretStr(api_key) if api_key else None


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
    DATASET_ALIAS_ENDPOINT = "/v1/datasets/:alias"
    DATASET_ALIAS_QUEUE_ENDPOINT = "/v1/datasets/:alias/queue"

    TEST_RUN_ENDPOINT = "/v1/test-run"
    TRACES_ENDPOINT = "/v1/traces"
    ANNOTATIONS_ENDPOINT = "/v1/annotations"
    PROMPTS_VERSION_ID_ENDPOINT = "/v1/prompts/:alias/versions/:versionId"
    PROMPTS_ENDPOINT = "/v1/prompts"
    PROMPTS_VERSIONS_ENDPOINT = "/v1/prompts/:alias/versions"
    SIMULATE_ENDPOINT = "/v1/simulate"
    EVALUATE_ENDPOINT = "/v1/evaluate"

    EVALUATE_THREAD_ENDPOINT = "/v1/evaluate/threads/:threadId"
    EVALUATE_TRACE_ENDPOINT = "/v1/evaluate/traces/:traceUuid"
    EVALUATE_SPAN_ENDPOINT = "/v1/evaluate/spans/:spanUuid"


class Api:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = get_confident_api_key()

        if not api_key:
            raise ValueError(
                f"No Confident API key found. Please run `deepeval login` or set the {CONFIDENT_API_KEY_ENV_VAR} environment variable in the CLI."
            )

        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "CONFIDENT_API_KEY": api_key,
            "X-DeepEval-Version": deepeval.__version__,
        }
        self.base_api_url = get_base_api_url()

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

    def _handle_response(
        self, response_data: Union[dict, Any]
    ) -> Tuple[Any, Optional[str]]:
        if not isinstance(response_data, dict):
            return response_data, None

        try:
            api_response = ApiResponse(**response_data)
        except Exception:
            return response_data, None

        if api_response.deprecated:
            deprecation_msg = "You are using a deprecated API endpoint. Please update your deepeval version."
            if api_response.link:
                deprecation_msg += f" See: {api_response.link}"
            logging.warning(deprecation_msg)

        if not api_response.success:
            error_message = api_response.error or "Request failed"
            raise ConfidentApiError(error_message, api_response.link)

        return api_response.data, api_response.link

    def send_request(
        self,
        method: HttpMethods,
        endpoint: Endpoints,
        body=None,
        params=None,
        url_params=None,
    ) -> Tuple[Any, Optional[str]]:
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
                response_data = res.json()
                return self._handle_response(response_data)
            except ValueError:
                return res.text, None
        else:
            try:
                error_data = res.json()
                return self._handle_response(error_data)
            except (ValueError, ConfidentApiError) as e:
                if isinstance(e, ConfidentApiError):
                    raise e
                error_message = (
                    error_data.get("error", res.text)
                    if "error_data" in locals()
                    else res.text
                )
                raise Exception(error_message)

    async def a_send_request(
        self,
        method: HttpMethods,
        endpoint: Endpoints,
        body=None,
        params=None,
        url_params=None,
    ) -> Tuple[Any, Optional[str]]:
        url = f"{self.base_api_url}{endpoint.value}"

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
                        response_data = await res.json()
                        return self._handle_response(response_data)
                    except aiohttp.ContentTypeError:
                        return await res.text(), None
                else:
                    try:
                        error_data = await res.json()
                        return self._handle_response(error_data)
                    except (aiohttp.ContentTypeError, ConfidentApiError) as e:
                        if isinstance(e, ConfidentApiError):
                            raise e
                        error_message = (
                            error_data.get("error", await res.text())
                            if "error_data" in locals()
                            else await res.text()
                        )
                        raise Exception(error_message)
