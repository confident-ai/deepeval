import logging
from typing import Optional, Any, Union, Tuple
import aiohttp
import requests
from enum import Enum
from tenacity import (
    retry,
    wait_exponential_jitter,
    retry_if_exception_type,
)


import deepteam
from deepeval.confident.types import ApiResponse, ConfidentApiError
from deepeval.confident.api import (
    get_confident_api_key,
    CONFIDENT_API_KEY_ENV_VAR,
    get_base_api_url,
    retryable_exceptions,
    log_retry_error,
    HttpMethods,
)


class Endpoints(Enum):
    RISK_ASSESSMENT_ENDPOINT = "/v1/risk-assessments"
    RT_FRAMEWORK_ENDPOINT = "/v1/rt-frameworks/:frameworkId"
    VULNERABILITIES_ENDPOINT = "/v1/vulnerabilities"
    VULNERABILITY_ENDPOINT = "/v1/vulnerabilities/:vulnerabilityId"


class Api:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = get_confident_api_key()

        if not api_key:
            raise ValueError(
                f"No Confident API key found. Please run `deepteam login` or set the {CONFIDENT_API_KEY_ENV_VAR} environment variable in the CLI."
            )

        self.api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "CONFIDENT_API_KEY": api_key,
            "X-DeepTeam-Version": deepteam.__version__,
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
            deprecation_msg = "You are using a deprecated API endpoint. Please update your deepteam version."
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
