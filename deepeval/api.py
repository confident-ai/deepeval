import os
import platform
import urllib.parse
import requests
import warnings
from requests.adapters import HTTPAdapter, Response, Retry
from deepeval.constants import API_KEY_ENV
from deepeval.key_handler import KEY_FILE_HANDLER
from enum import Enum

API_BASE_URL = "https://app.confident-ai.com/api"

# Parameters for HTTP retry
HTTP_TOTAL_RETRIES = 3  # Number of total retries
HTTP_RETRY_BACKOFF_FACTOR = 2  # Wait 1, 2, 4 seconds between retries
HTTP_STATUS_FORCE_LIST = [408, 429] + list(range(500, 531))
HTTP_RETRY_ALLOWED_METHODS = frozenset({"GET", "POST", "DELETE"})


class Endpoints(Enum):
    CREATE_DATASET_ENDPOINT = "/v1/dataset"
    CREATE_TEST_RUN_ENDPOINT = "/v1/test-run"


class Api:
    """Internal Api reference for handling http operations"""

    def __init__(
        self,
        api_key: str = os.getenv(API_KEY_ENV, ""),
        user_agent_extension=None,
        api_instance_url=None,
        verify=None,
        proxies=None,
        cert=None,
    ):
        if api_key == "":
            # get API key if none is supplied after you log in
            api_key = KEY_FILE_HANDLER.fetch_api_key()

        if api_key == "" or api_key is None:
            raise ValueError("Please provide a valid API Key.")

        self.api_key = api_key

        self._auth = (self.api_key, "")
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": self._generate_useragent(user_agent_extension),
            # "Authorization": "Bearer " + api_key,
            # This is what gets sent now - auth gets sent to firebase instead
            "CONFIDENT_API_KEY": api_key,
        }
        self._headers_multipart_form_data = {
            "User-Agent": self._generate_useragent(user_agent_extension),
        }
        self.base_api_url = api_instance_url or API_BASE_URL

        self.verify = verify
        self.proxies = proxies
        self.cert = cert

    @staticmethod
    def _http_request(
        method,
        url,
        headers=None,
        auth=None,
        params=None,
        body=None,
        files=None,
        data=None,
        verify=None,
        proxies=None,
        cert=None,
    ) -> Response:
        https = requests.Session()
        retry_strategy = Retry(
            total=HTTP_TOTAL_RETRIES,
            backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
            status_forcelist=HTTP_STATUS_FORCE_LIST,
            allowed_methods=HTTP_RETRY_ALLOWED_METHODS,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        https.mount("https://", adapter)

        https.cert = cert if cert else None
        https.verify = verify if verify else True
        if proxies:
            https.proxies.update(proxies)

        try:
            params = params or {}
            body = body or None

            res = https.request(
                method=method,
                url=url,
                headers=headers,
                # auth=auth,
                params=params,
                json=body,
                files=files,
                data=data,
            )

            return res
        except Exception as err:
            raise Exception(err) from err

    @staticmethod
    def _raise_on_response(res: Response):
        try:
            message = res.json().get("error", res.text)
        except ValueError:
            message = res.text
        if res.status_code == 410:
            warnings.warn(f"Deprecation Warning: {message}", DeprecationWarning)
        raise Exception(message)

    def _api_request(
        self,
        method,
        endpoint,
        headers=None,
        auth=None,
        params=None,
        body=None,
        files=None,
        data=None,
    ):
        """Generic HTTP request method with error handling."""

        url = f"{self.base_api_url}/{endpoint}"
        res = self._http_request(
            method,
            url,
            headers,
            auth,
            params,
            body,
            files,
            data,
            verify=self.verify,
            proxies=self.proxies,
            cert=self.cert,
        )
        json = None
        if res.status_code == 200:
            try:
                json = res.json()
            except ValueError:
                # Some endpoints only return 'OK' message without JSON
                return json
        elif (
            res.status_code == 409
            and "task" in endpoint
            and body.get("unique_id")
        ):
            retry_history = res.raw.retries.history
            # Example RequestHistory tuple
            # RequestHistory(method='POST',
            #   url='/v1/task/imageannotation',
            #   error=None,
            #   status=409,
            #   redirect_location=None)
            if retry_history != ():
                # See if the first retry was a 500 or 503 error
                if retry_history[0][3] >= 500:
                    uuid = body["unique_id"]
                    newUrl = f"{self.base_api_url}/tasks?unique_id={uuid}"
                    # grab task from api
                    newRes = self._http_request(
                        "GET", newUrl, headers=headers, auth=auth
                    )
                    json = newRes.json()["docs"][0]
            else:
                self._raise_on_response(res)
        else:
            self._raise_on_response(res)
        return json

    def get_request(self, endpoint, params=None):
        """Generic GET Request Wrapper"""
        return self._api_request(
            "GET",
            endpoint,
            headers=self._headers,
            auth=self._auth,
            params=params,
        )

    def post_request(self, endpoint, body=None, files=None, data=None):
        """Generic POST Request Wrapper"""
        return self._api_request(
            "POST",
            endpoint,
            headers=self._headers
            if files is None
            else self._headers_multipart_form_data,
            auth=self._auth,
            body=body,
            files=files,
            data=data,
        )

    def delete_request(self, endpoint, params=None, body=None):
        """Generic DELETE Request Wrapper"""
        return self._api_request(
            "DELETE",
            endpoint,
            headers=self._headers,
            auth=self._auth,
            params=params,
            body=body,
        )

    def put_request(self, endpoint, body=None, params=None):
        """Generic PUT Request Wrapper"""
        return self._api_request(
            "PUT",
            endpoint,
            body=body,
            headers=self._headers,
            auth=self._auth,
            params=params,
        )

    @staticmethod
    def _generate_useragent(extension: str = None) -> str:
        """Generates UserAgent parameter with module, Python
        and OS details

        Args:
            extension (str, optional): Option to extend UserAgent
            with source system

        Returns:
            str: Generated UserAgent parameter with platform versions
        """
        python_version = platform.python_version()
        os_platform = platform.platform()

        user_agent = " ".join(
            filter(
                None,
                [
                    f"Python/{python_version}",
                    f"OS/{os_platform}",
                    extension,
                ],
            )
        )
        return user_agent

    @staticmethod
    def quote_string(text: str) -> str:
        """Project and Batch names can be a part of URL, which causes
        an error in case of a special character used.
        Quotation assures the right object to be retrieved from API.

        `quote_string('a bc/def')` -> `a%20bc%2Fdef`

        Args:
            text (str): Input text to be quoted

        Returns:
            str: Quoted text in return
        """
        return urllib.parse.quote(text, safe="")
