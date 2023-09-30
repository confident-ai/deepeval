import os
import platform
import urllib.parse
import requests
import json
import warnings
from collections import defaultdict

from typing import Any, Optional
from pydantic import BaseModel, Field
from typing import List
from requests.adapters import HTTPAdapter, Response, Retry

from deepeval.constants import (
    API_KEY_ENV,
    PYTEST_RUN_ENV_VAR,
    PYTEST_RUN_TEST_NAME,
)
from deepeval.key_handler import KEY_FILE_HANDLER
from deepeval.metrics.metric import Metric
from deepeval.test_case import LLMTestCase

API_BASE_URL = "https://app.confident-ai.com/api"
# API_BASE_URL = "http://localhost:3000/api"

# Parameters for HTTP retry
HTTP_TOTAL_RETRIES = 3  # Number of total retries
HTTP_RETRY_BACKOFF_FACTOR = 2  # Wait 1, 2, 4 seconds between retries
HTTP_STATUS_FORCE_LIST = [408, 429] + list(range(500, 531))
HTTP_RETRY_ALLOWED_METHODS = frozenset({"GET", "POST", "DELETE"})


class MetricsMetadata(BaseModel):
    metric: str
    score: float
    minimum_score: float = Field(None, alias="minimumScore")


class APITestCase(BaseModel):
    name: str
    input: str
    actual_output: str = Field(..., alias="actualOutput")
    expected_output: str = Field(..., alias="expectedOutput")
    success: bool
    metrics_metadata: List[MetricsMetadata] = Field(
        ..., alias="metricsMetadata"
    )
    threshold: float
    run_duration: float = Field(..., alias="runDuration")
    context: Optional[list] = Field(None)


class MetricScore(BaseModel):
    metric: str
    score: float

    @classmethod
    def from_metric(cls, metric: Metric):
        return cls(metric=metric.__name__, score=metric.score)


class TestRunResponse(BaseModel):
    """Add Test Run Results"""

    testRunId: str
    projectId: str


class MetricDict:
    def __init__(self):
        self.metric_dict = {}
        self.metric_count = {}

    def add_metric(self, metric_name, score):
        if metric_name not in self.metric_dict:
            self.metric_dict[metric_name] = score
            self.metric_count[metric_name] = 1
        else:
            self.metric_dict[metric_name] += score
            self.metric_count[metric_name] += 1

    def get_average_metric_score(self):
        return [
            MetricScore(
                metric=metric,
                score=self.metric_dict[metric] / self.metric_count[metric],
            )
            for metric in self.metric_dict
        ]


class MetricsMetadataAverageDict:
    def __init__(self):
        self.metric_dict = defaultdict(list)
        self.min_score_dict = defaultdict(float)

    def add_metric(self, metric: Metric):
        self.metric_dict[metric.__name__].append(metric.score)
        self.min_score_dict[metric.__name__] = min(
            self.min_score_dict.get(metric.__name__, float("inf")),
            metric.minimum_score,
        )

    def get_metrics_metadata(self):
        return [
            MetricsMetadata(
                metric=metric_name,
                score=sum(scores) / len(scores),
                minimumScore=self.min_score_dict[metric_name],
            )
            for metric_name, scores in self.metric_dict.items()
        ]


class TestRun(BaseModel):
    test_file: Optional[str] = Field(
        # TODO: Fix test_file
        "test.py",
        alias="testFile",
    )
    test_cases: List[APITestCase] = Field(
        alias="testCases", default_factory=lambda: []
    )
    metric_scores: List[MetricScore] = Field(
        default_factory=lambda: [], alias="metricScores"
    )
    configurations: dict

    def add_llm_test_case(
        self, test_case: LLMTestCase, metrics: List[Metric], run_duration: float
    ):
        # Check if test case with the same ID already exists
        existing_test_case: APITestCase = next(
            (tc for tc in self.test_cases if tc.name == test_case.__name__),
            None,
        )

        metrics_metadata_dict = MetricsMetadataAverageDict()
        for metric in metrics:
            metrics_metadata_dict.add_metric(metric)
        metrics_metadata = metrics_metadata_dict.get_metrics_metadata()
        success = all([metric.is_successful() for metric in metrics])
        threshold = metrics[0].minimum_score

        if existing_test_case:
            # If it exists, append the metrics to the existing test case
            existing_test_case.metrics_metadata.extend(metrics_metadata)
            # Update the success status and threshold
            existing_test_case.success = success
            existing_test_case.threshold = threshold
        else:
            # If it doesn't exist, create a new test case
            # Adding backwards compatibility to ensure context still works.
            context = test_case.context
            if isinstance(context, str):
                context = [context]
            self.test_cases.append(
                APITestCase(
                    # Get the test from the pytest plugin
                    name=os.getenv(PYTEST_RUN_TEST_NAME, "-"),
                    input=test_case.query,
                    actualOutput=test_case.output,
                    expectedOutput=test_case.expected_output,
                    success=success,
                    metricsMetadata=metrics_metadata,
                    threshold=threshold,
                    runDuration=run_duration,
                    context=context,
                )
            )

        all_metric_dict = MetricDict()

        for test_case in self.test_cases:
            test_case: APITestCase
            metrics = test_case.metrics_metadata
            for metric in metrics:
                metric: MetricsMetadata
                all_metric_dict.add_metric(metric.metric, metric.score)

        self.metric_scores = all_metric_dict.get_average_metric_score()

    def save(self, file_path: Optional[str] = None):
        if file_path is None:
            file_path = os.getenv(PYTEST_RUN_ENV_VAR)
            # If file Path is None, remove it
            if not file_path:
                return
            elif not file_path.endswith(".json"):
                file_path = f"{file_path}.json"
        with open(file_path, "w") as f:
            json.dump(self.dict(by_alias=True, exclude_none=True), f)

        return file_path

    @classmethod
    def load(cls, file_path: Optional[str] = None):
        if file_path is None:
            file_path = os.getenv(PYTEST_RUN_ENV_VAR)
            # If file Path is None, remove it
            if not file_path:
                return
            elif not file_path.endswith(".json"):
                file_path = f"{file_path}.json"
        with open(file_path, "r") as f:
            return cls(**json.load(f))


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

    def add_test_case(
        self,
        query: str,
        actual_output: str,
        metric_score: float,
        metric_name: str,
        success: bool,
        datapoint_id: str,
        implementation_id: str,
        metrics_metadata: Any,
    ):
        """send test case data to the prod-data endpoint"""
        if not metrics_metadata:
            metrics_metadata = {}
        return self.post_request(
            endpoint="/v1/prod-data",
            body={
                "query": query,
                "actualOutput": actual_output,
                "defaultMetricScore": metric_score,
                "defaultMetricName": metric_name,
                "implementationId": implementation_id,
                "success": success,
                "metricsMetadata": metrics_metadata,
                "goldenId": datapoint_id,
            },
        )

    def add_golden(
        self,
        query: str = "-",
        expected_output: str = "-",
        context: str = "-",
        is_synthetic: bool = False,
    ):
        return self.post_request(
            endpoint="v1/golden",
            body={
                "query": query,
                "expectedOutput": expected_output,
                "context": context,
                "isSynthetic": is_synthetic,
            },
        )

    def list_prod_data(self, implementation_id: str):
        return self.get_request(
            endpoint="/v1/prod-data?implementationId=" + implementation_id
        )

    def create_implementation(
        self, name: str, description: Optional[str] = None
    ):
        body = {"name": name}
        if description:
            body["description"] = description
        return self.post_request(endpoint="/v1/implementation", body=body)

    def list_implementations(self):
        """
        Returns a list of implementations
        """
        return self.get_request(endpoint="/v1/implementation")

    def post_test_run(self, test_run: TestRun) -> TestRunResponse:
        """Post a test run"""
        try:
            # make sure to exclude none for `context` to ensure it is handled properly
            body = test_run.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = test_run.dict(by_alias=True, exclude_none=True)

        result = self.post_request(
            endpoint="/v1/test-run",
            body=body,
        )
        response = TestRunResponse(
            testRunId=result["testRunId"], projectId=result["projectId"]
        )
        return response
