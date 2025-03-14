from contextlib import contextmanager
import logging
import os
import socket
import sys
import uuid
import sentry_sdk
from enum import Enum
from typing import List, Dict
import requests
from deepeval.constants import LOGIN_PROMPT


class Feature(Enum):
    REDTEAMING = "redteaming"
    SYNTHESIZER = "synthesizer"
    EVALUATION = "evaluation"
    GUARDRAIL = "guardrail"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"


TELEMETRY_DATA_FILE = ".deepeval_telemetry.txt"

#########################################################
### Telemetry Config ####################################
#########################################################


def telemetry_opt_out():
    return os.getenv("DEEPEVAL_TELEMETRY_OPT_OUT") == "YES"


def blocked_by_firewall():
    try:
        socket.create_connection(("www.google.com", 80))
        return False
    except OSError:
        return True


def get_anonymous_public_ip():
    try:
        response = requests.get("https://api.ipify.org", timeout=5)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        pass
    return None


anonymous_public_ip = None

if not telemetry_opt_out():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    anonymous_public_ip = get_anonymous_public_ip()
    sentry_sdk.init(
        dsn="https://5ef587d58109ee45d6544f3657efdd1f@o4506098477236224.ingest.sentry.io/4506098479136768",
        profiles_sample_rate=1.0,
        traces_sample_rate=1.0,  # For performance monitoring
        send_default_pii=False,  # Don't send personally identifiable information
        attach_stacktrace=False,  # Don't attach stack traces to messages
        default_integrations=False,  # Disable Sentry's default integrations
    )

    # Set up the Tracer Provider
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()

    # New Relic License Key and OTLP Endpoint
    NEW_RELIC_LICENSE_KEY = "1711c684db8a30361a7edb0d0398772cFFFFNRAL"
    NEW_RELIC_OTLP_ENDPOINT = "https://otlp.nr-data.net:4317"
    otlp_exporter = OTLPSpanExporter(
        endpoint=NEW_RELIC_OTLP_ENDPOINT,
        headers={"api-key": NEW_RELIC_LICENSE_KEY},
    )

    # Add the OTLP exporter to the span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    logging.getLogger("opentelemetry.exporter.otlp").setLevel(logging.CRITICAL)

    # Create a tracer for your application
    tracer = trace.get_tracer(__name__)


if (
    os.getenv("ERROR_REPORTING") == "YES"
    and not blocked_by_firewall()
    and not os.getenv("TELEMETRY_OPT_OUT")
):

    def handle_exception(exc_type, exc_value, exc_traceback):
        print({"exc_type": exc_type, "exc_value": exc_value})
        sentry_sdk.capture_exception(exc_value)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


def is_running_in_jupyter_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        pass
    return False


IS_RUNNING_IN_JUPYTER = (
    "jupyter" if is_running_in_jupyter_notebook() else "other"
)

#########################################################
### Context Managers ####################################
#########################################################


@contextmanager
def capture_evaluation_run(type: str):
    if not telemetry_opt_out():
        with tracer.start_as_current_span(f"Ran {type}") as span:
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            span.set_attribute(
                "feature_status.evaluation",
                get_feature_status(Feature.EVALUATION),
            )
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            set_last_feature(Feature.EVALUATION)
            yield span
    else:
        yield


@contextmanager
def capture_recommend_metrics():
    if not telemetry_opt_out():
        with tracer.start_as_current_span("Recommend") as span:
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            yield span
    else:
        yield


@contextmanager
def capture_metric_type(
    metric_name: str, async_mode: bool, _track: bool = True
):
    if not telemetry_opt_out() and _track:
        with tracer.start_as_current_span(metric_name) as span:
            span.set_attribute("async_mode", async_mode)
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            yield span
    else:
        yield


@contextmanager
def capture_synthesizer_run(
    method: str, max_generations: int, num_evolutions: int, evolutions: Dict
):
    if not telemetry_opt_out() and max_generations is not None:
        with tracer.start_as_current_span(f"Invoked synthesizer") as span:
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            span.set_attribute(
                "feature_status.synthesizer",
                get_feature_status(Feature.SYNTHESIZER),
            )
            span.set_attribute("method", method)
            span.set_attribute("max_generations", max_generations)
            span.set_attribute("evolutions", num_evolutions)
            for evol, value in evolutions.items():
                span.set_attribute(f"evolution.{evol.value}", 1)
            set_last_feature(Feature.SYNTHESIZER)
            yield span
    else:
        yield


@contextmanager
def capture_red_teamer_run(
    attacks_per_vulnerability_type: int,
    vulnerabilities: List[str],
    attack_enhancements: Dict,
):
    if not telemetry_opt_out():
        with tracer.start_as_current_span(f"Invokved redteamer") as span:
            # if anonymous_public_ip:
            #     span.set_attribute("user.public_ip", anonymous_public_ip)
            # span.set_attribute("logged_in_with", get_logged_in_with())
            # span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            # span.set_attribute("user.status", get_status())
            # span.set_attribute("user.unique_id", get_unique_id())
            # span.set_attribute(
            #     "feature_status.redteaming",
            #     get_feature_status(Feature.REDTEAMING),
            # )
            # span.set_attribute(
            #     "attacks_per_vulnerability", attacks_per_vulnerability_type
            # )
            # for vuln in vulnerabilities:
            #     for types in vuln.get_types():
            #         span.set_attribute(f"vulnerability.{types.value}", 1)
            # for enhancement, value in attack_enhancements.items():
            #     span.set_attribute(
            #         f"attack_enhancement.{enhancement.value}", value
            #     )
            # set_last_feature(Feature.REDTEAMING)
            yield span
    else:
        yield


@contextmanager
def capture_guardrails(guards: List[str]):
    if not telemetry_opt_out():
        with tracer.start_as_current_span(f"Ran guardrails") as span:
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            span.set_attribute(
                "feature_status.guardrail",
                get_feature_status(Feature.GUARDRAIL),
            )
            for guard in guards:
                span.set_attribute(f"vulnerability.{guard}", 1)
            set_last_feature(Feature.GUARDRAIL)
            yield span
    else:
        yield


@contextmanager
def capture_benchmark_run(benchmark: str, num_tasks: int):
    if not telemetry_opt_out():
        with tracer.start_as_current_span(f"Ran benchmark") as span:
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            span.set_attribute(
                "feature_status.benchmark",
                get_feature_status(Feature.BENCHMARK),
            )
            span.set_attribute("benchmark", benchmark)
            span.set_attribute("num_tasks", num_tasks)
            set_last_feature(Feature.BENCHMARK)
            yield span
    else:
        yield


@contextmanager
def capture_login_event():
    if not telemetry_opt_out():
        with tracer.start_as_current_span(f"Login") as span:
            last_feature = get_last_feature()
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            span.set_attribute("last_feature", last_feature.value)
            span.set_attribute("completed", True)
            span.set_attribute("login_prompt", LOGIN_PROMPT)
            yield span
    else:
        yield


@contextmanager
def capture_pull_dataset():
    if not telemetry_opt_out():
        with tracer.start_as_current_span(f"Pull") as span:
            span.set_attribute("logged_in_with", get_logged_in_with())
            span.set_attribute("environment", IS_RUNNING_IN_JUPYTER)
            span.set_attribute("user.status", get_status())
            span.set_attribute("user.unique_id", get_unique_id())
            if anonymous_public_ip:
                span.set_attribute("user.public_ip", anonymous_public_ip)
            yield span
    else:
        yield


#########################################################
### Helper Functions ####################################
#########################################################


def read_telemetry_file() -> dict:
    """Reads the telemetry data file and returns the key-value pairs as a dictionary."""
    if not os.path.exists(TELEMETRY_DATA_FILE):
        return {}
    with open(TELEMETRY_DATA_FILE, "r") as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        key, _, value = line.strip().partition("=")
        data[key] = value
    return data


def write_telemetry_file(data: dict):
    """Writes the given key-value pairs to the telemetry data file."""
    with open(TELEMETRY_DATA_FILE, "w") as file:
        for key, value in data.items():
            file.write(f"{key}={value}\n")


def get_status() -> str:
    """Gets the status from the telemetry file."""
    data = read_telemetry_file()
    return data.get("DEEPEVAL_STATUS", "new")


def get_unique_id() -> str:
    """Gets or generates a unique ID and updates the telemetry file."""
    data = read_telemetry_file()
    unique_id = data.get("DEEPEVAL_ID")
    if not unique_id:
        unique_id = str(uuid.uuid4())
        data["DEEPEVAL_ID"] = unique_id
        data["DEEPEVAL_STATUS"] = "new"
    else:
        data["DEEPEVAL_STATUS"] = "old"
    write_telemetry_file(data)
    return unique_id


def get_last_feature() -> Feature:
    """Gets the last feature from the telemetry file."""
    data = read_telemetry_file()
    last_feature = data.get("DEEPEVAL_LAST_FEATURE")
    if last_feature and last_feature in Feature._value2member_map_:
        return Feature(last_feature)
    return Feature.UNKNOWN


def set_last_feature(feature: Feature):
    """Sets the last feature in the telemetry file."""
    if feature not in Feature:
        raise ValueError(f"Invalid feature: {feature}")
    data = read_telemetry_file()
    data["DEEPEVAL_LAST_FEATURE"] = feature.value
    feature_status_key = f"DEEPEVAL_{feature.value.upper()}_STATUS"
    data[feature_status_key] = "old"
    write_telemetry_file(data)


def get_feature_status(feature: Feature) -> str:
    """Gets the status of a feature ('new' or 'old') from the telemetry file."""
    data = read_telemetry_file()
    feature_status_key = f"DEEPEVAL_{feature.value.upper()}_STATUS"
    return data.get(feature_status_key, "new")


def set_logged_in_with(logged_in_with: str):
    data = read_telemetry_file()
    data["LOGGED_IN_WITH"] = logged_in_with
    write_telemetry_file(data)


def get_logged_in_with():
    data = read_telemetry_file()
    return data.get("LOGGED_IN_WITH", "NA")
