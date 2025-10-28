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
from deepeval.config.settings import get_settings
from deepeval.constants import LOGIN_PROMPT, HIDDEN_DIR, KEY_FILE
from posthog import Posthog


class Feature(Enum):
    REDTEAMING = "redteaming"
    SYNTHESIZER = "synthesizer"
    EVALUATION = "evaluation"
    COMPONENT_EVALUATION = "component_evaluation"
    GUARDRAIL = "guardrail"
    BENCHMARK = "benchmark"
    CONVERSATION_SIMULATOR = "conversation_simulator"
    UNKNOWN = "unknown"
    TRACING_INTEGRATION = "tracing_integration"


TELEMETRY_DATA_FILE = ".deepeval_telemetry.txt"
TELEMETRY_PATH = os.path.join(HIDDEN_DIR, TELEMETRY_DATA_FILE)

#########################################################
### Telemetry HELPERS ###################################
#########################################################


def telemetry_opt_out():
    return get_settings().DEEPEVAL_TELEMETRY_OPT_OUT


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


#########################################################
### Move Folders ########################################
#########################################################
if not telemetry_opt_out():
    if os.path.exists(KEY_FILE) and not os.path.isdir(HIDDEN_DIR):
        temp_deepeval_file_name = ".deepeval_temp"
        os.rename(KEY_FILE, temp_deepeval_file_name)
        os.makedirs(HIDDEN_DIR, exist_ok=True)
        os.rename(temp_deepeval_file_name, os.path.join(HIDDEN_DIR, KEY_FILE))

    os.makedirs(HIDDEN_DIR, exist_ok=True)

    if os.path.exists(TELEMETRY_DATA_FILE):
        os.rename(TELEMETRY_DATA_FILE, TELEMETRY_PATH)

    if os.path.exists(".deepeval-cache.json"):
        os.rename(".deepeval-cache.json", f"{HIDDEN_DIR}/.deepeval-cache.json")

    if os.path.exists(".temp_test_run_data.json"):
        os.rename(
            ".temp_test_run_data.json", f"{HIDDEN_DIR}/.temp_test_run_data.json"
        )

#########################################################
### Telemetry Config ####################################
#########################################################

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

    # Initialize PostHog
    posthog = Posthog(
        project_api_key="phc_IXvGRcscJJoIb049PtjIZ65JnXQguOUZ5B5MncunFdB",
        host="https://us.i.posthog.com",
    )


if (
    get_settings().ERROR_REPORTING
    and not blocked_by_firewall()
    and not telemetry_opt_out()
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
    if telemetry_opt_out():
        yield
    else:
        # data
        event = f"Ran {type}"
        distinct_id = get_unique_id()
        feature = (
            Feature.COMPONENT_EVALUATION
            if event == "Ran traceable evaluate()"
            else Feature.EVALUATION
        )
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
        }
        if feature == Feature.EVALUATION:
            properties["feature_status.evaluation"] = get_feature_status(
                feature
            )
        elif feature == Feature.COMPONENT_EVALUATION:
            properties["feature_status.component_evaluation"] = (
                get_feature_status(feature)
            )
        set_last_feature(feature)
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_recommend_metrics():
    if telemetry_opt_out():
        yield
    else:
        # data
        event = "Recommend"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
        }
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_metric_type(
    metric_name: str, async_mode: bool, in_component: bool, _track: bool = True
):
    if telemetry_opt_out():
        yield
    else:
        # data
        event = metric_name
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "async_mode": async_mode,
            "in_component": int(in_component),
        }
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_synthesizer_run(
    method: str, max_generations: int, num_evolutions: int, evolutions: Dict
):
    if telemetry_opt_out():
        yield
    else:
        # data
        event = "Invoked synthesizer"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "feature_status.synthesizer": get_feature_status(
                Feature.SYNTHESIZER
            ),
            "method": method,
            "max_generations": max_generations,
            "num_evolutions": num_evolutions,
            **{f"evolution.{evol.value}": 1 for evol in evolutions},
        }
        set_last_feature(Feature.SYNTHESIZER)
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_conversation_simulator_run(num_conversations: int):
    if telemetry_opt_out():
        yield
    else:
        # data
        event = "Invoked conversation simulator"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "feature_status.conversation_simulator": get_feature_status(
                Feature.CONVERSATION_SIMULATOR
            ),
            "num_conversations": num_conversations,
        }
        set_last_feature(Feature.CONVERSATION_SIMULATOR)
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_guardrails(guards: List[str]):
    if telemetry_opt_out():
        yield
    else:
        event = "Ran guardrails"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "feature_status.guardrail": get_feature_status(Feature.GUARDRAIL),
            **{f"vulnerability.{guard}": 1 for guard in guards},
        }
        set_last_feature(Feature.GUARDRAIL)
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_benchmark_run(benchmark: str, num_tasks: int):
    if telemetry_opt_out():
        yield
    else:
        event = "Ran benchmark"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "feature_status.benchmark": get_feature_status(Feature.BENCHMARK),
            "benchmark": benchmark,
            "num_tasks": num_tasks,
        }
        set_last_feature(Feature.BENCHMARK)
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_login_event():
    if telemetry_opt_out():
        yield
    else:
        event = "Login"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "last_feature": get_last_feature().value,
            "completed": True,
            "login_prompt": LOGIN_PROMPT,
        }
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_view_event():
    if telemetry_opt_out():
        yield
    else:
        event = "View"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "last_feature": get_last_feature().value,
            "completed": True,
            "login_prompt": LOGIN_PROMPT,
        }
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


@contextmanager
def capture_pull_dataset():
    if telemetry_opt_out():
        yield
    else:
        event = "Pull"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
        }
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


# track metrics that are components and metrics that aren't components


# number of traces
@contextmanager
def capture_send_trace():
    if telemetry_opt_out():
        yield
    else:
        event = "Send Trace"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
        }
        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            yield span


# tracing integration
@contextmanager
def capture_tracing_integration(integration_name: str):
    if telemetry_opt_out():
        yield
    else:
        event = f"Tracing Integration: deepeval.integrations.{integration_name}"
        distinct_id = get_unique_id()
        properties = {
            "logged_in_with": get_logged_in_with(),
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "user.unique_id": get_unique_id(),
            "user.public_ip": (
                anonymous_public_ip if anonymous_public_ip else "Unknown"
            ),
            "feature_status.tracing_integration": get_feature_status(
                Feature.TRACING_INTEGRATION
            ),
        }
        set_last_feature(Feature.TRACING_INTEGRATION)

        # capture posthog
        posthog.capture(
            distinct_id=distinct_id, event=event, properties=properties
        )
        # capture new relic
        with tracer.start_as_current_span(event) as span:
            for property, value in properties.items():
                span.set_attribute(property, value)
            # OTEL/New Relic filtering attributes
            span.set_attribute("integration.name", integration_name)
            yield span


#########################################################
### Helper Functions s####################################
#########################################################


def read_telemetry_file() -> dict:
    """Reads the telemetry data file and returns the key-value pairs as a dictionary."""
    if not os.path.exists(TELEMETRY_PATH):
        return {}
    with open(TELEMETRY_PATH, "r") as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        key, _, value = line.strip().partition("=")
        data[key] = value
    return data


def write_telemetry_file(data: dict):
    """Writes the given key-value pairs to the telemetry data file."""
    # respect opt out
    if telemetry_opt_out():
        return

    # ensure directory exists before write
    os.makedirs(HIDDEN_DIR, exist_ok=True)
    with open(TELEMETRY_PATH, "w") as file:
        for key, value in data.items():
            file.write(f"{key}={value}\n")


def get_status() -> str:
    """Gets the status from the telemetry file."""
    data = read_telemetry_file()
    return data.get("DEEPEVAL_STATUS", "new")


def get_unique_id() -> str:
    """Gets or generates a unique ID and updates the telemetry file."""
    # respect opt out
    if telemetry_opt_out():
        return "telemetry-opted-out"
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
