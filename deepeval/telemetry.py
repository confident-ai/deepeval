from contextlib import contextmanager
import logging
import os
import socket
import sys
import uuid
import sentry_sdk
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)


def get_unique_id():
    unique_id = os.getenv("DEEPEVAL_UNIQUE_ID")
    if unique_id is None:
        unique_id = str(uuid.uuid4())
        os.environ["DEEPEVAL_UNIQUE_ID"] = unique_id
    return unique_id


def telemetry_opt_out():
    return os.getenv("DEEPEVAL_TELEMETRY_OPT_OUT") == "YES"


def blocked_by_firewall():
    try:
        socket.create_connection(("www.google.com", 80))
        return False
    except OSError:
        return True


if not telemetry_opt_out():
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


@contextmanager
def capture_evaluation_run(type: str):
    if not telemetry_opt_out():
        with tracer.start_as_current_span(f"Ran {type}") as span:
            span.set_attribute("user.unique_id", get_unique_id())
            yield span
    else:
        yield


@contextmanager
def capture_metric_type(metric_name: str, _track: bool = True):
    if not telemetry_opt_out() and _track:
        with tracer.start_as_current_span(metric_name) as span:
            span.set_attribute("user.unique_id", get_unique_id())
            yield span
    else:
        yield


@contextmanager
def capture_synthesizer_run(max_generations: int = None, method: str = None):
    if not telemetry_opt_out() and max_generations is not None:
        with tracer.start_as_current_span(
            f"Invoked synthesizer ({max_generations}) | Method: {method}"
        ) as span:
            span.set_attribute("user.unique_id", get_unique_id())
            yield span
    else:
        yield


@contextmanager
def capture_red_teamer_run(task: str):
    if not telemetry_opt_out():
        with tracer.start_as_current_span(
            f"Invokved redteamer: ({task})"
        ) as span:
            span.set_attribute("user.unique_id", get_unique_id())
            yield span
    else:
        yield
