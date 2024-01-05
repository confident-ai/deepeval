import os
import socket
import sys
import sentry_sdk


def blocked_by_firewall():
    try:
        socket.create_connection(("www.google.com", 80))
        return False
    except OSError:
        return True


def capture_evaluation_count():
    sentry_sdk.capture_message("evaluation ran!")


sentry_sdk.init(
    dsn="https://5ef587d58109ee45d6544f3657efdd1f@o4506098477236224.ingest.sentry.io/4506098479136768",
    profiles_sample_rate=1.0,
    traces_sample_rate=1.0,  # For performance monitoring
    send_default_pii=False,  # Don't send personally identifiable information
    attach_stacktrace=False,  # Don't attach stack traces to messages
    default_integrations=False,  # Disable Sentry's default integrations
)

if os.getenv("ERROR_REPORTING") == "YES" and not blocked_by_firewall():

    def handle_exception(exc_type, exc_value, exc_traceback):
        print({"exc_type": exc_type, "exc_value": exc_value})
        sentry_sdk.capture_exception(exc_value)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception
