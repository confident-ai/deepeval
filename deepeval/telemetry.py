import os
import socket
import sys


def check_firewall():
    try:
        socket.create_connection(("www.google.com", 80))
        return False
    except OSError:
        return True


if os.getenv("ERROR_REPORTING") == "YES" and not check_firewall():
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn="https://5ef587d58109ee45d6544f3657efdd1f@o4506098477236224.ingest.sentry.io/4506098479136768",
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            traces_sample_rate=1.0,
            # Set profiles_sample_rate to 1.0 to profile 100%
            # of sampled transactions.
            # We recommend adjusting this value in production.
            profiles_sample_rate=1.0,
        )

        # Add a global error handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            print({"exc_type": exc_type, "exc_value": exc_value})
            sentry_sdk.capture_exception(exc_value)
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = handle_exception

    except ModuleNotFoundError:
        # sentry_sdk not installed
        pass
