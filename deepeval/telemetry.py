import os
import socket
import sys


def check_firewall():
    try:
        socket.create_connection(("www.google.com", 80))
        return False
    except OSError:
        return True


if os.getenv("TURN_OFF_DEEPEVAL_TELEMETRY") != "Y" and not check_firewall():
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn="https://a378e1dcd3cd09b8aacf236c9a03dfc0@o4506033502093312.ingest.sentry.io/4506033503797248",
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
