"""Simple Retry class
"""
import time
from functools import wraps


class retry:
    retry_count: int = 0
    success_count: int = 0

    def __init__(
        self, max_retries: int = 3, delay: int = 1, min_success: int = 1
    ) -> None:
        self.max_retries = max_retries
        self.delay = delay
        self.min_success = min_success

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None  # Store the last error
            while self.retry_count < self.max_retries:
                try:
                    self.retry_count += 1
                    for _ in range(self.min_success - self.success_count):
                        result = func(*args, **kwargs)
                        self.success_count += 1
                except AssertionError as e:
                    print(
                        f"Attempt {self.retry_count} to execute {func.__name__} failed due to AssertionError: {str(e)}"
                    )
                    last_error = e  # Update the last error
                except Exception as e:
                    print(
                        f"Attempt {self.retry_count} to execute {func.__name__} failed due to Exception: {str(e)}"
                    )
                    last_error = e  # Update the last error

                if self.retry_count < self.max_retries:
                    print(
                        f"Retrying execution of {func.__name__} in {self.delay} seconds..."
                    )
                    time.sleep(self.delay)
                else:
                    print(
                        f"Max retries ({self.max_retries}) for executing {func.__name__} exceeded."
                    )
                    if self.success_count < self.min_success:
                        if last_error is not None:
                            raise last_error  # Raise the last error

        return wrapper
