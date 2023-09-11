"""Simple Retry class
"""
import time
from functools import wraps


class retry:
    def __init__(self, max_retries=3, delay=1, min_success=1):
        self.max_retries = max_retries
        self.delay = delay
        self.min_success = min_success

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.retry_count = 0
            self.success_count = 0
            last_error = None  # Store the last error
            while self.retry_count < self.max_retries:
                try:
                    self.retry_count += 1
                    for _ in range(self.min_success - self.success_count):
                        result = func(*args, **kwargs)
                        self.success_count += 1
                except AssertionError as e:
                    print(f"Attempt {self.retry_count} failed: {str(e)}")
                    last_error = e  # Update the last error
                except Exception as e:
                    print(f"Attempt {self.retry_count} failed: {str(e)}")
                    last_error = e  # Update the last error

                if self.retry_count < self.max_retries:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    print(f"Max retries ({self.max_retries}) exceeded.")
                    if self.success_count < self.min_success:
                        if last_error is not None:
                            raise last_error  # Raise the last error

        return wrapper
