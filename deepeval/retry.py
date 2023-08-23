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
            while self.retry_count < self.max_retries:
                try:
                    self.retry_count += 1
                    for _ in range(self.min_success - self.success_count):
                        func(*args, **kwargs)
                        self.success_count += 1
                    return
                except AssertionError as e:
                    print(f"Attempt {self.retry_count} failed: {str(e)}")
                except Exception as e:
                    print(f"Attempt {self.retry_count} failed: {str(e)}")

                if self.retry_count < self.max_retries:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    print(f"Max retries ({self.max_retries}) exceeded.")
                    if self.success_count < self.min_success:
                        raise Exception(
                            "Operation failed the required number of times."
                        )

        return wrapper
