"""Simple Retry class
"""
import time

"""Simple Retry class
"""
import time


class Retry:
    def __init__(self, max_retries=3, delay=1, min_success=1):
        self.max_retries = max_retries
        self.delay = delay
        self.min_success = min_success

    def __enter__(self):
        self.retry_count = 0
        self.success_count = 0
        while self.retry_count < self.max_retries:
            try:
                yield self  # Yield the context manager instance.
                self.success_count += 1
                if self.success_count >= self.min_success:
                    return  # Operation succeeded the required number of times.
            except AssertionError as e:
                print(f"Attempt {self.retry_count + 1} failed: {str(e)}")
                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    print(f"Max retries ({self.max_retries}) exceeded.")
                    raise e
            except Exception as e:
                print(f"Attempt {self.retry_count + 1} failed: {str(e)}")
                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    print(f"Max retries ({self.max_retries}) exceeded.")
                    raise e

    def __exit__(self, exc_type, exc_value, traceback):
        return False
