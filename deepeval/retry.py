"""Simple Retry class
"""
import time


class Retry:
    def __init__(self, max_retries=3, delay=1):
        self.max_retries = max_retries
        self.delay = delay

    def __enter__(self):
        self.retry_count = 0
        while self.retry_count < self.max_retries:
            try:
                return self  # Return the context manager instance.
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
