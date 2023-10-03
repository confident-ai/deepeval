"""Simple Retry class
"""
import time
from functools import wraps
from typing import Callable, List


class retry:
    retry_count: int = 0
    success_count: int = 0

    def __init__(
        self,
        max_retries: int = 3,
        delay: int = 1,
        min_success: int = 1,
        fail_hooks: List[Callable] = None,
    ) -> None:
        self.max_retries = max_retries
        self.delay = delay
        self.min_success = min_success
        self.fail_hooks = fail_hooks if fail_hooks else []

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
                    for fail_hook in self.fail_hooks:
                        fail_hook(e)
                except Exception as e:
                    print(
                        f"Attempt {self.retry_count} to execute {func.__name__} failed due to Exception: {str(e)}"
                    )
                    last_error = e  # Update the last error
                    for fail_hook in self.fail_hooks:
                        fail_hook(e)

                if self.retry_count < self.max_retries:
                    print(
                        f"Retrying execution of {func.__name__} in {self.delay} seconds..."
                    )
                    time.sleep(self.delay)
                else:
                    if self.retry_count != 1:
                        # No need to print it out if you can only retry once
                        print(
                            f"Max retries ({self.max_retries}) for executing {func.__name__} exceeded."
                        )
                    if self.success_count < self.min_success:
                        if last_error is not None:
                            raise last_error  # Raise the last error

        return wrapper


def retry_with_input_modification(self, func, modify_input):
    """
    This function allows you to retry a function with modified input if it fails.

    Example:
    ```python
    def modify_input(*args, **kwargs):
        # Modify the input here
        # In this example, we add a word to the input string
        args = tuple(arg + " word" for arg in args)
        return args, kwargs

    @retry_with_input_modification(modify_input=modify_input)
    def test_function(input_string):
        # Function implementation here
    ```
    """

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
                for fail_hook in self.fail_hooks:
                    fail_hook(e)
                args, kwargs = modify_input(*args, **kwargs)
            except Exception as e:
                print(
                    f"Attempt {self.retry_count} to execute {func.__name__} failed due to Exception: {str(e)}"
                )
                last_error = e  # Update the last error
                for fail_hook in self.fail_hooks:
                    fail_hook(e)
                args, kwargs = modify_input(*args, **kwargs)

            if self.retry_count < self.max_retries:
                print(
                    f"Retrying execution of {func.__name__} in {self.delay} seconds..."
                )
                time.sleep(self.delay)
            else:
                if self.retry_count != 1:
                    # No need to print it out if you can only retry once
                    print(
                        f"Max retries ({self.max_retries}) for executing {func.__name__} exceeded."
                    )
                if self.success_count < self.min_success:
                    if last_error is not None:
                        raise last_error  # Raise the last error

    return wrapper
