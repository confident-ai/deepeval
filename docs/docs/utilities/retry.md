# Retry

With LLMs, they can sometimes output different things especially if they have a high temperature. It can often be worth running the same test multiple times.

While it's fairly easy to write a simple re-try loop, DeepEval does offer a built-in re-try loop already due to the amount of time it can take to actually re-try them.

```python
from deepeval.retry import retry

@retry(max_retries=5, min_success=2)
def retry_function():
    test_factual_consistency()

retry_function()
```

Sometimes you may want to update teh input

```python
def fail_hook(e):
    print(f"Error: {str(e)}")

@retry(max_retries=5, min_success=2, fail_hooks=[fail_hook])
def retry_function_with_fail_hook():
    test_factual_consistency()

retry_function_with_fail_hook()
```



## Parameters

- `max_retries` - The maximum number of retries with errors
- `min_success` - The minimum number of times it must succeed
- `delay` - The amount of delay in seconds between a retry if it errors
- `fail_hooks` - A list of functions to be called when an error occurs during a retry attempt


