# Tracking Configuration

If you would like to track additional metadata about the run and store it onto DeepEval's web UI platform, you can additionally store it in a `run_configuration` fixture.

:::note

A `fixture` is a fundamental part of Pytest, which allows for setup code to be written for tests. It's a function decorated with `@pytest.fixture` that returns the data you want. Once a fixture is created, it can be used by mentioning it as a parameter in the test functions. The fixture function gets invoked and its result can be used across multiple tests. This helps in writing more efficient and scalable tests by avoiding code duplication and promoting reusability.

:::

You can log configuration by adding the following to your test script

```python
# test_bias.py
@pytest.fixture
def run_configuration() -> dict:
    return {"model": "gpt2"}
```

:::warning
Make sure that you return a JSON-decodable object. Failure to do so will cause the test to not log.
:::
