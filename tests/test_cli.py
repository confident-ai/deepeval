"""Test with the following command and setup:

In pyproject.toml, include?:

```

[tool.pytest.ini_options]
addopts = "-m 'not skip_test'"
markers = [
    "skip_test: skip the test",
]
```

Now:

- both `pytest tests/test_cli.py` and `deepeval test run tests/test_cli.py` should all tests,
- `pytest tests/test_cli.py -m 'not skip_test'` should run the test, and
- `deepeval test run tests/test_cli.py -m skip_test` should run the test.

"""

import pytest


@pytest.mark.skip_test
def test_does_run_with_override():
    assert True
