# Bulk Running Test Cases

You can run a number of test cases which you can define either through CSV or through our hosted option.

```python

from deepeval import BulkTestRunner, TestCase

class BulkTester(BulkTestRunner):
    @property
    def bulk_test_cases(self):
        return [
            TestCase(
                input="What is the customer success number",
                expected_output="1800-213-123",
                tags=["Customer success"]
            ),
            Testcase(
                input="What do you think about the models?",
                expected_output="Not much - they are underperforming.",
                tags=["Machine learning"]
            )
        ]

tester = BulkTester()
tester.run(callable_fn=generate_llm_output)

```

## From CSV

You can import test cases from CSV.

```python
import pandas as pd
df = pd.read_csv('sample.csv')
from deepeval import TestCases
# Assuming you have the column names `input`, `expected_output`
class BulkTester(BulkTestRunner):
    @property
    def bulk_test_cases(self):
        return TestCases.from_csv(
            "sample.csv",
            input_column="input",
            expected_output_column="output",
            id_column="id"
        )

```
