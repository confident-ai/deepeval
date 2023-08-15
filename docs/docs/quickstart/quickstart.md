# Write a simple test case

You can write a simple test case as simply as:

```python
# test_sample.py
# test files must start 
from deepeval.test_utils import assert_llm_output

def generate_llm_output(input: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output

def test_llm_output():
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    assert_llm_output(output, expected_output, metric="entailment")
```

You can then run it in CLI using this: 

```bash
python -m pytest test_sample.py
```


