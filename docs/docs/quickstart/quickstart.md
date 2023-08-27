# Write a simple test case

If you are interested in running a quick Colab example, you can [click here](https://colab.research.google.com/drive/1Lfq5geYsvfVoquDqv84UkWS57SdAHm30?usp=sharing).

You can write a simple test case as simply as:

```python
import os
import openai
from deepeval.metrics.factual_consistency import assert_factual_consistency

openai.api_key = "sk-XXX"

# Write a sample ChatGPT function
def generate_chatgpt_output(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "The customer success phone line is 1200-231-231 and the customer success state is in Austin."},
            {"role": "user", "content": query}
        ]
    )
    llm_output = response.choices[0].message.content
    return llm_output

def test_factual_consistency():
    query = "What is the customer success phone line?"
    context = "Our customer success phone line is 1200-231-231."
    output = generate_chatgpt_output(query)
    assert_factual_consistency(output, context)

# Just run the following code in Python if required
test_factual_consistency()
```

### Running it in Pytest

To run this in Pytest, just run:

```python
# sample.py
import pytest

def test_factual_consistency():
    query = "What is the customer success phone line?"
    context = "Our customer success phone line is 1200-231-231."
    output = generate_chatgpt_output(query)
    assert_factual_consistency(output, context)
```

You can then run it in CLI using:

```bash
python -m pytest test_sample.py
```
