# LLM-Evaluation

LLM-Based evaluation is useful for more specific types of answers.

DeepEval makes it easy to output a metric.

```python
import openai
from deepeval.metrics.llm_eval import LLMEval
from deepeval.test_cases import LLMTestCase

openai.api_key = openai_api_key

def make_chat_completion_request(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

metric = LLMEval(
    criteria="How funny it is",
    completion_function=chatgpt
)

test_case = LLMTestCase(output="Mobile phones are rocks.")
score = metric.measure(test_case)
```

By default, it only outputs the score.

## Under the hood

Under the hood, it uses LLM evaluation.
