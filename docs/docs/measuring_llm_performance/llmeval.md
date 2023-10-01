# LLM-Evaluation

LLM-Based evaluation is useful for more specific types of answers such as how funny the LLM is where you want to use the LLM to score and critique it.

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
    completion_function=make_chat_completion_request
)

test_case = LLMTestCase(output="Mobile phones are rocks.")
score = metric.measure(test_case)
```

By default, it only outputs the score. If you would like to include a reason, you can set `include_reason` to `True`.

```
metric.measure(test_case, include_reason=True)
```
