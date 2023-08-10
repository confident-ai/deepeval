# Evals

Evals provides a Pythonic way to run offline evaluations on your LLM pipelines.

You can get started with adding an a ground truth like so:

```python
import os
from evals import Evaluator
# Grab API key from app.twilix.io
os.environ['TWILIX_API_KEY'] = "..."
eval = Evaluator()
eval.add_ground_truth(
    query="How do you contact them?",
    expected_response="You can contact our help center at 1800-000-000",
    tags=["Customer success"]
)
```
