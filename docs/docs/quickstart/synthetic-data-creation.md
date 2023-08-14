# Synthetic Data Creation

Generating synthetic queries allows you to quickly evaluate the queries related to your prompts.
We help developers get up and running with example queries from just raw text.

```python
# Loads the synthetic query model to generate them based on data you get.
# These automatically create synthetic queries and adds them to our online database
evaluator.generate_queries(
    texts=["Our customer success phone line is 1200-231-231"],
    tags=["customer success"]
)
```
