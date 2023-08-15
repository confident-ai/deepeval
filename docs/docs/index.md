
# DeepEval

DeepEval provides a Pythonic way to run offline evaluations on your LLM pipelines so you can launch comfortably into production.

## Introducing PyTest for LLMs

For any Python developer building production-grade apps, it is common to set up PyTest as the default testing suite as it provides a clean interface to quickly write tests and continue development. 

While common for Python software engineers to use PyTest, it is often uncommon for many machine learning engineers as their feedback is often in the form of an evaluation loss.

It is therefore important then to build a new type of testing framework for LLMs to ensure engineers can keep iterating on their prompts, agents and LLMs.


## Why we wrote this library

While the growth of LLMs, LangChain, LlamaIndex became prominent- we found that once these pipelines were built, it became really hard to continue iterating on these pipelines. Many engineers wanted to use LangChain as a quick start and then start adding guardrails, switch LLMs to Llama2. 
