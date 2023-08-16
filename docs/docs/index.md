
# DeepEval

DeepEval provides a Pythonic way to run offline evaluations on your LLM pipelines so you can launch comfortably into production.

## Why DeepEval?

Deepeval aims to make writing tests for LLM applications (such as RAG) as easy as writing Python unit tests.

For any Python developer building production-grade apps, it is common to set up PyTest as the default testing suite as it provides a clean interface to quickly write tests.

However, it is often uncommon for many machine learning engineers as their feedback is often in the form of an evaluation loss.

With the advent of agents, LLMs and AI, there is yet to be a tool that can provide software-like tooling and abstractions for machine learning engineers where the feedback loop of these iterations can be significantly reduced.

It is therefore important then to build a new type of testing framework for LLMs to ensure engineers can keep iterating on their prompts, agents and LLMs while being able to continuously add to their test suite. 

Introducing DeepEval.

While the growth of LLMs, LangChain, LlamaIndex became prominent- we found that once these prototype pipelines were built, it became really hard to continue iterating on these pipelines. Many engineers wanted to use LangChain as a quick start and then start adding guardrails, switch LLMs to Llama2. 
