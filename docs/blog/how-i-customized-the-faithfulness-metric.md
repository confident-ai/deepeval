---
title: How I Customized DeepEval's Faithfulness Metric
description: As the open-source LLM evaluation framework, DeepEval offers everything from evaluating LLM agents to generating synthetic datasets required for evaluation.
slug: how-i-customized-the-faithfulness-metric
authors:
  name: Lei Wang
  title: DeepEval Enthusiast
  url: https://github.com/realei
  image_url: https://github.com/realei.png
tags: [community]
hide_table_of_contents: false
---

As a [___] at [___], I first stumbled across DeepEval when I was [___]. DeepEval is an open-source LLM evaluation framework and I found it great, it [___offer some praises here if you want__]. But it had a problem. The faithfulness metric I was most interested in didn't fit exactly my use case because [___].

And so I decided to customize it.

## What is the faithfulness metric?

The [faithfulness metric](/docs/metrics-faithfulness) in `deepeval` is a metric that detects hallucination in RAG pipelines by assessing whether there are any contradictions between the retrieved text chunks in your retriever and the final generated text.

[__around two more paragraph of explanation here explaning the algorim and explaining saying why you needed it__]

## A small problem with DeepEval's metrics

Although the faithfulness metric... [__start talking about how it wasn't good enough, perhaps for custom models, custom decision criteria, this also makes the previous section of talking and explaining the algorithm very important because otherwise readers wouldn't know what they are reading here__]

## How prompt injection fixed faithfulness

But the problem wasn't un-fixable. Since the faithfulness metric is an [LLM-as-a-judge metric](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method), the solution is simple - fix the prompt template, and we're good to go. But how?

[__start talking about custom examples for in-context learning, maybe more guidance in the output format for valid JSONs, etc__]

[__can also talk about how in the end you included the prompt templates by mirroring how it is done on deepeval's github__]

## End result

[__anything you want to talk about here__]
