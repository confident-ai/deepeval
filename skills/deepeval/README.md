# DeepEval Skill

This skill helps coding agents add reliable DeepEval evaluation workflows to AI
applications. It covers app inspection, dataset generation or reuse, pytest
eval-suite creation, tracing, Confident AI reporting, and iterative improvement.

## Use When

- Adding evals to an LLM, RAG, chatbot, or agent application
- Generating synthetic goldens with `deepeval generate`
- Creating a committed `tests/evals` pytest suite
- Enabling DeepEval tracing or Confident AI reports
- Iterating on prompts, tools, retrieval, or agent behavior from eval failures

## Workflow Summary

1. Inspect the target app and existing DeepEval usage.
2. Ask the required intake questions.
3. Reuse existing metrics and datasets when available.
4. Generate or import goldens.
5. Add minimal tracing and a pytest eval suite.
6. Run `deepeval test run`.
7. Iterate for the requested number of rounds, defaulting to 5.

See [SKILL.md](./SKILL.md) for the agent instructions.
