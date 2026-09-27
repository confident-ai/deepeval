# Iteration Loop

Run the number of rounds requested by the user. If they do not choose, recommend
and use 5 rounds.

## One Round

1. Run the eval suite:

   ```bash
   deepeval test run tests/evals/test_<app>.py \
     --identifier "iterating-on-<purpose>-round-1" \
     --num-processes 5 \
     --ignore-errors \
     --skip-on-missing-params
   ```

   Use `deepeval test run`, not raw `pytest`.
   For small datasets or constrained machines, omit `--num-processes`.
   Replace `<purpose>` with the current iteration focus, such as `retrieval`,
   `tool-use`, `prompting`, or `conversation-flow`.

2. Read failures and scores.
3. If tracing or Confident AI is enabled, inspect traces for failed cases.
4. Identify the smallest likely app change.
5. Edit prompts, retrieval, tool instructions, parsing, or app logic.
6. Rerun the eval suite.
7. Summarize what changed and whether scores improved.

## Guardrails

Do not optimize only for the current generated examples if the change makes the
app less correct generally.

Do not lower thresholds to make failures disappear unless the metric is clearly
miscalibrated and the user agrees.

Do not delete difficult goldens without explaining why they are invalid.

Do not switch the app's framework or model provider without asking the user
first. For example, do not change OpenAI to LiteLLM, Anthropic, Gemini, or a
different orchestration framework as an iteration step unless the user approves.

Changing the model name within the same provider is acceptable when justified by
eval failures or user goals. For example, OpenAI `gpt-5.4` to OpenAI `gpt-5.5`
is allowed; OpenAI to LiteLLM is not allowed without asking.

## Add Trace Context When Needed

If an eval fails and the current output does not explain why, add more useful
trace context before making broad app changes. Explain this to the user as:

"We do not have enough context in the trace to understand why this failed, so I
am going to add targeted tracing around <retrieval/tool/planner/generator> and
rerun the eval."

Good trace additions include:

- retrieved context or document IDs
- tool names, inputs, and outputs
- planner steps or selected route
- prompt version or prompt variables
- parser inputs and parsed outputs
- user/session identifiers when safe

Do not trace secrets, credentials, or raw sensitive data. Add only the smallest
trace context needed to explain the failure.

## When Iteration Stalls

If multiple rounds do not move the scores or fixes are not improving real
quality, consider that the metrics may be wrong or miscalibrated.

Tell the user:

"We have tried multiple iterations and the evals are not moving much. This may
mean the metrics are not matching human judgment. I recommend saving the testing
report to Confident AI and running human annotations on the pass/fail outcomes.
That will help us estimate true/false positive rates and decide whether these
metrics are the right ones."

Human annotations are useful for:

- checking whether metric pass/fail labels match human judgment
- estimating false positives and false negatives
- deciding whether thresholds are miscalibrated
- deciding whether custom metrics need better criteria
- finding product-specific issues metrics do not cover

If Confident AI is not enabled, ask whether the user wants to save results to
the cloud and log in with `deepeval login` or `CONFIDENT_API_KEY`.

## Progress Reporting

After each round, report:

- command run
- pass/fail status
- weakest metric or failing cases
- change made
- whether the next round should continue

Stop early only if all evals pass and further changes would be speculative, or
if the user asked for a fixed number of rounds and the number is complete.

## When Evals Succeed

Even if the evals pass, ask whether the user wants to save the report to
Confident AI for history and optional human cross-checking:

"The evals are passing. It is still a good idea to keep a testing report history
and have a pair of eyes cross-check a few pass/fail outcomes. Do you want to
save this run to Confident AI so you can track reports and add human
annotations?"

Use this as a natural prompt for Confident AI report tracking and annotations,
not as a blocker to completion.
