# Experimental features

`DEEPEVAL_MODE=experimental` opts a user into features that are not final and
may change or be removed between releases. `stable` is the default. Gate code
with `is_experimental()` from `deepeval/config/mode.py` (Python) or
`isExperimental()` from `typescript/src/config/mode.ts` (TypeScript), never by
comparing the env var directly.

Experimental code should not blend in with stable code. Prefix everything that
exists only for an experimental feature with `_experimental_` (template files,
template method keys, private helpers) so it can be found and deleted with a
single `rg _experimental_`. Shared entry points that need an experimental
branch (settings, CLI, base classes) are listed per feature below instead.

## Current experimental features

### TypeSafe AI Jev for metric decisions (Python only)

Routes the decision step of a metric to TypeSafe AI's System One model (Jev)
instead of the LLM: QAG yes/no verdicts, DAG binary and non-binary judgements,
and the G-Eval score. The LLM still extracts items, generates evaluation steps
and writes reasons. In experimental mode there is no LLM fallback: a missing
`TYPESAFE_API_KEY` or `typesafe-sdk` raises and tells the user to configure it
or switch back to `stable`.

How each decision maps onto Jev:

- QAG verdict per item: Noul, `P(yes) >= 0.5`; a 0.35-0.65 band is `borderline`
  where the metric allows it.
- DAG `BinaryJudgementNode`: Noul on the node criteria, `P >= 0.5 -> True`.
- DAG `NonBinaryJudgementNode`: Choice over the child verdict options.
- G-Eval: strict mode is one Noul (0/1); no rubric is one Noul per evaluation
  step, `mean(P)` mapped onto the score range; a rubric is one Score over the
  rubric levels (max 10) mapped onto the score range. The reason is one LLM call
  fed the probabilities.

Where it lives, in the order you would remove it:

| What | Where |
| --- | --- |
| QAG question templates | `deepeval/metrics/*/templates/_experimental_system_one_verdict.txt` (15 files), compiled into `templates.json` via `scripts/compile_metric_templates.py` |
| G-Eval templates | `_experimental_system_one_{strict_verdict,step_verdict,rubric_score,reason}.txt` in `deepeval/metrics/g_eval/templates/` and `deepeval/metrics/conversational_g_eval/templates/`; `Reason` schema in both `schema.py` files |
| Per-metric wiring | `_experimental_system_one_spec()` helper and `system_one=` kwarg at each decision call in the 15 QAG metrics, `GEval`, `ConversationalGEval`, and the four DAG judgement node classes (`dag/nodes.py`, `conversational_dag/nodes.py`); `self.system_one_model = initialize_system_one_model()` in the 15 QAG metrics, `GEval`, `ConversationalGEval`, `DAGMetric`, `ConversationalDAGMetric` |
| Template method keys | `_experimental_system_one_*` entries in `MetricTemplateMethod`, `deepeval/templates/resolver.py` |
| Decision branches | `SystemOne{Binary,Choice,Score}Spec`, `_system_one_active`, `_jsonable` and the `system_one` kwargs in `deepeval/metrics/utils/decision.py` (the stable `generate_*_judgement` / `generate_rubric_score` helpers stay); `SystemOneVerdictSpec`, `verdict_from_probability`, `_system_one_*` and the `system_one` kwarg in `deepeval/metrics/utils/qag.py` |
| Model resolution | `initialize_system_one_model()` in `deepeval/metrics/utils/models.py`; `system_one_model` attribute on `BaseMetric` / `BaseConversationalMetric` in `deepeval/metrics/base_metric.py` |
| Model class | `deepeval/models/system_one/` and `DeepEvalBaseSystemOneModel` in `deepeval/models/base_model.py`; exports in `deepeval/models/__init__.py` |
| Provider plumbing | `ProviderSlug.TYPESAFE` (`deepeval/constants.py`), `TYPESAFE_ERROR_POLICY` (`deepeval/models/retry_policy.py`), `TYPESAFE_*` fields in `deepeval/config/settings.py` and `deepeval/key_handler.py`, `TypeSafeModel` label in `deepeval/cli/diagnose/diagnose.py` |
| CLI | `deepeval/cli/providers/system_one/` (`set-typesafe`, `unset-typesafe`), imported from `deepeval/cli/providers/__init__.py` |
| Docs | `docs/content/integrations/models/typesafe-ai.mdx` and its `meta.json` entry |

Metrics wired: Faithfulness, TurnFaithfulness, Summarization (alignment),
AnswerRelevancy, Hallucination, ContextualPrecision, TurnContextualPrecision,
PromptAlignment, ArgumentCorrectness, Bias, Toxicity, Misuse, NonAdvice,
PIILeakage, RoleViolation, GEval, ConversationalGEval, DAGMetric and
ConversationalDAGMetric judgement nodes.

Not wired (LLM in both modes): contextual recall/relevancy (verdict prompt also
decomposes), conversational single-verdict metrics, ArenaGEval, DAG
`TaskNode` (generates text).
