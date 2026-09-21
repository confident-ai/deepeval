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

### TypeSafe Jev for QAG verdicts (Python only)

Routes the yes/no decision step of QAG metrics to TypeSafe's System One model
(Jev) instead of the LLM. The LLM still extracts items and writes reasons. In
experimental mode there is no LLM fallback: a missing `TYPESAFE_API_KEY` or
`typesafe-sdk` raises and tells the user to configure it or switch back to
`stable`.

Where it lives, in the order you would remove it:

| What | Where |
| --- | --- |
| Per-metric question templates | `deepeval/metrics/*/templates/_experimental_system_one_verdict.txt` (15 files), compiled into `templates.json` via `scripts/compile_metric_templates.py` |
| Per-metric wiring | `_experimental_system_one_spec()` helper, `system_one=` kwarg on `generate_qag_verdicts` calls, and `self.system_one_model = initialize_system_one_model()` in each of the 15 metric `__init__`s |
| Template method key | `_experimental_system_one_verdict` in `MetricTemplateMethod`, `deepeval/templates/resolver.py` |
| QAG branch | `SystemOneVerdictSpec`, `verdict_from_probability`, `_system_one_*` helpers and the `system_one` kwarg in `deepeval/metrics/utils/qag.py` |
| Model resolution | `initialize_system_one_model()` in `deepeval/metrics/utils/models.py`; `system_one_model` attribute on `BaseMetric` / `BaseConversationalMetric` in `deepeval/metrics/base_metric.py` |
| Model class | `deepeval/models/system_one/` and `DeepEvalBaseSystemOneModel` in `deepeval/models/base_model.py`; exports in `deepeval/models/__init__.py` |
| Provider plumbing | `ProviderSlug.TYPESAFE` (`deepeval/constants.py`), `TYPESAFE_ERROR_POLICY` (`deepeval/models/retry_policy.py`), `TYPESAFE_*` fields in `deepeval/config/settings.py` and `deepeval/key_handler.py`, `TypeSafeModel` label in `deepeval/cli/diagnose/diagnose.py` |
| CLI | `deepeval/cli/providers/system_one/` (`set-typesafe`, `unset-typesafe`), imported from `deepeval/cli/providers/__init__.py` |
| Docs | `docs/content/integrations/models/typesafe.mdx` and its `meta.json` entry |

Metrics wired: Faithfulness, TurnFaithfulness, Summarization (alignment),
AnswerRelevancy, Hallucination, ContextualPrecision, TurnContextualPrecision,
PromptAlignment, ArgumentCorrectness, Bias, Toxicity, Misuse, NonAdvice,
PIILeakage, RoleViolation.

Not yet wired (would follow the same pattern): contextual recall/relevancy
(verdict prompt also decomposes), conversational single-verdict metrics, DAG
judgement nodes (Choice), G-Eval rubric scoring (Score).
