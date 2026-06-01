# Google ADK trace schemas

Captured trace JSON snapshots used by `test_sync.py` and `test_async.py`. Each `*_schema.json` here is the structural fixture for one test method — `assert_trace_json` compares the live trace produced by Gemini + the OpenInference instrumentor against this file with the relaxed structural matcher in `tests/test_integrations/utils.py`.

## Regenerating schemas

These files are LIVE-CAPTURED — never hand-edit them. Regenerate via:

```bash
GOOGLE_API_KEY=... GENERATE_SCHEMAS=true \
  poetry run pytest tests/test_integrations/test_googleadk/test_sync.py \
                    tests/test_integrations/test_googleadk/test_async.py
```

The `GENERATE_SCHEMAS=true` flag flips `trace_test(...)` (defined in the package conftest) from `assert_trace_json` to `generate_trace_json`, which writes the captured trace dict to the schema path instead of asserting against it. Each test still runs end-to-end through Gemini, so the schemas reflect a real ADK execution.

For the evals iterator test, regenerate separately (it doesn't write a schema, but exercising it confirms the metric stash path):

```bash
GOOGLE_API_KEY=... OPENAI_API_KEY=... \
  poetry run pytest tests/test_integrations/test_googleadk/test_evaluate_agent.py
```

## When to regenerate

- The OpenInference Google ADK instrumentor's attribute namespace changes (e.g. semconv-genai migration): every `*_schema.json` will drift in lockstep — regenerate the full directory.
- `OpenInferenceSpanInterceptor`'s `_serialize_framework_attrs` adds / renames a `confident.*` attr: regenerate.
- Google ADK adds new event types / span shapes (e.g. an additional `chain` wrapper around `LlmAgent`): regenerate.

If a single test drifts but the others don't, you almost always want to investigate the test rather than regenerate — schema drift is an early warning that the trace shape changed in a way the matcher couldn't absorb. The matcher already tolerates LangChain v1.x-style `usage_metadata` / `response_metadata` drift and unordered span/tool-call lists; if you're hitting drift outside those allowances, it's signal.

## What's covered

| Schema | Source test | Notes |
| --- | --- | --- |
| `googleadk_simple_schema.json` | `test_sync.py::TestSimpleApp::test_simple_greeting` | Greeting; agent + LLM spans, no tools. |
| `googleadk_tool_schema.json` | `test_sync.py::TestToolApp::test_tool_calculation` | Single calculator tool call. |
| `googleadk_tool_metric_collection_schema.json` | `test_sync.py::TestToolApp::test_tool_metric_collection` | Same shape as `tool` but with `next_tool_span(metric_collection=...)` populating `confident.span.metric_collection` on the tool span. |
| `googleadk_multiple_tools_weather_schema.json` | `test_sync.py::TestMultipleToolsApp::test_multiple_tools_weather_only` | Single `get_weather` call from a multi-tool agent. |
| `googleadk_multiple_tools_time_schema.json` | `test_sync.py::TestMultipleToolsApp::test_multiple_tools_time_only` | Single `get_time` call from the same multi-tool agent. |
| `googleadk_parallel_tools_schema.json` | `test_sync.py::TestMultipleToolsApp::test_parallel_tool_calls` | `get_weather` + `get_time` called for the same city. Span / tool-call ordering is matcher-unordered. |
| `googleadk_features_sync.json` | `test_sync.py::TestDeepEvalFeatures::test_full_features_sync` | All POC migration features stacked: trace `metric_collection` override, `next_agent_span(metrics=[...])`, `next_llm_span(metric_collection=...)`, and `update_current_span(metric_collection=...)` from inside `special_tool`. |
| `googleadk_async_simple_schema.json` | `test_async.py::TestAsyncSimpleApp::test_async_simple_greeting` | Async path through `runner.run_async(...)`. |
| `googleadk_async_tool_schema.json` | `test_async.py::TestAsyncToolApp::test_async_tool_calculation` | Async tool call. |
| `googleadk_async_parallel_tools_schema.json` | `test_async.py::TestAsyncMultipleToolsApp::test_async_parallel_tool_calls` | Async parallel tools. |
| `googleadk_features_async.json` | `test_async.py::TestDeepEvalFeaturesAsync::test_full_features_async` | Async equivalent of `googleadk_features_sync.json`. |

## Sanity-check before committing

After regenerating, scan the diff for:

1. **Empty traces**: a `*_schema.json` that's `{}` (or near-empty) means `trace_testing_manager.wait_for_test_dict()` timed out — the spans were probably routed to OTLP instead of REST. Re-check that the test isn't running outside an `@observe` / `evals_iterator` context AND that the integration's `ContextAwareSpanProcessor` is correctly attached. `assert_trace_json` has a guard against this (`_assert_trace_capture_succeeded`), so the test would already have been failing.
2. **Missing `confident.span.tools_called`**: tool calls dropped → either the OpenInference instrumentor stopped emitting them on the LLM output messages, or `_extract_tool_calls` has drifted from the OpenInference message shape.
3. **`type` vs `spanType` flips**: deepeval's serializer key for span type drift is a known compatibility gate; the matcher is tolerant but a wholesale flip means an upstream version bump.
