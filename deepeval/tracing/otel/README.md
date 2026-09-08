# Native OTEL / confident-trace bridge

Pydantic AI, Strands, AgentCore, and Google ADK retain their existing DeepEval
entry points. Framework enablement and framework-specific extraction remain in
those adapters. Shared code handles live contexts, ownership, tree construction,
and routing:

- `provider.py` installs one router on each provider and selects an adapter by
  scope/integration, with parent inheritance for unclassified children. Capture
  state is shared across adapters on that provider.
- `live_context.py` supplies typed placeholders for `update_current_*` and
  `next_*_span`, and restores their contexts.
- `capture.py` binds each span to its trace and evaluation session at start and
  ingests ended spans synchronously. Production transport is selected at start;
  evaluation does not depend on a background batch export or context lookup.

A DeepEval wrapper owns the lifecycle of a mixed trace. Bare OTEL evaluation
entries finish after their captured spans end. Iterator boundaries mark unfinished
spans as errors and release their live state before scoring or resetting the
session. Late callbacks cannot become production uploads or join a later run.
The synthetic synchronous iterator wrapper is removed without discarding siblings;
trace metrics receive all roots when a trace has more than one root.

DeepEval-created providers preserve normal production sampling and record
iterator spans regardless of their production sampler. Application providers and
exporters are not replaced. Applications supplying a provider must configure it
to record evaluation spans; an empty asynchronous capture raises an evaluation
setup error rather than returning a successful empty run.

## Developing against unpublished confident-trace

There is deliberately no dependency on a nonexistent package release. Development
uses the sibling confident-trace source checkout, including the new private
`confident_trace._core.attachment.attach_native` hook. That hook stamps recognized
native integration scopes and forwards processor callbacks synchronously. It
installs no exporter, enables no instrumentation, and changes no global provider.

From the DeepEval repository, run the contract and public iterator tests with:

```sh
PYTHONPATH=../confident-trace/python/src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DEEPEVAL_TELEMETRY_OPT_OUT=YES \
.venv/bin/python -m pytest \
  tests/test_core/test_tracing/test_otel_bridge \
  tests/test_integrations/test_pydanticai/test_instrumentation_compatibility.py \
  --import-mode=importlib -q
```

The source tests invoke the real confident-trace decorator and consume its
`spec/genai-vectors.json`; they do not substitute hand-built spans for that
contract check. They skip when the source package is unavailable. Other bridge
tests run without confident-trace, including every combination of the four
adapters and the synchronous / asynchronous iterator execution paths.

Without the private hook, DeepEval attaches its router directly to the OTEL
provider. This supports existing installations, including Python 3.9. The
attachment was developed against confident-trace revision
`52fa2302e00927a96160e4dc0fe55bd8b7ac5449` plus the attachment change on
`migration/tracing`. Before making confident-trace a required dependency, publish
that change, verify its dependency range, and add the Python-version-conditional
release pin. Do not ship a relative filesystem dependency.

An application that separately calls `confident_trace.init(...)` still owns that
initializer's exporter. DeepEval routes its own copy exactly once; it does not
silence application-owned exporters.

## Validation baseline

Five existing Google ADK `TestFrameworkAttrExtraction` tests fail identically on
the starting DeepEval commit and this migration (flattened LLM messages, token
counts, tool calls, tool results, and trace input/output). These failures are not
used as passing bridge coverage. New tests assert completed span trees and metric
execution independently. Live network/model evaluations are not needed for this
suite; the Pydantic compatibility tests use its local test model.

Two legacy exporter tests also reproduce baseline failures in this sandbox because
their prompt fixtures fetch Confident API commits. They are network-dependent;
they are not counted as successful offline validation.

## Remaining callback and wrapper integrations

The same optional bridge now backs LangChain/LangGraph, LlamaIndex, CrewAI,
OpenAI Agents, OpenAI, and Anthropic. Their existing DeepEval imports, constructors,
callback registration, decorators, and instrumentation functions remain the public
entry points. Hugging Face's training/evaluation callback is not a tracing adapter
and is unchanged.

`frameworks.py` attaches each selected confident-trace emitter to a private
runtime. It does not call `confident_trace.init()` or replace the global OTEL
provider. If the application has already initialized confident-trace, the bridge
uses that runtime's provider and preserves application-owned instrumentation and
exporters. Without the unpublished `instrument_framework` contract, the existing
DeepEval implementations remain active. OpenAI Agents also retains its existing
implementation if confident-trace's optional OpenInference Agents extra is absent.
There is still no package-release dependency or relative-path dependency.

Compatibility details:

- LangChain handlers enroll only runs receiving the explicit callback. Root runs
  remain agent spans. Handler metrics stay on the trace; LLM metadata and staged
  `next_*` options supply component metrics, with staged options taking precedence.
- LlamaIndex retains the supplied dispatcher, including additional dispatchers,
  and records local/custom model calls as well as provider-backed calls. Delegated
  `chat`/`complete` calls form one LLM span.
- CrewAI retains metric options on its wrapper classes and tools. Reset clears
  integration state while leaving instrumentation enabled. Native LLM subclasses
  are covered even when constructed before instrumentation.
- OpenAI Agents retains explicit processor registration, agent/tool metric
  wrappers, and its native supplemental model wrapper for custom models. The
  processor delegates framework spans to confident-trace's OpenInference bridge.
- OpenAI and Anthropic preserve synchronous, asynchronous and streaming clients;
  unpatch restores only owned patches. Stream termination and early close finish
  the captured span synchronously.

The remaining-framework tests use real framework calls, local models and mock HTTP
transports. They cover the public iterator's sync, async and scheduled-task paths,
LangGraph fanout, component metric precedence, stream lifecycle, reset/re-enabling,
and application-exporter ownership. Run them in an environment containing the
frameworks, confident-trace's runtime dependencies (including `wrapt`), and its
OpenAI Agents extra. Tests skip unavailable optional SDKs; they must run without
skips in the full framework validation environment:

```sh
PYTHONPATH=../confident-trace/python/src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DEEPEVAL_TELEMETRY_OPT_OUT=YES \
python -m pytest \
  tests/test_core/test_tracing/test_otel_bridge/test_frameworks.py \
  --import-mode=importlib -q
```
