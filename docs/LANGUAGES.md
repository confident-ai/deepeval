# Python / TypeScript docs migration

DeepEval is becoming Python- and TypeScript-native, and the docs have to follow.
This document tracks where that migration stands. Authoring conventions live in
[`.cursor/rules/docs-languages.mdc`](../.cursor/rules/docs-languages.mdc); this
is the status and rationale.

## Objectives

- **Both languages are first-class.** TypeScript is not an appendix to a Python
  site. Where the SDKs have parity, the docs should have parity.
- **The reader picks once.** One selection applies across the site rather than
  being re-made per page.
- **Gaps are explicit, never silent.** A page that cannot serve the reader's
  language leaves the sidebar, and shows a "501 Not Implemented" page if reached
  anyway. Quietly rendering the other language is the one outcome we treat as a
  bug.
- **Support is declared, not inferred.** Which languages a page covers is
  metadata validated at build time, not something guessed from its code fences.

## Current state

215 MDX files: 119 docs, 37 integrations, 20 guides, 17 tutorials, 18 blog,
4 changelog. [`languages-audit.json`](languages-audit.json) lists the 78 of
them that actually carry a `<Switch>`, `<Term>` or `<Only>` tag, with each
page's url, declared `languages`, tag counts and code-fence tallies; a
`languages` declaration alone does not put a page in the file. It is a hand-run
snapshot, not generated at build time, so treat it as of its `generatedAt` date.

**Page-level support is declared and validated.** Every page in `content/docs`
and `content/integrations` carries a `languages` frontmatter field, enforced by
a strict Zod schema in [`source.config.ts`](source.config.ts) — 96 declare
`[python, typescript]`, 57 declare `[python]` and 3 declare `[typescript]`.
Guides, tutorials, changelog and blog may omit the field and simply follow the
reader's preference; 19 of the 20 guides and 13 of the 17 tutorials declare
`[python]` anyway. The five that don't are the four tutorial index pages, which
stay undeclared so they cannot strand a `[python]` child, and
`guides-llm-observability`, which is prose with no code fence in it.

**The sidebar follows the selection.** `languages` rides along on each page-tree
node, and [`lib/lang/page-tree.ts`](lib/lang/page-tree.ts) prunes the tree to the
reader's language before Fumadocs renders it. Folders and separator groups
collapse once they empty out, so selecting TypeScript today drops the whole
"Synthetic Data Generation" and "Prompt Optimization" sections, plus the
framework and vector-database halves of Integrations, rather than leaving
headings over dead links. The same pruned tree drives the prev/next footer. Readers still reach the
501 page through direct links, bookmarks and search hits.

**The selector only appears on docs and integrations.** Those are the reference
surfaces where the choice is meaningful, and they are now the only surfaces with
anything to switch. Guides and tutorials carry no language tag at all: the 187
`<Switch>` blocks that were spread across 26 of those pages have been collapsed
to their Python case, and every one of the 26 now declares `languages: [python]`.
Those pages are narrative rather than reference — each walks through one Python
codebase end to end — so a half-translated walkthrough served the reader worse
than an honest 501, which is what a TypeScript reader now gets.

**Within docs, the workflow pages went the same way.** Twelve more pages are
now `[python]`: the 5-minute quickstart and its five use-case siblings, and the
evaluation workflow set — `evaluation-introduction`, single- and multi-turn
end-to-end (with the folder index that holds them), component-level, and CI/CD.
That collapsed 88 more `<Switch>` blocks and 17 `<Term>` spans. What a
TypeScript reader keeps is the part that is reference rather than walkthrough:
every concepts page including test cases, datasets and tracing, every individual
metric page, the Others group, and integrations. The cost is that many of the
pages they do keep link into one of these 12, mostly to
`evaluation-component-level-llm-evals`, and those links now land on a 501.

`getting-started`, `getting-started-rag`, `getting-started-llm-arena` and
`getting-started-mcp` have since been brought back to `[python, typescript]`.
They are the walkthroughs in the group whose narrative does not depend on a
single Python codebase — install, write a test case, run the CLI, add tracing —
so each step has a real TypeScript counterpart rather than a half-translation.
`getting-started` is also the page a TypeScript reader is most likely to arrive
on first, and a 501 there reads as "this SDK is not ready" rather than "this page
is not written". The other three followed it because every API each reaches for
already exists in the TypeScript SDK: for RAG, all five RAG metrics and their
four turn-level counterparts, the CSV/JSON golden loaders, `evalsIterator()` and
`ConversationSimulator`; for arena, `ArenaTestCase`, `Contestant`, `ArenaGEval`
and `compare()`; for MCP, `MCPServer`, the three call classes, and all three MCP
metrics.

`getting-started-chatbots` followed for the same reason — `ConversationalTestCase`,
`Turn`, `ConversationalGolden`, `ConversationSimulator`, `TurnRelevancyMetric` and
`KnowledgeRetentionMetric` all ship in TypeScript. Its Python `simulate()` example
was also stale (`goldens=`, `max_turns=`, and a `deepeval.conversation_simulator`
import path that no longer exists) and was corrected to
`conversational_goldens=` / `max_user_simulations=` off `deepeval.simulator`.

`evaluation-unit-testing-in-ci-cd` followed the agents page, since it shares the
`cicd-agent-framework-tabs` snippet and every CLI surface it documents exists in
TypeScript: `npx deepeval test run`, its `-o/--official` flag, `official` and
`hyperparameters` on `evaluate()`, `flaky` on both test case classes, and
`logHyperparameters()` as the counterpart to `@deepeval.log_hyperparameters`. Its
multi-turn Python example also carried the same stale
`deepeval.conversation_simulator` import as the chatbots page and was corrected to
`deepeval.simulator`.

Its `<FAQs>` block is the one part that stayed language-neutral prose rather than
being switched. `question` is typed `string` (it seeds the React key, the
`<details name>` group, and the FAQPage JSON-LD `name`), so a `<Term>` cannot go
there at all. And a `<Switch>` inside an `answer` would be walked by `extractText`,
which concatenates every child — so both languages' text would land in the
schema.org JSON-LD as one contradictory blob. Answers were reworded to say "your
language's test runner" and "unit testing" instead of naming `pytest` /
`assert_test()`, which keeps the structured data clean for crawlers.

`evaluation-end-to-end-single-turn` completes the evaluation section. Both of its
approaches convert cleanly: the iterator has the same five-vs-six option split
described under the component-level page, and `evaluate()` takes its two mandatory
arguments positionally in TypeScript with the rest in a third options object (no
`asyncConfig`, since metric execution is always concurrent and awaited).

Two headings were deliberately **left** with Python spellings —
`## Approach 1: evals_iterator() with tracing (recommended)` and
`## Approach 2: evaluate()`. A `<Term>` in a heading breaks, and these headings
are anchor targets linked from `evaluation-component-level-llm-evals`,
`evaluation-end-to-end-llm-evals`, and this page's own body, so renaming them
would break cross-page links for a cosmetic gain.

The page also carries a pre-existing `[TODO: ...]` authoring note inside the sync
vs async callout. That callout is `AsyncConfig`-specific, so it now lives in the
Python case and the TODO stays Python-side rather than being inherited by
TypeScript readers.

`evaluation-end-to-end-multi-turn` prompted the extraction of
`snippets/evaluation/model-callback-tabs.mdx`. The six-tab `model_callback` block
was duplicated across this page, `evaluation-unit-testing-in-ci-cd`, and
`getting-started-chatbots`. The first two were byte-identical apart from
`showLineNumbers={true}` vs `showLineNumbers`, but the chatbots copy had
**silently drifted and was broken**: a stray `"` in the fence meta, `title=main.py`
unquoted, a `model_callback` annotated `-> str` that returned a `Turn`, a missing
`from typing import List` under a `List[Turn]` annotation, three tabs missing
`from deepeval.test_case import Turn` while returning `Turn(...)`, and a LangChain
example still on `RunnableWithMessageHistory` rather than `create_agent`.
Centralizing fixed all of it by construction — the snippet carries the
end-to-end/CI-CD version.

The TypeScript case gives the callback three tabs rather than a
`<NotImplemented>` wholesale, because the Python "Python" and "OpenAI" tabs are
not deepeval integrations at all — they just call a chat client inside the
callback, which ports directly. LangChain was added for the same reason: the tab
is `createAgent` + `MemorySaver` keyed by `threadId`, with no deepeval adapter
involved. LlamaIndex, OpenAI Agents and Pydantic AI have no tab because their
JS thread/session APIs don't line up one-to-one with the Python ones shown.
Note the TS `ModelCallback` takes a
**single object** `{ input, turns, threadId }` and always passes all three, unlike
Python's optional argument injection, so callout prose was reworded to "take only
the ones you need" instead of "may optionally accept".

While converting, two stale Python bugs were fixed on this page and in the CI/CD
page: `from deepeval.conversation_simulator import ConversationSimulator` (the
module is `deepeval.simulator`) and a `simulate(goldens=..., max_turns=...)` call
whose real parameters are `conversational_goldens=` and `max_user_simulations=`.

TypeScript has no `AsyncConfig` and no `ConversationSimulator(max_concurrent=...)`
— the simulator runs every golden concurrently via `Promise.all` — so the
Async/Sync tabs and the concurrency note are per-language rather than shared.

`evaluation-end-to-end-llm-evals/index` is the cheapest conversion so far and a
useful shape to recognize: a **concepts page with no code fences at all**, so it
needed zero `<Switch>` blocks — only `<Term>` on test case field names
(`actual_output`, `retrieval_context`, `tools_called`, `expected_outcome`,
`user_description`) and on the iterator/observe spellings. Two of its `<Term>`s
sit inside markdown table cells, which renders fine (`environment-variables`
already does this).

Prose that named an API purely to refer to a workflow was reworded to name the
workflow instead — "the recommended evals iterator path", "the iterator form is
single-turn only", "using `pytest`/`vitest` and `deepeval test run`" — since a
`<Term>` per mention would have been noise where no reader needs the exact
identifier. The Single-Turn vs Multi-Turn table needed no changes at all: every
symbol in it (`LLMTestCase`, `Golden`, `ConversationalGolden`, `BaseMetric`,
`ConversationSimulator`) is spelled identically in both SDKs.

`evaluation-component-level-llm-evals` went bilingual once its three includes
(`load-dataset`, `component-level-agent-framework-tabs`,
`sub-agent-framework-tabs`) already were, leaving only prose and the CI/CD tail.
Two shape differences are called out per language rather than smoothed over:
TypeScript's `evalsIterator()` takes five options in a single object and has no
`asyncConfig` or `cacheConfig`, because it is an async generator with no sync
variant — so the Async/Sync bullets above the integration tabs are wrapped in
`<Only id="python">`. It does accept `hyperparameters` directly, which is why the
Hyperparameters section switches to passing them to the iterator instead of
mirroring Python's `@deepeval.log_hyperparameters` decorator.

Its mermaid diagram was **neutralized rather than switched**. `<Term>` does not
render inside a code fence, and duplicating a 17-line sequence diagram per
language is noise, so participants and messages were reworded to concepts
("Evals iterator", "Open a test run over the dataset"). This follows
`evaluation-introduction`, whose diagram already labels participants
conceptually.

`getting-started-agents` is bilingual, and phase two has now landed for the four
shared framework snippets. `component-level-agent-framework-tabs`,
`cicd-agent-framework-tabs` and `end-to-end-agent-framework-tabs` each give
TypeScript its own seven-tab set — Manual Instrumentation, LangChain, Mastra,
LangGraph, OpenAI, OpenAI Agents, Vercel AI SDK — and `sub-agent-framework-tabs`
gives it five, dropping OpenAI and the Vercel AI SDK because neither opens an
agent span to stage onto. Mastra sits third rather than last: the tab order is a
promotion decision, not an alphabetical or per-SDK-module one. The `<NotImplemented>` blocks are gone from all four.
The two blockers recorded here earlier are also gone: `nextAgentSpan` ships in
`tracing/pending-context.ts` and `AgentSpanContext.metrics` is live in
`tracing/trace-context.ts`.

Two asymmetries remain by design. `evalsIterator()` is async-only, so the
Async/Sync split stays a Python-only concept and the TypeScript tabs state that
once at the top rather than nesting a second tab layer. And the tab lists differ
per language because the adapter sets do: Pydantic AI, AgentCore, Strands,
Anthropic, LlamaIndex, Google ADK and CrewAI are Python-only, while the Vercel AI
SDK and Mastra are TypeScript-only. OpenInference ships in `typescript/src` but
has no page to link, so it is not a tab yet.

The dead per-tab `<Case id="typescript">` blocks that used to sit inside the
Python case of `cicd-` and `end-to-end-` were deleted rather than promoted. They
were unreachable, and they had drifted: `import { metrics } from "deepeval"`
namespace access, `test_langchain_app.ts` snake-case filenames, and a
CI/CD snippet whose "TypeScript" half ran `evalsIterator` instead of the
`toPass()` matcher the surrounding page is about.

All four replaced their hand-rolled judge-model tab set with
`snippets/models/configure-llm-judge-tabs.mdx`, which is bilingual and already
maintained. The old tabs enumerated eight Python `deepeval.models` classes each,
which had no TypeScript half and would have had to be written three more times.

A few spots on these pages are worth knowing about:

- The RAG install `<details>` is a `<Switch>` because the TUI is a `[inspect]`
  extra in Python and an optional `ink` dependency in TypeScript — a real
  packaging difference, not a docs gap.
- RAG's retriever and generator spans call
  `updateCurrentSpan({ input, retrievalContext })` directly rather than passing a
  `testCase:`. TypeScript's `LLMTestCase` requires `actualOutput` and a retriever
  span has none, and this is how `evaluation-llm-tracing` already writes them.
- Arena's "log prompts and models" is a `<Switch>` over a real SDK difference,
  not a gap. Both languages merge each `Contestant`'s `hyperparameters` onto that
  contestant's test run, but a `Prompt` has to reach the platform as a version
  reference: Python pushes an unpushed prompt for you, while TypeScript's
  normalizer is a sync path that can only warn, so the TypeScript snippet pulls
  first. Writing this section also corrected the Python snippet, which passed
  `hyperparameters=` to `compare()` — that parameter exists in neither SDK; the
  field lives on `Contestant`.
- The MCP page's GitHub links to `examples/mcp_evaluation/*.py` are `<Only
  id="python">`, on the same reasoning as `getting-started`'s full example: they
  link a `.py` file, not a docs page. There is no `typescript/examples/mcp` yet.

The wrappers on them mark differences in the *SDKs*, not in the docs' own coverage:
the retry section is a `<Switch>` because TypeScript delegates retries to the
provider client, and the full example is `<Only id="python">` because it links a
`.py` file. Links into pages that are still `[python]` — the use-case
quickstarts, the synthesizer, component-level and end-to-end evals, the
custom-LLM guide — are left shared, and a TypeScript reader who follows one gets
the 501 with its switch button. Rewriting the sentence per language to route
around a link states a permanent difference where there is only an unwritten
page, and leaves prose to re-merge by hand once the page lands. The 501 is the
mechanism for that gap; `<Switch>` is not.

`evaluation-flags-and-configs` was in that set and has since been brought back
to `[python, typescript]`. It is reference rather than walkthrough — a flag and
field index, section by section — so it splits cleanly per language in a way the
workflow pages did not, and it was the most-linked-to member of the group. It is
also the one page where a *silent* gap was most expensive: a reader who cannot
see which fields their SDK honors will pass one that does nothing. Fields with
no TypeScript counterpart (`results_subfolder`, `inspect_after_run`, HTML
export, `-r`, the `on_test_run_end` hook) are `<Only id="python">` rather than
flagged as missing, and the two `AsyncConfig` fields the TS runner accepts but
ignores (`runAsync`, `throttleValue`) are documented for neither.

**Code blocks are done where they remain.** 523 `<Switch>` blocks across 91
files, most carrying a TypeScript fence and the rest `bash` / `yaml` / prose
pairs. None is one-sided — `validate-terms` now enforces that — and no
Python-only page leaks a TypeScript fence.

**Prose is well underway.** `<Term>` covers 1,419 spans (130 distinct
spellings) across 71 files. The remaining hardcoded snake_case in shared prose
has not been recounted since the framework rewrites; the mechanism is settled —
see [Inline prose terms](#inline-prose-terms) — so what is left is volume, not
design.

## Inline prose terms

`<C>` used to take an id into a registry in `lib/lang/terms.ts`, which paired
the two spellings of a single identifier. That registry is gone, and the tag
was renamed to `<Term>`; both spellings are now written at the call site as
`<Term py="actual_output" ts="actualOutput"/>`.

The name stays capitalized despite naming an inline span. A lowercase
`<term>` was tried and reverted: MDX compiles a lowercase tag written in the
markdown flow to a literal DOM element (`_jsx("term", …)`) instead of a
component-map lookup, so 89 of 98 spans rendered as empty `<term>` elements
with the build passing clean. The other 9 — the ones inside the `<FAQs>` JSX
prop — compiled to `_jsx(_components.term, …)` and worked, which is what
makes lowercase disqualifying rather than merely wrong: the failure is
silent and depends on where the tag sits.

The registry's unit was the identifier, but the unit authors actually write is
the code span. That mismatch is why the authoring rule had to forbid `<Term>` for
"anything with parens, operators, dotted access, or assignment" — not a style
preference, but the limit of what a lookup can substitute. And the forbidden
set is where much of the remaining work lives: `touch test_example.py`,
`deepeval test run test_example.py`, `metric.measure(test_case)`. A registry
has no expressible answer for those, and splitting them into a shared backtick
span plus a switched one renders as two adjacent `<code>` elements with a seam
down the middle.

Two things were given up, both smaller than they look:

- **Single-edit mass-fix.** One registry entry used to cover every call site.
  Now a wrong spelling is `rg 'ts="…"'` plus a substitution — the values are
  exact strings, so this stays mechanical.
- **Fail-loud on a bad id.** `getTerm` threw on an unknown id at build time.
  That check only ever caught typos in keys the registry itself forced authors
  to invent. What it never caught — a plausible but wrong TypeScript spelling —
  is the failure that actually reaches readers, and it is unchanged either way
  (Outstanding item 4).

The registry's one substantive guarantee, that a Python spelling has exactly
one TypeScript counterpart, is enforced instead by
[`scripts/validate-terms.mjs`](scripts/validate-terms.mjs), which runs in
`prebuild`. It also fails on a tag missing either attribute — MDX props are not
typechecked, so nothing else covers that — on a `<Term>` nested inside a
`<Case>` or `<Only>`, where the wrapper has already fixed the language, and on a
`<Switch>` missing a case for some language.

Coverage stays measurable: remaining raw snake_case spans in prose is the same
metric [Current state](#current-state) already tracks, and it does not depend
on a registry existing.

`lowerCodeTerms` became `lowerTerms` in [`lib/lang/term.ts`](lib/lang/term.ts),
sitting alongside the prop names it has to agree with. It now parses attributes
rather than matching a fixed id, so attribute order does not matter, and it
restores the character references mdast emits for attribute quotes. A tag
missing its `py` spelling throws, which keeps the `/llms.*` routes and page
builds failing loudly on a malformed tag.

Self-mapping terms were dropped rather than migrated. In the registry a
self-map documented intent ("identical in both, deliberately"); inline it is
just a duplicated string, so `LLMTestCase`, `deepeval` and friends went back to
plain backtick spans.

## Import order in TypeScript test files

Every TypeScript example that is a test file orders its imports so the two
Vitest lines bracket the block:

- `import { it, expect } from "vitest";` is **first**, above every `deepeval`
  import. `it` and `expect` are generic test-runner plumbing that carries no
  `deepeval` meaning, so it belongs out of the way at the top rather than
  interleaved with the imports a reader is actually there to look at.
- `import "deepeval/vitest";` is **last**. It is a side-effecting import with no
  bindings — it registers the `toPass()` matcher — and putting it on the closing
  line is what marks the file as a test file at a glance.

Everything else (`deepeval/metrics`, `deepeval/dataset`, `deepeval/tracing`,
local application imports) sits between them in whatever order reads best.

This is presentational, not functional: `npx deepeval test run` injects the
matcher regardless, so `deepeval/vitest` only matters when the file is run with
`vitest` directly. Nine examples were reordered to this convention across
`getting-started`, `evaluation-introduction`, `evaluation-datasets`,
`getting-started-rag` and `cicd-agent-framework-tabs`. Two of those blocks carry
a `{6}` line highlight; reordering preserved the line count, so the highlight
still lands on the same statement.

## Integration coverage

Audited against `typescript/src` at `0.1.32`. This records what the SDKs
support; frontmatter tracks what each page actually *contains*, so a page stays
`[python]` until its TypeScript examples exist.

| Group | TypeScript SDK surface | Docs verdict |
| --- | --- | --- |
| Frameworks | `openai`, `integrations/langchain` (LangGraph included, via `integrations/langchain/langgraph-utils.ts`), `integrations/openai-agents`, `integrations/mastra`, `integrations/ai-sdk`, `integrations/openinference` | **done bar one** — of 14 pages, 4 are bilingual (`langchain`, `langgraph`, `openai`, `openai-agents`), 2 are TypeScript-only (`mastra`, `ai-sdk`), 8 are permanently Python-only; only OpenInference has no page |
| Models | every provider except LiteLLM — `LocalModel` covers both LM Studio and vLLM, `KimiModel` is Moonshot, `GeminiModel` takes `useVertexAI` | **done** — 14 of 16 pages bilingual, `ai-sdk` is TypeScript-only, `litellm` permanently Python-only |
| Vector databases | none | all 6 permanently Python-only |

`models/ai-sdk` is the site's first `languages: [typescript]` page. The rule it
replaces — "no integration page should be `[typescript]`" — held only while every
documented integration existed in Python; `AISDKModel` has no Python counterpart.

`frameworks/mastra` is the second, and the first `[typescript]` page in the
Frameworks group. Writing it meant the group could no longer be wrapped whole in
`<Only id="python">` on the integrations index: the heading and intro are now
shared and the card grid is a `<Switch>`, so a TypeScript reader gets a
Frameworks section with Mastra in it rather than no section at all.
`frameworks/ai-sdk` — the tracing exporter, distinct from the `models/ai-sdk`
judge model — is the third.

`frameworks/openai` and `frameworks/langgraph` were the last two rewrites, and
both were written by reading `typescript/src` rather than from the Python page.
Two things that surfaced doing so are on the pages themselves rather than
smoothed over: `instrumentOpenAI` patches the *first* client it is given and
returns early after that, which a reader constructing two clients would
otherwise hit silently; and `LlmSpanContext` is scope-wide in TypeScript while
`nextLlmSpan(...)` is the one-shot form, so the OpenAI page documents both
rather than presenting `setTracingContext` as a straight translation of Python's
`with trace(llm_span_context=...)`.

Writing them also added a `./prompt` subpath export to `typescript/package.json`
(mirrored into `typesVersions`). `Prompt` had been root-only, so
`frameworks/openai-agents` was already documenting an
`import { Prompt } from "deepeval/prompt"` that threw
`ERR_PACKAGE_PATH_NOT_EXPORTED`. The subpath now exists and the root export
stays for call sites that predate it; note `deepeval/test-case` exports an
unrelated MCP `type Prompt`, which is why the specifier matters.

Two things surfaced while writing it, both since fixed in the SDK rather than
documented around. `DeepEvalExporter` disabled itself when `CONFIDENT_API_KEY`
was unset, which made a keyless local eval score nothing; the OTel integrations
(`ai-sdk`, `openinference`) had the same gate one level up, returning no span
processors at all. And both OTel processors overwrote a span's `input`/`output`
with `undefined` when the OTel attributes carried none, silently erasing values
staged by `next*Span(...)` or `updateCurrentSpan(...)`. See
[Keyless operation](#keyless-operation).

### What blocks wider TypeScript coverage

**Existing pages missing TypeScript examples.** None left in Frameworks. The
four eligible pages — `langchain`, `openai-agents`, `openai`, `langgraph` — are
all bilingual, each rewritten from `typescript/src` rather than patched snippet
by snippet, on the reasoning that a plausible-looking `deepeval/integrations/*`
import that does not resolve costs a reader more than an honest 501.

The four shared framework snippets under `snippets/evaluation/` used to be the
place where the docs understated what ships — their TypeScript case was manual
`observe` instrumentation plus a `<NotImplemented>` claiming the adapters had no
pages. All four now carry real TypeScript tab sets; see the
`getting-started-agents` notes above for what each includes and why the tab lists
differ per language. Note the snippets live outside `content/`, so
`validate-terms` does not scan them and any `<NotImplemented>` tags they carry
are not in the count the script prints.

**Integrations with no page at all.** These need writing from scratch; no amount
of metadata helps.

- **OpenInference** — `integrations/openinference`, present in both SDKs.

### Documented ahead of the SDK

The model pages describe three behaviours the TypeScript SDK does not have yet.
This was a deliberate call — the docs describe the intended surface and the SDK
is being brought up to meet it — but until it lands, these pages overstate what
ships:

- **`deepeval set-*` / `unset-*` commands.** `typescript/src/cli/index.ts`
  implements only `gate`. Every model page shows a `### Command Line` section
  outside the `<Switch>`, so TypeScript readers see commands that currently
  print "Unknown command".
- **`USE_*` provider routing.** `initializeModel` in
  `typescript/src/metrics/utils.ts` turns any string into an `OpenAIModel`, with
  no provider dispatch. The `set-*` commands are inert without it, so the two
  have to land together. The per-page ENV tabs work around this today by
  constructing the provider class directly rather than passing a bare string.
- **`.env.local` autoloading.** `typescript/src/utils.ts` calls a bare
  `dotenv.config()`, which reads `.env` only. Python loads `.env`, then
  `.env.{APP_ENV}`, then `.env.local`. The shared "Setting Up Your API Key"
  prose claims the Python behaviour for both.

**Settled as permanently Python-only.** No TypeScript equivalent exists, so
these keep `languages: [python]` and need no revisiting: `models/litellm`, all
six `vector-databases/*`, and the framework pages `agentcore`, `anthropic`,
`crewai`, `google-adk`, `huggingface`, `llamaindex`, `pydanticai` and `strands`.

### Keyless operation

Every integration works without a Confident AI API key, in both SDKs. This is
load-bearing for the docs: each integration page opens with a local eval, and a
page that has to say "log in first" before its first snippet is describing a
worse product than the one that ships.

Python has been explicit about this for a while — `DeepEvalInstrumentationSettings`
documents `api_key` as "fully optional" and gates only the outbound
`x-confident-api-key` header. TypeScript had drifted, in three places:

- `DeepEvalExporter` (Mastra) set a `disabled` flag in its constructor and
  dropped every span, so a keyless local eval scored nothing.
- `createDeepEvalProcessors` (`ai-sdk`) and `createOpenInferenceProcessors`
  returned `[]`, which removed the *local* span processor along with the OTLP
  exporter. The local one is what evals read; only the exporter needs a key.
- `isConfident()` printed `console.error("Confident AI API key not found.")` as
  a side effect of being asked a question, so the now-normal keyless path
  logged an error per call.

The fix keeps one rule: **a missing key removes the upload and nothing else.**
`postTrace` already skipped uploading and said so, so nothing above it needed to
know. For the OTel integrations there is one extra wrinkle — with no exporter
installed, routing a span to OTLP drops it on the floor — so `resolveSpanRoute`
takes `otlpEnabled` and routes in-process when no transport exists.

Ts Integration CI does not inject `CONFIDENT_API_KEY`, so every suite exercises
the keyless path by default.

## Outstanding

Roughly in order of how much they hurt a TypeScript reader.

1. **Prose identifiers.** Hardcoded Python identifiers still sitting in prose a
   TypeScript reader sees, on the pages `<Term>` has not reached. The design
   question is answered (see [Inline prose terms](#inline-prose-terms)); this is
   now a page-by-page conversion job with no blocker in front of it, and the
   count wants re-running before it is quoted again.
2. **Integrations, per [Integration coverage](#integration-coverage).** The
   models half is bilingual, the frameworks half now keeps six pages for a
   TypeScript reader, and the four shared framework tab snippets have real
   TypeScript tabs, so what is left is OpenInference (no page). Vector
   databases vanish entirely and always will.
3. **Partial gaps.** Roughly 116 unpaired ` ```python ` fences sit on 28
   otherwise bilingual pages, mostly framework integration tabs, the pytest and
   `deepeval test run` workflows, and Pydantic schemas. Each needs
   its TypeScript case written, or `<Only id="python">` around it until someone
   does — today they render to TypeScript readers as if they were theirs.
4. **Unverified TypeScript spellings.** `validate-terms` proves the docs agree
   with themselves, not that they agree with the SDK
   (`typescript/package.json`, currently `0.1.32`). A spelling that is wrong
   on every page passes. The spellings on the four converted pages were read
   off `typescript/src` by hand, which is the standard the authoring rule sets,
   but it is a convention rather than a check. Resolving each `ts` value
   against the SDK's exported type surface would close this; it was not
   possible against the old registry either, so nothing regressed.
5. **TypeScript-only pages are barely exercised.** Three now — `models/ai-sdk`,
   `frameworks/mastra` and `frameworks/ai-sdk` — and none of them is load-bearing
   for a sidebar group any more, since the Frameworks group holds six pages for
   a TypeScript reader.
6. **No preference persistence across visits.** The selection is `useState`
   with no cookie/localStorage, so a new tab or hard refresh does not remember
   the last pick. Mono-language pages now SSR and open in their only language
   (TS-only → TypeScript, Python-only → Python) via the generated
   `page-languages` map, and bilingual pages still default to Python — so a
   crawler hitting Mastra indexes real content instead of a 501. Soft-nav onto
   a mono-language page adopts that language; bilingual pages keep the current
   preference. Cross-visit persistence still needs localStorage (or a cookie)
   plus a pre-hydration script if we want to avoid resetting bilingual pages
   to Python on every landing.
7. **`llms.txt` is Python-only.** `lowerTerms` in
   [`lib/lang/term.ts`](lib/lang/term.ts) hardcodes Python when lowering
   `<Term>` tags. `languages` now makes per-language output possible.
8. **No build-time content check.** Nothing asserts that a `languages: [python]`
   page contains no TypeScript fence, or that a page with one declares it, so
   the metadata can drift from the content.

## What this migration changed

Replacing the `<PythonOnly />` marker component with `languages` frontmatter:

- Page support moved from a body-level React side effect to build-validated
  metadata. A missing or misspelled value now fails `next build` with a file
  path instead of silently doing nothing.
- A page that cannot serve the reader's language now renders a "501 Not
  Implemented" page (with a playable dinosaur, because why not) instead of its
  content. Previously it silently forced Python and greyed the option out of
  the dropdown, which left a reader unsure whether they had hit a gap or a bug.
- Unsupported pages leave the sidebar rather than sitting there as links to a
  501. Keeping that honest needs one invariant — a page may never support a
  language its parent folder's index page doesn't, or it would disappear along
  with the parent and have nothing left linking to it — so
  `assertPageTreeLanguages` in [`lib/lang/validate.ts`](lib/lang/validate.ts)
  walks every section's tree at build time and fails with the offending URLs.
- Soft-navigating onto a **mono-language** page adopts that page's only
  language so the reader (and SSR) never land on a 501 for a page that can
  serve them. Bilingual pages keep the current preference across navigations.
- The selector is scoped to docs and integrations, which is now also the only
  place a language tag appears at all. Inheriting the choice had made the 187
  `<Switch>` blocks in guides and tutorials reachable for the first time, and
  that is what showed the TypeScript half was not worth keeping: both sections
  walk through a single Python codebase, so the blocks were reduced to their
  Python case and all 26 pages declared `[python]`. A TypeScript reader gets a
  501 there instead of a walkthrough that switches its snippets but not its
  narrative.
- Clamping became local to the page subtree, so the root `pythonOnlyPage` state
  and its mount/unmount flash are gone.

Separately, `<LangSwitch>` became
[`<Switch>`](components/lang/switch.tsx) with explicit `<Case id="…">`
children, rewritten across all 493 call sites by
[`scripts/langswitch-to-switch-case.mjs`](scripts/langswitch-to-switch-case.mjs).
The old component picked a child by index, which forced Python first and
TypeScript second with nothing validating the order, limited each side to a
single element, and could not distinguish the 32 same-language pairs (`bash`,
`yaml`). Naming the language removes all three constraints, and lets
`validate-terms` require that a `<Switch>` cover every language — a one-sided
block is an `<Only>`, not a `<Switch>` with a case missing.

Unchanged: the partial-gap backlog has not moved.
