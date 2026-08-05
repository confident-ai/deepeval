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

212 MDX files: 119 docs, 35 integrations, 20 guides, 17 tutorials, 17 blog,
4 changelog. [`languages-audit.json`](languages-audit.json) lists the 78 of
them that actually carry a `<Switch>`, `<Term>` or `<Only>` tag, with each
page's url, declared `languages`, tag counts and code-fence tallies; a
`languages` declaration alone does not put a page in the file. It is a hand-run
snapshot, not generated at build time, so treat it as of its `generatedAt` date.

**Page-level support is declared and validated.** Every page in `content/docs`
and `content/integrations` carries a `languages` frontmatter field, enforced by
a strict Zod schema in [`source.config.ts`](source.config.ts) — 81 declare
`[python, typescript]`, 72 declare `[python]` and 1 declares `[typescript]`.
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
metrics. `getting-started-agents` and `getting-started-chatbots` stay `[python]`.

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

**Code blocks are done where they remain.** 287 `<Switch>` blocks, 251 of which
carry a TypeScript fence; the other 36 are `bash` and `yaml` pairs. None is
one-sided — `validate-terms` now enforces that — and no Python-only page leaks
a TypeScript fence.

**Prose has barely started.** `<Term>` covers 128 spans across 7 pages, while
1,188 snake_case identifiers across 66 files (293 distinct) are still hardcoded
Python in prose that TypeScript readers see. That count is over the pages a
TypeScript reader can still reach, so it fell with every page that went
`[python]` rather than through any conversion work. The mechanism is now
settled — see [Inline prose terms](#inline-prose-terms) — so what remains is
volume, not design.

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

## Integration coverage

Audited against `typescript/src` at `0.1.32`. This records what the SDKs
support; frontmatter tracks what each page actually *contains*, so a page stays
`[python]` until its TypeScript examples exist.

| Group | TypeScript SDK surface | Docs verdict |
| --- | --- | --- |
| Frameworks | `langchain` (LangGraph included, via `integrations/langchain/langgraph-utils.ts`), `openai-agents`, `openai` | all 12 pages `[python]` — 4 are eligible and awaiting a rewrite, the other 8 permanently Python-only |
| Models | every provider except LiteLLM — `LocalModel` covers both LM Studio and vLLM, `KimiModel` is Moonshot, `GeminiModel` takes `useVertexAI` | **done** — 14 of 16 pages bilingual, `ai-sdk` is TypeScript-only, `litellm` permanently Python-only |
| Vector databases | none | all 6 permanently Python-only |

`models/ai-sdk` is the site's first `languages: [typescript]` page. The rule it
replaces — "no integration page should be `[typescript]`" — held only while every
documented integration existed in Python; `AISDKModel` has no Python counterpart.

### What blocks wider TypeScript coverage

**Existing pages missing TypeScript examples.** The SDK already supports these,
so the work is writing the `<Switch>` blocks and then widening `languages`.

- `frameworks/langchain`, `frameworks/openai` and `frameworks/openai-agents`.
  These three were bilingual and have been taken back to `[python]`: their 15
  `<Switch>` blocks were collapsed to their Python case and the TypeScript
  cases deleted. The examples were written against the SDK rather than read off
  it, so the whole framework story is being redocumented from scratch instead of
  patched snippet by snippet. This is the one place where deleting working
  content was the right call — a plausible-looking `deepeval/integrations/*`
  import that does not resolve costs a reader more than an honest 501.
- `frameworks/langgraph`, covered by `integrations/langchain/langgraph-utils.ts`.

The shared component-level snippet
([`snippets/evaluation/component-level-agent-framework-tabs.mdx`](snippets/evaluation/component-level-agent-framework-tabs.mdx))
is where this gap is now visible to readers rather than only to us. Its Python
case is the twelve-framework tab set; its TypeScript case is manual `observe`
instrumentation plus a `<NotImplemented>` for `deepeval.integrations`, so a
TypeScript reader on `getting-started` is told the adapters ship but their pages
do not exist yet. Closing any of the four eligible framework pages above means
filling that case in with real tabs. Note the snippet lives outside `content/`,
so `validate-terms` does not scan it and its `<NotImplemented>` is not in the
count the script prints.

**Integrations with no page at all.** These need writing from scratch; no amount
of metadata helps.

- **Mastra** — `integrations/mastra`, exporting `DeepEvalExporter`.
  TypeScript-only.
- **Vercel AI SDK as a framework** — `integrations/ai-sdk`, the tracing exporter.
  Distinct from `models/ai-sdk`, which covers `AISDKModel` as a judge model and
  is written.
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

## Outstanding

Roughly in order of how much they hurt a TypeScript reader.

1. **Prose identifiers.** The 1,188 spans above. The design question is
   answered (see [Inline prose terms](#inline-prose-terms)); this is now a
   page-by-page conversion job with no blocker in front of it.
2. **Integrations, per [Integration coverage](#integration-coverage).** The
   models half is now bilingual, so a TypeScript reader keeps Integrations'
   largest section; frameworks and vector databases both vanish entirely.
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
5. **No TypeScript-only pages.** `languages: [typescript]` parses and filters
   correctly but nothing uses it, so the case is untested. Mastra and the Vercel
   AI SDK are the natural first users once written.
6. **No preference persistence.** The selection is `useState` with no storage,
   so it resets to Python on every refresh, new tab, and search-result landing.
   Since most docs traffic arrives directly on an interior page, a TypeScript
   reader currently has to re-select on nearly every visit. Needs localStorage
   plus a pre-hydration script to avoid a flash.
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
- The reader's preference is never overridden, only the current page's
  rendering. Selecting TypeScript, passing through a Python-only page, and
  landing on a bilingual one keeps the selection intact.
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
