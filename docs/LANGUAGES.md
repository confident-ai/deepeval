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

211 MDX files: 119 docs, 34 integrations, 20 guides, 17 tutorials, 17 blog,
4 changelog.

**Page-level support is declared and validated.** Every page in `content/docs`
and `content/integrations` carries a `languages` frontmatter field, enforced by
a strict Zod schema in [`source.config.ts`](source.config.ts) — 98 declare
`[python, typescript]`, 55 declare `[python]` and 1 declares `[typescript]`.
Guides, tutorials, changelog and blog may omit the field and simply follow the
reader's preference; 6 guides declare `[python]` because the features they cover
are Python-only.

**The sidebar follows the selection.** `languages` rides along on each page-tree
node, and [`lib/lang/page-tree.ts`](lib/lang/page-tree.ts) prunes the tree to the
reader's language before Fumadocs renders it. Folders and separator groups
collapse once they empty out, so selecting TypeScript today drops the whole
"Synthetic Data Generation" and "Prompt Optimization" sections, plus the
vector-database half of Integrations, rather than leaving headings over dead
links. The same pruned tree drives the prev/next footer. Readers still reach the
501 page through direct links, bookmarks and search hits.

**The selector only appears on docs and integrations.** Those are the reference
surfaces where the choice is meaningful. Guides and tutorials still honour a
selection made elsewhere — their 187 `<Switch>` blocks render TypeScript for
a reader who picked it — they just don't offer their own control.

**Code blocks are essentially done.** 493 `<Switch>` blocks, 461 of which
carry a TypeScript fence; the other 32 are `bash` and `yaml` pairs. None is
one-sided, and no Python-only page leaks a TypeScript fence.

**Prose has barely started.** `<Term>` covers 98 spans across 4 pages, while
2,265 snake_case identifiers across 170 files (446 distinct) are still
hardcoded Python in prose that TypeScript readers see. The mechanism is now
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
typechecked, so nothing else covers that — and on a `<Term>` nested inside a
`<Case>` or `<Only>`, where the wrapper has already fixed the language and a
one-case `<Switch>` would otherwise render TypeScript text inside a block
labelled Python.

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
| Frameworks | `langchain` (LangGraph included, via `integrations/langchain/langgraph-utils.ts`), `openai-agents`, `openai` | 3 of 12 pages already bilingual, `langgraph` eligible, the other 8 permanently Python-only |
| Models | every provider except LiteLLM — `LocalModel` covers both LM Studio and vLLM, `KimiModel` is Moonshot, `GeminiModel` takes `useVertexAI` | **done** — 14 of 16 pages bilingual, `ai-sdk` is TypeScript-only, `litellm` permanently Python-only |
| Vector databases | none | all 6 permanently Python-only |

`models/ai-sdk` is the site's first `languages: [typescript]` page. The rule it
replaces — "no integration page should be `[typescript]`" — held only while every
documented integration existed in Python; `AISDKModel` has no Python counterpart.

### What blocks wider TypeScript coverage

**Existing pages missing TypeScript examples.** The SDK already supports these,
so the work is writing the `<Switch>` blocks and then widening `languages`.

- `frameworks/langgraph`, covered by `integrations/langchain/langgraph-utils.ts`.

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

1. **Prose identifiers.** The 2,265 spans above. The design question is
   answered (see [Inline prose terms](#inline-prose-terms)); this is now a
   page-by-page conversion job with no blocker in front of it.
2. **Integrations, per [Integration coverage](#integration-coverage).** The
   models half is now bilingual, so a TypeScript reader keeps Integrations'
   largest section; vector databases still vanish entirely and 9 of 12 framework
   pages remain Python-only.
3. **Partial gaps.** 239 unpaired ` ```python ` fences sit on 52 otherwise
   bilingual pages, mostly framework integration tabs, the pytest and
   `deepeval test run` workflows, Pydantic schemas, and `save_as()`. Each
   should become a one-case `<Switch>`, which labels the gap without
   anyone writing TypeScript.
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
- The selector is scoped to docs and integrations. Guides and tutorials inherit
  the choice without offering their own, so the 187 `<Switch>` blocks there
  are finally reachable — they previously had a TypeScript variant no reader
  could ever display — without implying those sections are language-complete.
- Clamping became local to the page subtree, so the root `pythonOnlyPage` state
  and its mount/unmount flash are gone.

Separately, `<LangSwitch>` became
[`<Switch>`](components/lang/switch.tsx) with explicit `<Case id="…">`
children, rewritten across all 493 call sites by
[`scripts/langswitch-to-switch-case.mjs`](scripts/langswitch-to-switch-case.mjs).
The old component picked a child by index, which forced Python first and
TypeScript second with nothing validating the order, limited each side to a
single element, and could not distinguish the 32 same-language pairs (`bash`,
`yaml`). Naming the language removes all three constraints, and a one-sided
block is now the explicit way to mark a gap.

Unchanged: the partial-gap backlog has not moved.
