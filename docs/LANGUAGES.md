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
a strict Zod schema in [`source.config.ts`](source.config.ts) — 82 declare
`[python, typescript]` and 71 declare `[python]`. Guides, tutorials, changelog
and blog may omit the field and simply follow the reader's preference; 6 guides
declare `[python]` because the features they cover are Python-only.

**The sidebar follows the selection.** `languages` rides along on each page-tree
node, and [`lib/lang/page-tree.ts`](lib/lang/page-tree.ts) prunes the tree to the
reader's language before Fumadocs renders it. Folders and separator groups
collapse once they empty out, so selecting TypeScript today drops the whole
"Synthetic Data Generation" and "Prompt Optimization" sections, plus the models
and vector-database halves of Integrations, rather than leaving headings over
dead links. The same pruned tree drives the prev/next footer. Readers still
reach the 501 page through direct links, bookmarks and search hits.

**The selector only appears on docs and integrations.** Those are the reference
surfaces where the choice is meaningful. Guides and tutorials still honour a
selection made elsewhere — their 187 `<Switch>` blocks render TypeScript for
a reader who picked it — they just don't offer their own control.

**Code blocks are essentially done.** 493 `<Switch>` blocks, 461 of which
carry a TypeScript fence; the other 32 are `bash` and `yaml` pairs. None is
one-sided, and no Python-only page leaks a TypeScript fence.

**Prose has barely started.** `<C>` is used on exactly one page, against a
registry of 15 terms, while 2,334 snake_case identifiers across 171 files
(446 distinct) are still hardcoded Python in prose that TypeScript readers see.

## Integration coverage

Audited against `typescript/src` at `0.1.32`. This records what the SDKs
support; frontmatter tracks what each page actually *contains*, so a page stays
`[python]` until its TypeScript examples exist.

| Group | TypeScript SDK surface | Docs verdict |
| --- | --- | --- |
| Frameworks | `langchain` (LangGraph included, via `integrations/langchain/langgraph-utils.ts`), `openai-agents`, `openai` | 3 of 12 pages already bilingual, `langgraph` eligible, the other 8 permanently Python-only |
| Models | every provider except LiteLLM — `LocalModel` covers both LM Studio and vLLM, `KimiModel` is Moonshot, `GeminiModel` takes `useVertexAI` | 14 of 15 pages eligible, all still `[python]` because none carries a TypeScript example yet |
| Vector databases | none | all 6 permanently Python-only |

**No integration page should be `languages: [typescript]` today.** Every
documented integration exists in Python; the TypeScript-only ones have no page
at all.

### What blocks wider TypeScript coverage

**Existing pages missing TypeScript examples.** The SDK already supports these,
so the work is writing the `<Switch>` blocks and then widening `languages`.

- 14 of the 15 `models/*` pages — everything but `litellm`. Two code blocks
  each. Watch the three non-obvious mappings: `lmstudio` and `vllm` are both
  `LocalModel`, `moonshot` is `KimiModel`, and `vertex-ai` is `GeminiModel` with
  `useVertexAI`.
- `frameworks/langgraph`, covered by `integrations/langchain/langgraph-utils.ts`.

**Integrations with no page at all.** These need writing from scratch; no amount
of metadata helps.

- **Mastra** — `integrations/mastra`, exporting `DeepEvalExporter`.
  TypeScript-only, so it would be the site's first `languages: [typescript]`
  page.
- **Vercel AI SDK** — appears twice, as a framework (`integrations/ai-sdk`) and
  as a model provider (`AISDKModel`). TypeScript-only.
- **OpenInference** — `integrations/openinference`, present in both SDKs.

**Settled as permanently Python-only.** No TypeScript equivalent exists, so
these keep `languages: [python]` and need no revisiting: `models/litellm`, all
six `vector-databases/*`, and the framework pages `agentcore`, `anthropic`,
`crewai`, `google-adk`, `huggingface`, `llamaindex`, `pydanticai` and `strands`.

## Outstanding

Roughly in order of how much they hurt a TypeScript reader.

1. **Prose identifiers.** The 2,334 spans above. Requires a decision first:
   whether [`lib/lang/terms.ts`](lib/lang/terms.ts) stays a fully manual
   registry or gains a default camelCase derivation with the registry reserved
   for genuine exceptions. Roughly 446 entries and 2,334 call sites either way,
   so the answer matters.
2. **Integrations, per [Integration coverage](#integration-coverage).** A
   TypeScript reader sees Integrations reduced to three framework pages: models
   and vector databases vanish entirely. The 14 model pages are the cheapest
   parity win on the site at two code blocks each, and
   [`typescript/src/models/README.md`](../typescript/src/models/README.md)
   documents every class, default model and env var — it says outright that it
   is raw material for these docs.
3. **Partial gaps.** 239 unpaired ` ```python ` fences sit on 52 otherwise
   bilingual pages, mostly framework integration tabs, the pytest and
   `deepeval test run` workflows, Pydantic schemas, and `save_as()`. Each
   should become a one-case `<Switch>`, which labels the gap without
   anyone writing TypeScript.
4. **Unverified TypeScript spellings.** The registry's TypeScript values were
   authored by hand and nothing checks them against the shipped SDK
   (`typescript/package.json`, currently `0.1.32`). They are best guesses.
5. **No TypeScript-only pages.** `languages: [typescript]` parses and filters
   correctly but nothing uses it, so the case is untested. Mastra and the Vercel
   AI SDK are the natural first users once written.
6. **No preference persistence.** The selection is `useState` with no storage,
   so it resets to Python on every refresh, new tab, and search-result landing.
   Since most docs traffic arrives directly on an interior page, a TypeScript
   reader currently has to re-select on nearly every visit. Needs localStorage
   plus a pre-hydration script to avoid a flash.
7. **`llms.txt` is Python-only.** `lowerCodeTerms` in [`lib/source.ts`](lib/source.ts)
   hardcodes Python when lowering `<C>` tags. `languages` now makes per-language
   output possible.
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

Unchanged: `<C>` behaves exactly as before, and neither prose coverage nor the
partial-gap backlog moved.
