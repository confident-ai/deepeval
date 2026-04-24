export const appName = 'DeepEval';

/**
 * Canonical public origin for the site. Single source of truth for
 * every absolute URL we emit (sitemap, robots, JSON-LD, `metadataBase`,
 * OG/image URLs, etc.) so a domain change only needs one edit.
 */
export const siteUrl = 'https://deepeval.com';

/**
 * Site title used as the default `<title>` on routes that don't set
 * their own, and as the suffix in the root layout's title template
 * (`%s | {siteTitle}`). Kept verbatim from the old Docusaurus
 * `config.title` for SERP continuity.
 */
export const siteTitle =
  'DeepEval by Confident AI - The LLM Evaluation Framework';

/**
 * Short meta-description used on the homepage and as the fallback for
 * pages without a frontmatter `description:` and no extractable body
 * paragraph.
 */
export const siteDescription =
  'DeepEval is the open-source LLM evaluation framework for testing and benchmarking LLM applications.';

export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';

/**
 * Raw-markdown API route prefix for any section. We host a Next.js
 * route handler at `/llms.mdx/<section>/<slug>/content.md` for every
 * section that wants the "Copy as Markdown" button.
 *
 * Pass either a section name (`"docs"`) or a source's `baseUrl`
 * (`"/guides"`) — both work.
 */
export function contentRouteFor(sectionOrBaseUrl: string) {
  const section = sectionOrBaseUrl.replace(/^\/+/, '').split('/')[0];
  return `/llms.mdx/${section}`;
}

/** Back-compat alias. */
export const docsContentRoute = contentRouteFor('docs');

export const gitConfig = {
  user: 'confident-ai',
  repo: 'deepeval',
  branch: 'main',
};

/** Community Discord invite — used by the `<DiscordButton>` CTA and
 *  referenced from the Kapa disclaimer copy. Single source of truth so
 *  rotating the invite is a one-line change. */
export const discordUrl = 'https://discord.gg/a3K9c8GRGt';

/**
 * Kapa.ai Ask-AI config. Values mirror what the old Docusaurus site
 * shipped (`old_deepeval_docs/docusaurus.config.ts`) but re-mapped to
 * the *current* Kapa widget API — several attribute names were
 * renamed in the 2024 refresh (see
 * https://docs.kapa.ai/integrations/website-widget/configuration/behavior
 * and `.../component-styles`). `websiteId` is the public Kapa project
 * identifier; safe to ship in client bundles.
 *
 * The widget is loaded with `data-launcher-button-hidden="true"` in
 * `app/layout.tsx` so Kapa's default floating launcher never renders;
 * every click on an element with class `triggerClass` opens the modal
 * via `data-modal-override-open-class`. `<AskAIButton>` applies that
 * class, so any button rendered through it doubles as a Kapa trigger
 * with no JS handler of our own.
 */
export const kapaConfig = {
  websiteId: 'a3177869-c654-4b86-9c92-e4b4416f66e0',
  projectName: 'DeepEval',
  // Required by Kapa. Used as the modal accent / brand color.
  projectColor: '#ffffff',
  projectLogo:
    'https://pbs.twimg.com/profile_images/1888060560161574912/qbw1-_2g_400x400.png',
  modalTitle: 'Ask DeepEval',
  chatDisclaimer:
    "All the following results are AI generated, if you can't find the solution you're looking for, ping us in [Discord](https://discord.gg/a3K9c8GRGt) we'd be happy to have you!",
  exampleQuestions:
    'Can I create a dataset using my knowledge base?, Can I create a custom metrics for my use-case?',
  uncertainAnswerCallout:
    "It would be better to ask this question directly in DeepEval's [Discord](https://discord.gg/a3K9c8GRGt) channel.",
  /**
   * Any element that carries this class opens the Kapa modal on click.
   * Stored as a bare class name (no leading dot) because Kapa's
   * `data-modal-override-open-class` expects the class name, not a
   * CSS selector.
   */
  triggerClass: 'ask-ai-trigger',
} as const;
