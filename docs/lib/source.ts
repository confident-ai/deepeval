import {
  docs,
  guides,
  tutorials,
  integrations,
  changelog,
  blog,
} from 'collections/server';
import { loader, type PageTreeTransformer } from 'fumadocs-core/source';
import { lucideIconsPlugin } from 'fumadocs-core/source/lucide-icons';
import { contentRouteFor, docsImageRoute } from './shared';

/**
 * Docusaurus-style `sidebar_label` → override the sidebar node's name
 * while leaving the page's H1 (driven by `title`) alone.
 *
 * The schema for this field is defined in `source.config.ts`. Pages
 * without a `sidebar_label` fall through and keep their default name
 * (their `title`), so this is purely additive.
 *
 * Typed as `PageTreeTransformer<any>` because the transformer is
 * collection-agnostic — each per-section `loader()` has its own
 * strongly-typed storage generic that wouldn't unify otherwise.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sidebarLabelTransformer: PageTreeTransformer<any> = {
  file(node) {
    const ref = node.$ref;
    if (!ref) return node;
    const file = this.storage.read(ref);
    if (!file || file.format !== 'page') return node;
    const label = (file.data as { sidebar_label?: unknown }).sidebar_label;
    if (typeof label === 'string' && label.length > 0) {
      node.name = label;
    }
    return node;
  },
};

const pageTree = { transformers: [sidebarLabelTransformer] };

export const docsSource = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
  pageTree,
});

export const guidesSource = loader({
  baseUrl: '/guides',
  source: guides.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
  pageTree,
});

export const tutorialsSource = loader({
  baseUrl: '/tutorials',
  source: tutorials.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
  pageTree,
});

export const integrationsSource = loader({
  baseUrl: '/integrations',
  source: integrations.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
  pageTree,
});

export const changelogSource = loader({
  baseUrl: '/changelog',
  source: changelog.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
  pageTree,
});

export const blogSource = loader({
  baseUrl: '/blog',
  source: blog.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
  pageTree,
});

// Backwards-compatible alias so scaffold-generated routes that still import
// `source` (llms.txt, llms-full.txt, og image routes, search route) keep
// targeting the primary /docs section.
export const source = docsSource;

export function getPageImage(page: (typeof source)['$inferPage']) {
  const segments = [...page.slugs, 'image.png'];

  return {
    segments,
    url: `${docsImageRoute}/${segments.join('/')}`,
  };
}

/**
 * Build the raw-markdown URL for a page in *any* section. The section
 * prefix is inferred from the page's `url` (e.g. a page at `/guides/foo`
 * lives under the `guides` section), so the same helper works for docs,
 * guides, tutorials, integrations, and changelog as long as each has a
 * matching `/llms.mdx/<section>` route handler.
 *
 * The second arg is kept for backwards-compat with older callers that
 * pass a source; it's ignored in favor of `page.url` which is always
 * the canonical source of truth for the section prefix.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function getPageMarkdownUrl(page: any, _src?: unknown) {
  const segments = [...page.slugs, 'content.md'];

  return {
    segments,
    url: `${contentRouteFor(page.url)}/${segments.join('/')}`,
  };
}

export async function getLLMText(page: (typeof source)['$inferPage']) {
  // `getText` is injected by fumadocs-mdx when `postprocess.includeProcessedMarkdown`
  // is set (see source.config.ts) but isn't part of the static PageData type,
  // so we reach for it through an explicit cast.
  const data = page.data as typeof page.data & {
    getText: (format: 'raw' | 'processed') => Promise<string>;
  };
  const processed = await data.getText('processed');

  return `# ${page.data.title} (${page.url})

${processed}`;
}

/**
 * Extract a meta-description-sized blurb for a page, preferring explicit
 * `description:` frontmatter and falling back to the first real paragraph
 * of the MDX body. Matches the old Docusaurus behavior of auto-filling
 * `<meta name="description">` from the first paragraph, which we lost when
 * switching to Fumadocs (it leaves `page.data.description` undefined and
 * does not synthesize one).
 *
 * The fallback path strips common MDX noise (front-of-file `import` lines,
 * JSX tags, admonition fences, headings, blockquote markers, list bullets,
 * link/emphasis syntax) so crawlers see prose, then truncates at a word
 * boundary to ~160 chars — the sweet spot Google still tends to render in
 * SERPs without cutting mid-word.
 */
const DESCRIPTION_MAX = 160;

function cleanMarkdownForDescription(md: string): string {
  let text = md;

  // Drop import / export lines (MDX directives at top of file).
  text = text.replace(/^\s*(?:import|export)\b[^\n]*\n/gm, '');

  // Drop admonition fences `:::tip[title]` / `:::` on their own lines.
  text = text.replace(/^:::[^\n]*$/gm, '');

  // Drop HTML/MDX comments.
  text = text.replace(/<!--[\s\S]*?-->/g, '');

  // Drop fenced code blocks entirely — they rarely make useful descriptions.
  text = text.replace(/```[\s\S]*?```/g, '');

  // Drop self-closing JSX tags like <ImageDisplayer ... /> and paired
  // tags like <VideoDisplayer>...</VideoDisplayer>. Keep inner text for
  // paired tags so `<Envelope>…</Envelope>` style components don't nuke
  // the surrounding paragraph.
  text = text.replace(/<([A-Z][\w]*)\b[^>]*\/>/g, '');
  text = text.replace(/<\/?[A-Z][\w]*\b[^>]*>/g, '');

  return text;
}

function extractFirstParagraph(md: string): string {
  const cleaned = cleanMarkdownForDescription(md);
  const blocks = cleaned
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter(Boolean);

  for (const block of blocks) {
    // Skip headings, blockquotes, horizontal rules, list-only blocks.
    if (/^#{1,6}\s/.test(block)) continue;
    if (/^>\s/.test(block)) continue;
    if (/^-{3,}$|^\*{3,}$/.test(block)) continue;
    if (/^(?:[-*+]\s|\d+\.\s)/.test(block)) continue;

    // Strip inline markdown syntax and collapse whitespace.
    const prose = block
      .replace(/`([^`]+)`/g, '$1')
      .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/__([^_]+)__/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/_([^_]+)_/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();

    if (prose.length > 0) return prose;
  }

  return '';
}

function truncateOnWord(text: string, max: number): string {
  if (text.length <= max) return text;
  const slice = text.slice(0, max);
  const lastSpace = slice.lastIndexOf(' ');
  const base = lastSpace > max * 0.6 ? slice.slice(0, lastSpace) : slice;
  return `${base.replace(/[\s.,;:!?-]+$/, '')}…`;
}

export async function getPageDescription(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  page: any,
): Promise<string | undefined> {
  const frontmatter = page.data?.description;
  if (typeof frontmatter === 'string' && frontmatter.length > 0) {
    return frontmatter;
  }

  const data = page.data as {
    getText?: (format: 'raw' | 'processed') => Promise<string>;
  };
  if (typeof data.getText !== 'function') return undefined;

  try {
    const processed = await data.getText('processed');
    const para = extractFirstParagraph(processed);
    if (!para) return undefined;
    return truncateOnWord(para, DESCRIPTION_MAX);
  } catch {
    return undefined;
  }
}
