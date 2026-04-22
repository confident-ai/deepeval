import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import lastModified from 'fumadocs-mdx/plugins/last-modified';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import { z } from 'zod';
import remarkMath from 'remark-math';
import remarkDirective from 'remark-directive';
import rehypeKatex from 'rehype-katex';
import { remarkAdmonitions } from './lib/remark-admonitions';
import { AUTHOR_IDS } from './lib/authors';
import { BLOG_CATEGORY_IDS } from './lib/blog-categories';

/**
 * Extend Fumadocs' default page frontmatter with a Docusaurus-style
 * `sidebar_label`. When set, the page's sidebar label is overridden
 * (see the tree transformer in `lib/source.ts`); the page's H1 still
 * uses the regular `title` field.
 *
 * Note: fumadocs-mdx only allows collection/config exports from this
 * file, so this schema stays internal (non-exported).
 */
const extendedPageSchema = pageSchema.extend({
  sidebar_label: z.string().optional(),
});

const commonOptions = {
  docs: {
    schema: extendedPageSchema,
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
} as const;

/**
 * Blog-specific frontmatter. Kept separate from `commonOptions` so
 * docs/guides/etc don't silently accept `authors`/`date` fields they
 * would ignore. `z.enum(AUTHOR_IDS)` locks `authors` to known IDs
 * from `lib/authors.ts` — a typo fails the build with a clear path.
 */
const blogPageSchema = extendedPageSchema.extend({
  authors: z.array(z.enum(AUTHOR_IDS)).min(1).optional(),
  date: z.coerce.date().optional(),
  image: z.string().url().optional(),
  // Optional — pins a post to one of the known categories in
  // `lib/blog-categories.ts`. Kept as a single value (not an array)
  // because the sidebar groupings are also single-section.
  category: z.enum(BLOG_CATEGORY_IDS).optional(),
});

const blogOptions = {
  docs: {
    schema: blogPageSchema,
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
} as const;

export const docs = defineDocs({ dir: 'content/docs', ...commonOptions });
export const guides = defineDocs({ dir: 'content/guides', ...commonOptions });
export const tutorials = defineDocs({
  dir: 'content/tutorials',
  ...commonOptions,
});
export const integrations = defineDocs({
  dir: 'content/integrations',
  ...commonOptions,
});
export const changelog = defineDocs({
  dir: 'content/changelog',
  ...commonOptions,
});
export const blog = defineDocs({ dir: 'content/blog', ...blogOptions });

export default defineConfig({
  // `lastModified` reads each file's latest git commit timestamp at build
  // time and injects it as `page.data.lastModified`. Outside a git tree
  // (e.g. fresh checkouts before first commit) the value is `null`, which
  // `<PageLastUpdate>` silently no-ops on — safe to always enable.
  plugins: [lastModified()],
  mdxOptions: {
    // remarkDirective parses `:::type[title]` container directives;
    // remarkAdmonitions rewrites the recognized ones into <Callout>.
    remarkPlugins: [remarkMath, remarkDirective, remarkAdmonitions],
    // rehypeKatex must run before the syntax highlighter
    rehypePlugins: (v) => [rehypeKatex, ...v],
  },
});
