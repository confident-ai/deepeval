import {
  docsSource,
  guidesSource,
  tutorialsSource,
  integrationsSource,
  changelogSource,
  blogSource,
  getPageMarkdownUrl,
  getPageImage,
} from '@/lib/source';
import { createSection } from '@/lib/section';
import BlogPostMeta from '@/src/components/BlogPostMeta';
import SchemaInjector from '@/src/components/SchemaInjector/SchemaInjector';
import {
  buildArticleSchema,
  buildBlogHomeSchema,
} from '@/src/utils/schema-helpers';
import { getAuthor, type AuthorId } from '@/lib/authors';
import type { BlogCategoryId } from '@/lib/blog-categories';

type BlogFrontmatter = {
  title: string;
  description?: string;
  authors?: AuthorId[];
  date?: Date | string;
  category?: BlogCategoryId;
  lastModified?: number | string | Date | null;
  // Optional per-post cover image (absolute URL). When present it
  // overrides the site-wide `og:image` fallback set in `app/layout.tsx`
  // so social previews show the post's hero art instead of the generic
  // social card. Validated in `blogPageSchema` (source.config.ts).
  image?: string;
};

/**
 * Pull the publish / modified dates off a blog page as ISO strings.
 * `date` is author-supplied frontmatter; `lastModified` is injected by
 * the `fumadocs-mdx/plugins/last-modified` plugin (git-derived).
 */
function toIso(value: unknown): string | undefined {
  if (!value) return undefined;
  if (value instanceof Date) return value.toISOString();
  const parsed = new Date(value as string);
  return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

export const docsSection = createSection({
  source: docsSource,
  contentDir: 'content/docs',
  getMarkdownUrl: (page) => getPageMarkdownUrl(page, docsSource).url,
  getImageUrl: (page) => getPageImage(page).url,
  showContributors: true,
});

export const guidesSection = createSection({
  source: guidesSource,
  contentDir: 'content/guides',
  getMarkdownUrl: (page) => getPageMarkdownUrl(page, guidesSource).url,
  showContributors: true,
});

export const tutorialsSection = createSection({
  source: tutorialsSource,
  contentDir: 'content/tutorials',
  getMarkdownUrl: (page) => getPageMarkdownUrl(page, tutorialsSource).url,
  showContributors: true,
});

export const integrationsSection = createSection({
  source: integrationsSource,
  contentDir: 'content/integrations',
  getMarkdownUrl: (page) => getPageMarkdownUrl(page, integrationsSource).url,
  showContributors: true,
});

export const changelogSection = createSection({
  source: changelogSource,
  contentDir: 'content/changelog',
  getMarkdownUrl: (page) => getPageMarkdownUrl(page, changelogSource).url,
});

export const blogSection = createSection({
  source: blogSource,
  contentDir: 'content/blog',
  getMarkdownUrl: (page) => getPageMarkdownUrl(page, blogSource).url,
  renderBeforeBody: (page) => {
    const data = page.data as BlogFrontmatter;
    const { authors, category, title, description, date } = data;

    // Blog index (`/blog`) — no authors/date; emit a `Blog` JSON-LD
    // listing all posts instead so Google can surface the post set
    // directly. Matches what the old Docusaurus blog plugin emitted.
    if (!authors) {
      const posts = blogSource
        .getPages()
        .filter((p) => {
          const d = p.data as BlogFrontmatter;
          return Array.isArray(d.authors) && d.authors.length > 0;
        })
        .map((p) => {
          const d = p.data as BlogFrontmatter;
          return {
            title: d.title,
            description: d.description ?? '',
            slug: p.slugs[p.slugs.length - 1] ?? '',
            authors: (d.authors ?? []).map((id) => getAuthor(id).name),
            date: toIso(d.date) ?? '',
          };
        });
      return <SchemaInjector schema={buildBlogHomeSchema(posts)} />;
    }

    // Per-post byline (unchanged) + Article / TechArticle JSON-LD.
    // `date` is still required in frontmatter for the git-less publish
    // sort / OG metadata, but we don't display it in the byline row.
    const authorNames = authors.map((id) => getAuthor(id).name);
    const articleSchema = buildArticleSchema({
      title,
      description,
      url: page.url,
      datePublished: toIso(date),
      dateModified: toIso(data.lastModified ?? undefined),
      authors: authorNames,
    });

    return (
      <>
        <SchemaInjector schema={articleSchema} />
        <BlogPostMeta authors={authors} category={category} />
      </>
    );
  },
  // Individual posts get `openGraph.type = 'article'` + publish /
  // modified timestamps + author list, so social previews render as
  // proper article cards instead of a generic website card. If the
  // post sets `image:` in frontmatter we also promote it to
  // `openGraph.images` / `twitter.images` so the share card shows the
  // post's hero art instead of the generic site-wide social_card.png.
  extendMetadata: (page) => {
    const data = page.data as BlogFrontmatter;
    if (!data.authors) return {};

    const publishedTime = toIso(data.date);
    const modifiedTime = toIso(data.lastModified ?? undefined);
    const authorNames = data.authors.map((id) => getAuthor(id).name);
    const image = data.image;

    return {
      openGraph: {
        type: 'article',
        ...(publishedTime ? { publishedTime } : {}),
        ...(modifiedTime ? { modifiedTime } : {}),
        authors: authorNames,
        // Per-post hero art overrides the site-wide `/img/social_card.png`
        // default set in `app/layout.tsx`. We intentionally DO NOT also
        // override `twitter.images` here: Next.js replaces (doesn't
        // deep-merge) the `twitter` object across nested `generateMetadata`
        // calls, so setting it would also wipe the layout's `card`,
        // `site`, and `creator`. X/Twitter's card renderer falls back
        // to `og:image` when `twitter:image` is absent, and other
        // `summary_large_image` consumers (LinkedIn, Slack, Discord)
        // read `og:image` directly — so the single override covers
        // every surface.
        ...(image ? { images: image } : {}),
      },
    };
  },
});
