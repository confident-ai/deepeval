import type { ReactNode } from "react";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Banner } from "fumadocs-ui/components/banner";
import { DocsLayout } from "fumadocs-ui/layouts/notebook";
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
  MarkdownCopyButton,
  ViewOptionsPopover,
} from "fumadocs-ui/layouts/notebook/page";
import { createRelativeLink } from "fumadocs-ui/mdx";
import { baseOptions } from "@/lib/layout.shared";
import { getMDXComponents } from "@/components/mdx";
import { gitConfig } from "@/lib/shared";
import { getPageContributors } from "@/lib/contributors";
import { getPageDescription } from "@/lib/source";
import Footer from "@/src/layouts/Footer";
import NavHeader from "@/src/layouts/NavHeader";
import TocFooter from "@/src/components/TocFooter";
import SidebarSearch from "@/src/layouts/SidebarSearch";
import Link from "next/link";

// Each section's fumadocs-mdx collection resolves to a differently-typed
// `LoaderOutput` (docs vs guides vs integrations all have their own
// schema generics). The cross-section factory here is intentionally
// agnostic to that shape, so the source is typed loosely. Using a
// stricter shared type (`ReturnType<typeof loader>`) doesn't unify
// across collections and would require each caller to cast.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Source = any;

type SectionPageProps = {
  params: Promise<{ slug?: string[] }>;
};

// Pages produced by our fumadocs-mdx collections carry the standard MDX frontmatter
// (title, description) plus body/toc/full injected by fumadocs-mdx. The core loader
// type is generic over this, so cast to a minimal shape we rely on here.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Page = any;

export type SectionConfig = {
  /** Fumadocs loader for this section. */
  source: Source;
  /** Relative path inside the repo where the MDX files live, used to build the "Edit on GitHub" URL. */
  contentDir: string;
  /** Optional helper returning the public raw-markdown URL for a page (enables the copy-markdown / view-options buttons). */
  getMarkdownUrl?: (page: Page) => string;
  /** Optional helper returning an OG image URL for a page. */
  getImageUrl?: (page: Page) => string;
  /**
   * Optional custom content rendered between the page description/copy-markdown
   * header and the main MDX body. Used by the blog section to surface author
   * avatars + date; other sections leave this undefined and get the default
   * layout.
   */
  renderBeforeBody?: (page: Page) => ReactNode;
  /**
   * Show the build-time git-derived contributor strip below the
   * "last updated" line. Opt-in per section — docs has it, blog
   * already surfaces authors in the byline so it skips this.
   */
  showContributors?: boolean;
  /**
   * Optional per-section metadata extension. Return value is shallow-merged
   * over the base metadata produced by `generateMetadata` (title,
   * description, canonical, optional OG image) — with `openGraph` and
   * `alternates` deep-merged so a section that sets
   * `openGraph.type = 'article'` doesn't clobber the per-page OG image.
   *
   * Used by the blog section to set `openGraph.type`, `publishedTime`,
   * `modifiedTime`, and the author list on individual posts.
   */
  extendMetadata?: (page: Page) => Promise<Metadata> | Metadata;
};

/**
 * Build the layout + page handlers for a docs section.
 *
 * Usage in `app/<section>/layout.tsx`:
 *   export default sectionDocs.Layout;
 *
 * Usage in `app/<section>/[[...slug]]/page.tsx`:
 *   export default sectionDocs.Page;
 *   export const generateStaticParams = sectionDocs.generateStaticParams;
 *   export const generateMetadata = sectionDocs.generateMetadata;
 */
export function createSection(config: SectionConfig) {
  const {
    source,
    contentDir,
    getMarkdownUrl,
    getImageUrl,
    renderBeforeBody,
    showContributors,
    extendMetadata,
  } = config;

  function Layout({ children }: { children: ReactNode }) {
    const { nav, ...rest } = baseOptions();
    return (
      <>
        <Banner id="docs-announcement" height="30px">
          DeepEval just got a new look 🎉 Read the{" "}
          <Link href="/blog/deepeval-got-a-new-look">announcement</Link> to learn more.
        </Banner>
        <DocsLayout
          {...rest}
          nav={{ ...nav, mode: "top" }}
          tabMode="navbar"
          tree={source.getPageTree()}
          // Swizzled header: three-column grid aligned with the body
          // grid (sidebar / main / toc). See NavHeader for the layout;
          // `slots.header` is the documented override point.
          slots={{ header: NavHeader }}
          // Search lives at the top of the sidebar instead of inside
          // the top nav. Fumadocs' Sidebar appends the `banner` node
          // inside its own `p-4 pb-2` wrapper, so we get consistent
          // spacing above the first page-tree section (e.g. "Getting
          // Started") and in the mobile drawer. The header still
          // keeps the compact (magnifying-glass) search trigger for
          // mobile reachability — see NavHeader col 3.
          sidebar={{ banner: <SidebarSearch key="sidebar-search" /> }}
        >
          {children}
        </DocsLayout>
        <Footer />
      </>
    );
  }

  async function Page(props: SectionPageProps) {
    const params = await props.params;
    const rawPage = source.getPage(params.slug);
    if (!rawPage) notFound();
    const page = rawPage as Page;

    const MDX = page.data.body;
    const markdownUrl = getMarkdownUrl?.(page);

    // Meta strip rendered underneath the TOC (and mirrored into the
    // mobile TOC popover) — "Last updated" + contributor avatars. Kept
    // together so they share one small attribution column next to the
    // prose instead of pushing the `next/prev` nav further down the
    // page. Passed to both `tableOfContent.footer` and
    // `tableOfContentPopover.footer` so the mobile/condensed TOC (which
    // Fumadocs renders as a popover, not the sidebar) gets parity.
    const contributors = showContributors
      ? getPageContributors(contentDir, page.path)
      : [];
    const tocFooter = (
      <TocFooter
        contributors={contributors}
        lastModified={page.data.lastModified}
      />
    );

    return (
      <DocsPage
        toc={page.data.toc}
        full={page.data.full}
        tableOfContent={{ style: "normal", footer: tocFooter }}
        tableOfContentPopover={{ footer: tocFooter }}
      >
        <DocsTitle>{page.data.title}</DocsTitle>
        <DocsDescription className="mb-0 text-[15px] font-light">
          {page.data.description}
        </DocsDescription>
        {markdownUrl ? (
          // `MarkdownCopyButton` / `ViewOptionsPopover` default to fumadocs'
          // `size="sm"` variant (the smallest they expose). The className
          // overrides here trim padding + icon size one notch smaller so
          // the header feels less button-heavy. `cn()` inside fumadocs
          // merges our classes after the defaults, so tailwind-merge wins
          // for padding/gap. Icons need `!` because `ViewOptionsPopover`
          // hardcodes `size-3.5` directly on its chevron child — a plain
          // parent selector loses that specificity fight, so we force it.
          <div className="flex flex-row gap-2 items-center mb-4">
            <MarkdownCopyButton
              markdownUrl={markdownUrl}
              className="px-1.5 py-1 gap-1.5 [&_svg]:!size-3"
            />
            <ViewOptionsPopover
              markdownUrl={markdownUrl}
              githubUrl={`https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/${contentDir}/${page.path}`}
              className="px-1.5 py-1 gap-1.5 [&_svg]:!size-3"
            />
          </div>
        ) : null}
        {renderBeforeBody?.(page)}
        <DocsBody>
          <MDX
            components={getMDXComponents({
              a: createRelativeLink(source, page),
            })}
          />
        </DocsBody>
      </DocsPage>
    );
  }

  async function generateStaticParams() {
    return source.generateParams();
  }

  async function generateMetadata(props: SectionPageProps): Promise<Metadata> {
    const params = await props.params;
    const page = source.getPage(params.slug);
    if (!page) notFound();

    const imageUrl = getImageUrl?.(page);
    // Prefer frontmatter `description:`; otherwise derive from the first
    // real paragraph of the MDX body (matches the old Docusaurus
    // auto-description behavior we lost in the migration).
    const description = await getPageDescription(page);

    // Per-section override (e.g. blog sets `openGraph.type = 'article'`).
    // Shallow-merge `extra` at top-level, but deep-merge `openGraph` and
    // `alternates` so a section adding article fields doesn't clobber
    // the per-page OG image or the canonical we computed above.
    const extra = (await extendMetadata?.(page)) ?? {};
    const {
      openGraph: extraOg,
      alternates: extraAlternates,
      ...extraTop
    } = extra;

    const baseOg: NonNullable<Metadata["openGraph"]> = imageUrl
      ? { images: imageUrl }
      : {};
    const mergedOg = { ...baseOg, ...(extraOg ?? {}) };

    return {
      title: page.data.title,
      ...(description ? { description } : {}),
      ...extraTop,
      // Relative URL — resolved against the root `metadataBase` in
      // `app/layout.tsx`. `page.url` is the public path like
      // `/docs/metrics-faithfulness`.
      alternates: { canonical: page.url, ...(extraAlternates ?? {}) },
      ...(Object.keys(mergedOg).length > 0
        ? { openGraph: mergedOg as Metadata["openGraph"] }
        : {}),
    };
  }

  return { Layout, Page, generateStaticParams, generateMetadata };
}
