import type { MetadataRoute } from 'next';
import {
  docsSource,
  guidesSource,
  tutorialsSource,
  integrationsSource,
  changelogSource,
  blogSource,
} from '@/lib/source';
import { siteUrl } from '@/lib/shared';
import { getBenchmarkSlugs } from '@/lib/benchmarks/repository';

// Matches the old Docusaurus-generated `sitemap.xml` (enabled by default
// via the classic preset). We enumerate every page from each Fumadocs
// source and emit absolute URLs anchored at `siteUrl`. `lastModified`
// is populated by `fumadocs-mdx/plugins/last-modified` at build time
// (see `source.config.ts`); pages without a git history (e.g. fresh
// checkouts) omit it rather than faking a date.

// Typed loosely because each section's loader has its own storage
// generic that wouldn't unify — we only touch `getPages()` and
// `page.url` / `page.data.lastModified`, which are identical across
// every section.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnySource = any;

const sources: AnySource[] = [
  docsSource,
  guidesSource,
  tutorialsSource,
  integrationsSource,
  changelogSource,
  blogSource,
];

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const entries: MetadataRoute.Sitemap = sources.flatMap((source) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    source.getPages().map((page: any) => {
      const lastModified = page.data?.lastModified;
      return {
        url: `${siteUrl}${page.url}`,
        ...(lastModified ? { lastModified: new Date(lastModified) } : {}),
      };
    }),
  );

  const benchmarkSlugs = await getBenchmarkSlugs();

  return [
    { url: `${siteUrl}/`, lastModified: new Date() },
    { url: `${siteUrl}/enterprise` },
    { url: `${siteUrl}/benchmarks` },
    ...benchmarkSlugs.map((slug) => ({
      url: `${siteUrl}/benchmarks/${slug}`,
    })),
    ...entries,
  ];
}
