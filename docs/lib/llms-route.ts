import { notFound } from 'next/navigation';
import { getLLMText, getPageMarkdownUrl } from '@/lib/source';

// Each fumadocs collection produces its own `LoaderOutput` generic,
// so we intentionally accept any source here — the runtime surface
// (`getPage`, `getPages`) is the same across all of them.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Source = any;

/**
 * Factory for the `/llms.mdx/<section>/[[...slug]]/route.ts` handler.
 * Each section re-uses this to serve raw markdown at a predictable URL
 * for the "Copy as Markdown" button.
 */
export function createLLMsRoute(source: Source) {
  async function GET(_req: Request, { params }: { params: Promise<{ slug?: string[] }> }) {
    const { slug } = await params;
    const page = source.getPage(slug?.slice(0, -1));
    if (!page) notFound();

    return new Response(await getLLMText(page), {
      headers: { 'Content-Type': 'text/markdown' },
    });
  }

  function generateStaticParams() {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return source.getPages().map((page: any) => ({
      slug: getPageMarkdownUrl(page, source).segments,
    }));
  }

  return { GET, generateStaticParams, revalidate: false as const };
}
