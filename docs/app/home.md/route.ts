import {
  blogSource,
  changelogSource,
  docsSource,
  guidesSource,
  integrationsSource,
  tutorialsSource,
} from '@/lib/source';
import { llms } from 'fumadocs-core/source';

// Markdown rendition of the homepage for agents that send
// `Accept: text/markdown` — `proxy.ts` rewrites `/` here.
//
// Built from all six sections rather than `llms(source).index()`, which
// covers `/docs` alone: `proxy.ts` negotiates markdown for every section,
// so an index that listed only docs would tell an agent the other five do
// not exist. Each source contributes its own `# <section>` heading.
export const revalidate = false;

// Each collection produces its own `LoaderOutput` generic, so the array is
// typed loosely for the same reason `lib/llms-route.ts` is: the runtime
// surface `llms()` needs is identical across all six.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sources: any[] = [
  docsSource,
  guidesSource,
  tutorialsSource,
  integrationsSource,
  changelogSource,
  blogSource,
];

export function GET() {
  const body = sources.map((source) => llms(source).index()).join('\n\n');

  return new Response(body, {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  });
}
