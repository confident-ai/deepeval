import { source } from '@/lib/source';
import { llms } from 'fumadocs-core/source';

// Markdown rendition of the homepage for agents that send
// `Accept: text/markdown` — `proxy.ts` rewrites `/` here. Reuses the
// llms.txt index (site overview + per-page links) as the body.
export const revalidate = false;

export function GET() {
  return new Response(llms(source).index(), {
    headers: { 'Content-Type': 'text/markdown' },
  });
}
