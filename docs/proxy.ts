import { NextRequest, NextResponse } from 'next/server';
import { isMarkdownPreferred, rewritePath } from 'fumadocs-core/negotiation';
import { contentRouteFor } from '@/lib/shared';

/**
 * Every MDX-backed section gets two rewrites into its raw-markdown
 * route (`/llms.mdx/<section>/.../content.md`): an explicit `.mdx`
 * suffix that always works, and an `Accept: text/markdown`
 * content-negotiation rewrite so agents can request any page as
 * markdown while browsers keep getting HTML.
 */
const sections = [
  'docs',
  'guides',
  'tutorials',
  'integrations',
  'changelog',
  'blog',
] as const;

const rewrites = sections.map((section) => {
  const contentRoute = contentRouteFor(section);
  return {
    suffix: rewritePath(
      `/${section}{/*path}.mdx`,
      `${contentRoute}{/*path}/content.md`,
    ).rewrite,
    negotiated: rewritePath(
      `/${section}{/*path}`,
      `${contentRoute}{/*path}/content.md`,
    ).rewrite,
  };
});

export default function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  for (const { suffix } of rewrites) {
    const result = suffix(pathname);
    if (result) {
      return NextResponse.rewrite(new URL(result, request.nextUrl));
    }
  }

  if (isMarkdownPreferred(request)) {
    if (pathname === '/') {
      return NextResponse.rewrite(new URL('/home.md', request.nextUrl));
    }

    for (const { negotiated } of rewrites) {
      const result = negotiated(pathname);
      if (result) {
        return NextResponse.rewrite(new URL(result, request.nextUrl));
      }
    }
  }

  return NextResponse.next();
}
