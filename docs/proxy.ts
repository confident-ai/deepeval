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

/**
 * Mark a negotiated response as varying on `Accept` (RFC 9110 §12.5.5):
 * these URLs return HTML or markdown for the same path, so a shared cache
 * that ignores `Accept` could serve one representation to a client that
 * asked for the other.
 *
 * Only the markdown branch can carry this. Next 16 sets its own `Vary` on
 * rendered HTML routes and overrides both a middleware append and a
 * `next.config.mjs` header, verified against `next start`. What limits the
 * fallout is that negotiation happens as a rewrite to a distinct path, so
 * any cache keyed on the resolved path stores the two representations
 * separately; a cache keyed on the request URL alone is the residual gap.
 */
function varyOnAccept(response: NextResponse) {
  response.headers.append('Vary', 'Accept');
  return response;
}

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
      return varyOnAccept(
        NextResponse.rewrite(new URL('/home.md', request.nextUrl)),
      );
    }

    for (const { negotiated } of rewrites) {
      const result = negotiated(pathname);
      if (result) {
        return varyOnAccept(
          NextResponse.rewrite(new URL(result, request.nextUrl)),
        );
      }
    }
  }

  return NextResponse.next();
}
