// RFC 9727 API catalog, reached through the /.well-known/api-catalog rewrite
// in next.config.mjs (the app router ignores dot-prefixed folders, so the
// canonical path can't be a route directory). The linkset format is RFC 9264.
// DeepEval itself ships no HTTP API — the entries point at the Confident AI
// platform APIs, whose OpenAPI spec and reference live on confident-ai.com
// (published by confident-landing PR #130).
export const revalidate = false;

const confidentSite = 'https://www.confident-ai.com';

const apiEntry = (anchor: string) => ({
  anchor,
  'service-desc': [
    {
      href: `${confidentSite}/docs/openapi.yaml`,
      type: 'application/vnd.oai.openapi',
    },
  ],
  'service-doc': [
    { href: `${confidentSite}/docs/api-reference`, type: 'text/html' },
  ],
  status: [{ href: 'https://status.confident-ai.com/' }],
});

const catalog = {
  linkset: [
    apiEntry('https://api.confident-ai.com/'),
    apiEntry('https://eu.api.confident-ai.com/'),
    {
      anchor: 'https://mcp.confident-ai.com/mcp',
      'service-doc': [
        {
          href: `${confidentSite}/docs/coding-agents/mcp`,
          type: 'text/html',
        },
      ],
      status: [{ href: 'https://status.confident-ai.com/' }],
    },
  ],
};

export function GET() {
  return new Response(JSON.stringify(catalog, null, 2), {
    headers: { 'Content-Type': 'application/linkset+json' },
  });
}
