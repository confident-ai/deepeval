// RFC 9727 API catalog; the linkset format is RFC 9264. DeepEval itself ships
// no HTTP API — the entries point at the Confident AI platform APIs, whose
// OpenAPI spec and reference live on confident-ai.com (published by
// confident-landing PR #130).
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

const mcpEntry = (anchor: string) => ({
  anchor,
  'service-doc': [
    {
      href: `${confidentSite}/docs/coding-agents/mcp`,
      type: 'text/html',
    },
  ],
  'service-desc': [
    {
      href: `${confidentSite}/.well-known/mcp/server-card.json`,
      type: 'application/mcp-server-card+json',
    },
  ],
  status: [{ href: 'https://status.confident-ai.com/' }],
});

// Both regions for both services. The MCP server card lists a US and an EU
// remote, so listing only US here would have left half the deployment
// undiscoverable while the REST entries advertised both.
const catalog = {
  linkset: [
    apiEntry('https://api.confident-ai.com/'),
    apiEntry('https://eu.api.confident-ai.com/'),
    mcpEntry('https://mcp.confident-ai.com/mcp'),
    mcpEntry('https://eu.mcp.confident-ai.com/mcp'),
  ],
};

export function GET() {
  return new Response(JSON.stringify(catalog, null, 2), {
    headers: { 'Content-Type': 'application/linkset+json' },
  });
}
