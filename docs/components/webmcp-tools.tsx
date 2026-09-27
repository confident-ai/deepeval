'use client';

import { useEffect } from 'react';

// WebMCP (W3C draft: https://webmachinelearning.github.io/webmcp/): registers
// read-only tools with the browser's model context so in-browser agents can
// search the docs, read any page as markdown, and enumerate the site's agent
// resources. The spec surface is document.modelContext; Chrome's preview
// shipped it on navigator first and keeps that name as a deprecated alias, so
// the effect probes both. Everywhere else this is a no-op. Tool failures are
// reported in-band as text results so the agent sees an actionable message
// instead of a raw exception.

type WebMcpToolResult = { content: Array<{ type: 'text'; text: string }> };

type WebMcpTool = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  execute: (input: Record<string, unknown>) => Promise<WebMcpToolResult>;
};

type ModelContext = {
  registerTool?: (tool: WebMcpTool & { signal?: AbortSignal }) => unknown;
  provideContext?: (context: { tools: WebMcpTool[] }) => void;
};

type SearchResult = {
  id: string;
  url: string;
  type: 'page' | 'heading' | 'text';
  content: string;
};

function text(value: string): WebMcpToolResult {
  return { content: [{ type: 'text', text: value }] };
}

// `/api/search` covers `/docs` only, so an empty result set is not evidence
// that the content is missing. Both the empty and the failed branch point at
// the surfaces that do cover the whole site.
const OTHER_SECTIONS =
  'Guides, tutorials, integrations, changelog, and blog are not in this index: read / with Accept: text/markdown for the full site index.';
const SEARCH_FALLBACK = 'Search is unavailable. Fetch /llms.txt instead.';

const tools: WebMcpTool[] = [
  {
    name: 'search_deepeval_docs',
    // Scoped to /docs on purpose: `/api/search` indexes `docsSource` only.
    // Naming the other sections here would turn "not indexed" into "does not
    // exist" for any agent that searches for a guide or a blog post.
    description:
      'Search the DeepEval documentation (the /docs section). Returns titles and URLs. Request any result URL with Accept: text/markdown (or append .mdx) for markdown. For guides, tutorials, integrations, changelog, and blog, read / with Accept: text/markdown for the full site index.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search terms' },
      },
      required: ['query'],
    },
    execute: async (input) => {
      const query = String(input.query ?? '').trim();
      if (!query) return text('Provide a search query.');
      try {
        const params = new URLSearchParams({ query });
        const res = await fetch(`/api/search?${params}`);
        if (!res.ok) return text(`${SEARCH_FALLBACK} (HTTP ${res.status})`);
        const results = (await res.json()) as SearchResult[];
        const pages = results.filter((r) => r.type === 'page').slice(0, 5);
        const top = pages.length > 0 ? pages : results.slice(0, 5);
        if (top.length === 0) {
          return text(`No results in /docs. ${OTHER_SECTIONS}`);
        }
        return text(
          top
            .map((r) => `${r.content}\n${location.origin}${r.url}`)
            .join('\n\n'),
        );
      } catch (err) {
        console.warn('WebMCP: search failed', err);
        return text(SEARCH_FALLBACK);
      }
    },
  },
  {
    name: 'get_page_markdown',
    description:
      'Fetch a page on this site as markdown. Works for the homepage, docs, guides, tutorials, integrations, changelog, and blog.',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description:
            'Site-relative path, e.g. /docs/introduction or /blog. Defaults to the current page.',
        },
      },
    },
    execute: async (input) => {
      // Two separate failures with opposite remedies: a malformed path is the
      // caller's to fix, a failed fetch is worth a retry. One catch around
      // both would tell an agent holding a correct path to reformat it.
      let target: URL;
      try {
        target = new URL(
          String(input.path ?? location.pathname),
          location.origin,
        );
      } catch (err) {
        console.warn('WebMCP: get_page_markdown got an unparseable path', err);
        return text(
          'Could not parse that path. Provide a site-relative path such as /docs/introduction.',
        );
      }
      if (target.origin !== location.origin) {
        return text('Only paths on this site are allowed.');
      }
      try {
        const res = await fetch(target, {
          headers: { accept: 'text/markdown' },
        });
        if (!res.ok) return text(`HTTP ${res.status} for ${target.pathname}`);
        const type = res.headers.get('content-type') ?? '';
        if (!type.includes('text/markdown')) {
          return text(`No markdown version exists for ${target.pathname}.`);
        }
        return text(await res.text());
      } catch (err) {
        console.warn('WebMCP: get_page_markdown fetch failed', err);
        return text(
          `Network error while fetching ${target.pathname}. The path is valid, so retry.`,
        );
      }
    },
  },
  {
    name: 'list_agent_resources',
    description:
      "List DeepEval's machine-readable resources for agents: llms.txt, agent skills, and the Confident AI API catalog.",
    inputSchema: { type: 'object', properties: {} },
    execute: async () => {
      try {
        const res = await fetch('/.well-known/ai-catalog.json');
        if (!res.ok) return text(`HTTP ${res.status}`);
        return text(await res.text());
      } catch (err) {
        console.warn('WebMCP: list_agent_resources failed', err);
        return text('Could not fetch /.well-known/ai-catalog.json.');
      }
    },
  },
];

// Module-level guard against StrictMode's dev double-mount registering every
// tool twice. `registerTool` cleanup always calls `controller.abort()`, so a
// spec-conformant host has released the tools by the time cleanup returns
// whether or not it handed back a handle — the flag has to clear either way,
// or a host that honors the signal and returns nothing would leave the second
// mount with zero tools and no error. `provideContext` has no abort path, so
// only that branch keeps the flag set.
let registered = false;

export default function WebMcpTools() {
  useEffect(() => {
    const host =
      (document as Document & { modelContext?: ModelContext }).modelContext ??
      (navigator as Navigator & { modelContext?: ModelContext }).modelContext;
    if (!host || registered) return;
    registered = true;
    const controller = new AbortController();
    const unregisters: Array<() => void> = [];
    const usedRegisterTool = Boolean(host.registerTool);
    if (host.registerTool) {
      for (const tool of tools) {
        try {
          const handle = host.registerTool({
            ...tool,
            signal: controller.signal,
          });
          if (typeof handle === 'function') {
            unregisters.push(handle as () => void);
          } else if (
            handle !== null &&
            typeof handle === 'object' &&
            typeof (handle as { unregister?: unknown }).unregister === 'function'
          ) {
            unregisters.push(() =>
              (handle as { unregister: () => void }).unregister(),
            );
          }
        } catch (err) {
          console.warn(`WebMCP: failed to register ${tool.name}`, err);
        }
      }
    } else if (host.provideContext) {
      try {
        host.provideContext({ tools });
      } catch (err) {
        console.warn('WebMCP: provideContext failed', err);
      }
    }
    return () => {
      controller.abort();
      if (usedRegisterTool) registered = false;
      for (const unregister of unregisters) {
        try {
          unregister();
        } catch (err) {
          console.warn('WebMCP: unregister failed', err);
        }
      }
    };
  }, []);

  return null;
}
