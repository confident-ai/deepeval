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

const tools: WebMcpTool[] = [
  {
    name: 'search_deepeval_docs',
    description:
      'Search the DeepEval documentation, guides, tutorials, integrations, and blog. Returns titles and URLs. Request any result URL with Accept: text/markdown (or append .mdx) for markdown.',
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
        if (!res.ok) return text(`HTTP ${res.status}`);
        const results = (await res.json()) as SearchResult[];
        const pages = results.filter((r) => r.type === 'page').slice(0, 5);
        const top = pages.length > 0 ? pages : results.slice(0, 5);
        if (top.length === 0) return text('No results.');
        return text(
          top
            .map((r) => `${r.content}\n${location.origin}${r.url}`)
            .join('\n\n'),
        );
      } catch (err) {
        console.warn('WebMCP: search failed', err);
        return text('Search is unavailable. Fetch /llms.txt instead.');
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
      try {
        const target = new URL(
          String(input.path ?? location.pathname),
          location.origin,
        );
        if (target.origin !== location.origin) {
          return text('Only paths on this site are allowed.');
        }
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
        console.warn('WebMCP: get_page_markdown failed', err);
        return text(
          'Could not fetch that path. Provide a site-relative path such as /docs/introduction.',
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

// Module-level guard: if the host ignores both the abort signal and the
// returned handle, effect cleanup cannot unregister, and StrictMode's dev
// double-mount would register every tool twice. The flag resets only when
// cleanup had real handles to unregister with.
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
      if (unregisters.length > 0) registered = false;
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
