import type { InspectSpan, InspectTrace, MetricLike } from "@/inspect/model";

export interface SpanAccent {
  label: string;
  color: string;
}

const ACCENTS: Record<string, SpanAccent> = {
  agent: { label: "AGENT", color: "magenta" },
  llm: { label: "LLM", color: "cyan" },
  retriever: { label: "RETRIEVER", color: "yellow" },
  tool: { label: "TOOL", color: "green" },
  base: { label: "SPAN", color: "blue" },
};

export function accentFor(type: string | undefined): SpanAccent {
  return ACCENTS[(type ?? "base").toLowerCase()] ?? ACCENTS.base!;
}

export function isErrored(node: {
  status?: string;
  error?: string | null;
}): boolean {
  return node.status === "ERRORED" || Boolean(node.error);
}

export function durationMs(node: {
  startTime?: string;
  endTime?: string;
}): number | undefined {
  if (!node.startTime || !node.endTime) return undefined;
  const start = Date.parse(node.startTime);
  const end = Date.parse(node.endTime);
  if (Number.isNaN(start) || Number.isNaN(end)) return undefined;
  return Math.max(0, end - start);
}

export function formatDuration(ms: number | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)}s`;
  const minutes = Math.floor(ms / 60_000);
  return `${minutes}m ${((ms % 60_000) / 1000).toFixed(0)}s`;
}

export function formatCost(cost: number | undefined): string {
  if (cost == null) return "—";
  return cost < 0.01 ? `$${cost.toFixed(6)}` : `$${cost.toFixed(4)}`;
}

export function spanCost(span: InspectSpan): number | undefined {
  const inputCost = (span.inputTokenCount ?? 0) * (span.costPerInputToken ?? 0);
  const outputCost =
    (span.outputTokenCount ?? 0) * (span.costPerOutputToken ?? 0);
  const total = inputCost + outputCost;
  return total > 0 ? total : undefined;
}

export function metricsOf(node: InspectSpan | InspectTrace): MetricLike[] {
  const raw = node.metricsData;
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (m): m is MetricLike => typeof m === "object" && m !== null,
  );
}

export function metricTally(node: InspectSpan | InspectTrace): {
  passed: number;
  failed: number;
  errored: number;
} {
  let passed = 0;
  let failed = 0;
  let errored = 0;
  for (const metric of metricsOf(node)) {
    if (metric.error) errored += 1;
    // A metric with no threshold has no verdict, so it tallies as neither.
    else if (metric.success == null) continue;
    else if (metric.success) passed += 1;
    else failed += 1;
  }
  return { passed, failed, errored };
}

/** Total spans in the trace, independent of what is currently expanded. */
export function countSpans(trace: InspectTrace): number {
  const walk = (span: InspectSpan): number =>
    1 + span.children.reduce((total, child) => total + walk(child), 0);
  return trace.rootSpans.reduce((total, span) => total + walk(span), 0);
}

export function truncate(text: string, width: number): string {
  if (width <= 0) return "";
  return text.length <= width
    ? text
    : `${text.slice(0, Math.max(0, width - 1))}…`;
}

export function stringify(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/**
 * Greedy word wrap. The detail pane scrolls by an exact line offset, so content
 * has to be pre-broken to the pane width rather than reflowed by the terminal.
 */
export function wrapText(text: string, width: number): string[] {
  if (width <= 1) return [text];
  const out: string[] = [];

  for (const paragraph of text.split("\n")) {
    const indent = /^[ \t]*/.exec(paragraph)?.[0] ?? "";
    const content = paragraph.slice(indent.length).trimEnd();
    if (content === "") {
      out.push("");
      continue;
    }

    const available = Math.max(4, width - indent.length);
    let line = "";
    const flush = () => {
      out.push(`${indent}${line}`);
      line = "";
    };

    for (const word of content.split(/\s+/).filter(Boolean)) {
      if (line === "") line = word;
      else if (line.length + 1 + word.length <= available)
        line = `${line} ${word}`;
      else {
        flush();
        line = word;
      }

      while (line.length > available) {
        out.push(`${indent}${line.slice(0, available)}`);
        line = line.slice(available);
      }
    }
    if (line !== "") flush();
  }

  return out;
}
