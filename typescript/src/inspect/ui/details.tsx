import React from "react";
import { Box, Text } from "ink";
import type { InspectSpan, InspectTrace } from "@/inspect/model";
import {
  accentFor,
  durationMs,
  formatCost,
  formatDuration,
  isErrored,
  metricsOf,
  spanCost,
  stringify,
  wrapText,
} from "./styling.js";

export interface DetailLine {
  text: string;
  color?: string;
  bold?: boolean;
  dim?: boolean;
}

class LineBuffer {
  readonly lines: DetailLine[] = [];

  constructor(private readonly width: number) {}

  blank(): void {
    if (this.lines.length > 0) this.lines.push({ text: "" });
  }

  heading(text: string): void {
    this.blank();
    this.lines.push({ text: text.toUpperCase(), bold: true, color: "cyan" });
  }

  fact(label: string, value: string): void {
    this.lines.push({ text: `${label.padEnd(14)}${value}` });
  }

  body(text: string, options: { indent?: number; dim?: boolean } = {}): void {
    const indent = " ".repeat(options.indent ?? 0);
    for (const line of wrapText(
      text,
      Math.max(8, this.width - indent.length),
    )) {
      this.lines.push({ text: `${indent}${line}`, dim: options.dim });
    }
  }

  raw(line: DetailLine): void {
    this.lines.push(line);
  }
}

function appendMetrics(
  buffer: LineBuffer,
  node: InspectSpan | InspectTrace,
): void {
  const metrics = metricsOf(node);
  if (metrics.length === 0) return;

  buffer.heading("metrics");
  for (const metric of metrics) {
    const name = metric.name ?? "metric";
    const score = metric.score != null ? metric.score.toFixed(4) : "—";
    const threshold = metric.threshold != null ? metric.threshold : "—";

    if (metric.error) {
      buffer.raw({ text: `${name}  ERRORED`, bold: true, color: "red" });
      buffer.body(metric.error, { indent: 2, dim: true });
      continue;
    }

    // A metric with no threshold reports a score but no verdict.
    const verdict =
      metric.success == null ? "NONE" : metric.success ? "PASS" : "FAIL";
    buffer.raw({
      text: `${name}  ${score} / ${threshold}  ${verdict}`,
      bold: true,
      color: metric.success == null ? "gray" : metric.success ? "green" : "red",
    });
    if (metric.reason) buffer.body(metric.reason, { indent: 2 });
    if (metric.evaluationModel) {
      buffer.raw({ text: `  model: ${metric.evaluationModel}`, dim: true });
    }
  }
}

function appendValue(
  buffer: LineBuffer,
  heading: string,
  value: unknown,
): void {
  if (value == null) return;
  if (Array.isArray(value) && value.length === 0) return;
  buffer.heading(heading);
  buffer.body(stringify(value));
}

/** Flat lines rather than nested components, so the pane can scroll by offset. */
export function buildDetailLines(
  node: InspectSpan | null,
  trace: InspectTrace,
  width: number,
): DetailLine[] {
  const buffer = new LineBuffer(width);
  const target = node ?? trace;
  const accent = node
    ? accentFor(node.type)
    : { label: "TRACE", color: "white" };

  buffer.raw({
    text: node
      ? (node.name ?? node.type ?? "span")
      : (trace.caseName ?? trace.name ?? "trace"),
    bold: true,
  });
  buffer.raw({
    text: `${accent.label}${isErrored(target) ? "  ERRORED" : ""}`,
    color: isErrored(target) ? "red" : accent.color,
    bold: true,
  });

  buffer.blank();
  buffer.fact("duration", formatDuration(durationMs(target)));

  if (node) {
    if (node.model) buffer.fact("model", node.model);
    if (node.inputTokenCount != null || node.outputTokenCount != null) {
      buffer.fact(
        "tokens",
        `${node.inputTokenCount ?? 0} in / ${node.outputTokenCount ?? 0} out`,
      );
    }
    const cost = spanCost(node);
    if (cost != null) buffer.fact("cost", formatCost(cost));
    if (node.embedder) buffer.fact("embedder", node.embedder);
    if (node.topK != null) buffer.fact("top k", String(node.topK));
    if (node.chunkSize != null) {
      buffer.fact("chunk size", String(node.chunkSize));
    }
    if (node.description) buffer.fact("description", node.description);
    buffer.fact("uuid", node.uuid);
  } else {
    if (trace.environment) buffer.fact("environment", trace.environment);
    if (trace.threadId) buffer.fact("thread", trace.threadId);
    if (trace.userId) buffer.fact("user", trace.userId);
    if (trace.tags?.length) buffer.fact("tags", trace.tags.join(", "));
    if (trace.uuid) buffer.fact("uuid", trace.uuid);
  }

  if (node?.error) {
    buffer.heading("error");
    buffer.raw({ text: node.error, color: "red" });
  }

  appendMetrics(buffer, target);
  appendValue(buffer, "input", target.input);
  appendValue(buffer, "output", target.output);
  appendValue(buffer, "expected output", target.expectedOutput);
  appendValue(buffer, "retrieval context", target.retrievalContext);
  appendValue(buffer, "tools called", target.toolsCalled);
  appendValue(buffer, "expected tools", target.expectedTools);
  if (node?.availableTools?.length) {
    appendValue(buffer, "available tools", node.availableTools);
  }
  if (target.metadata && Object.keys(target.metadata).length > 0) {
    appendValue(buffer, "metadata", target.metadata);
  }

  return buffer.lines;
}

export function DetailsPane({
  lines,
  width,
  height,
  scroll,
}: {
  lines: DetailLine[];
  width: number;
  height: number;
  scroll: number;
}): React.JSX.Element {
  const visible = lines.slice(scroll, scroll + height);

  return (
    <Box flexDirection="column" width={width} flexGrow={1}>
      {visible.map((line, index) => (
        <Text
          key={`${scroll + index}-${line.text}`}
          bold={line.bold}
          color={line.color}
          dimColor={line.dim}
          wrap="truncate"
        >
          {line.text === "" ? " " : line.text}
        </Text>
      ))}
    </Box>
  );
}
