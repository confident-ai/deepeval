import React from "react";
import { Box, Text } from "ink";
import type { InspectSpan, InspectTrace } from "@/inspect/model";
import {
  accentFor,
  durationMs,
  formatDuration,
  isErrored,
  metricTally,
  truncate,
} from "./styling.js";

export const TRACE_ROW_KEY = "__trace__";

export interface TreeRow {
  key: string;
  depth: number;
  /** Whether each ancestor level still has a sibling below it, for the guides. */
  guides: boolean[];
  isLast: boolean;
  /** `null` on the synthetic row standing in for the trace itself. */
  span: InspectSpan | null;
  hasChildren: boolean;
  expanded: boolean;
  label: string;
  tag: string;
  color: string;
  duration?: number;
  errored: boolean;
  passed: number;
  failed: number;
  metricErrors: number;
}

/** Visible rows, honouring collapsed nodes, so selection is a plain index. */
export function flattenTrace(
  trace: InspectTrace,
  collapsed: ReadonlySet<string>,
): TreeRow[] {
  const rows: TreeRow[] = [];
  const rootExpanded = !collapsed.has(TRACE_ROW_KEY);
  const traceTally = metricTally(trace);

  rows.push({
    key: TRACE_ROW_KEY,
    depth: 0,
    guides: [],
    isLast: true,
    span: null,
    hasChildren: trace.rootSpans.length > 0,
    expanded: rootExpanded,
    label: trace.caseName ?? trace.name ?? "trace",
    tag: "TRACE",
    color: "white",
    duration: durationMs(trace),
    errored: isErrored(trace),
    passed: traceTally.passed,
    failed: traceTally.failed,
    metricErrors: traceTally.errored,
  });

  if (!rootExpanded) return rows;

  const walk = (spans: InspectSpan[], depth: number, guides: boolean[]) => {
    spans.forEach((span, index) => {
      const isLast = index === spans.length - 1;
      const expanded = !collapsed.has(span.uuid);
      const accent = accentFor(span.type);
      const tally = metricTally(span);
      rows.push({
        key: span.uuid,
        depth,
        guides,
        isLast,
        span,
        hasChildren: span.children.length > 0,
        expanded,
        label: span.name ?? span.type ?? "span",
        tag: accent.label,
        color: accent.color,
        duration: durationMs(span),
        errored: isErrored(span),
        passed: tally.passed,
        failed: tally.failed,
        metricErrors: tally.errored,
      });
      if (expanded && span.children.length > 0) {
        walk(span.children, depth + 1, [...guides, !isLast]);
      }
    });
  };

  walk(trace.rootSpans, 1, []);
  return rows;
}

function prefixFor(row: TreeRow): string {
  if (row.depth === 0) return "";
  const guides = row.guides.map((hasSibling) => (hasSibling ? "│  " : "   "));
  return `${guides.join("")}${row.isLast ? "└─ " : "├─ "}`;
}

function badgesFor(row: TreeRow): string {
  const parts: string[] = [];
  if (row.passed) parts.push(`✓${row.passed}`);
  if (row.failed) parts.push(`✗${row.failed}`);
  if (row.metricErrors) parts.push(`!${row.metricErrors}`);
  if (row.errored) parts.push("ERR");
  return parts.join(" ");
}

function Row({
  row,
  width,
  selected,
}: {
  row: TreeRow;
  width: number;
  selected: boolean;
}): React.JSX.Element {
  const prefix = prefixFor(row);
  const chevron = row.hasChildren ? (row.expanded ? "▾" : "▸") : " ";
  const right = [badgesFor(row), formatDuration(row.duration)]
    .filter(Boolean)
    .join(" ");

  const fixed = prefix.length + 2 + row.tag.length + 1;
  const name = truncate(
    row.label,
    Math.max(3, width - fixed - right.length - 1),
  );
  const pad = Math.max(1, width - (fixed + name.length) - right.length);

  if (selected) {
    return (
      <Text backgroundColor="blue" color="white" bold wrap="truncate">
        {`${prefix}${chevron} ${row.tag} ${name}${" ".repeat(pad)}${right}`}
      </Text>
    );
  }

  return (
    <Text wrap="truncate">
      <Text dimColor>{prefix}</Text>
      <Text color={row.color}>{chevron} </Text>
      <Text color={row.color} bold>
        {row.tag}
      </Text>
      <Text> {name}</Text>
      <Text>{" ".repeat(pad)}</Text>
      <Text dimColor={!row.errored} color={row.errored ? "red" : undefined}>
        {right}
      </Text>
    </Text>
  );
}

export function SpanTree({
  rows,
  selectedIndex,
  width,
  height,
}: {
  rows: TreeRow[];
  selectedIndex: number;
  width: number;
  height: number;
}): React.JSX.Element {
  const maxOffset = Math.max(0, rows.length - height);
  const offset = Math.max(
    0,
    Math.min(selectedIndex - Math.floor(height / 2), maxOffset),
  );
  const visible = rows.slice(offset, offset + height);

  return (
    <Box flexDirection="column" width={width} flexShrink={0}>
      {visible.map((row, index) => (
        <Row
          key={row.key}
          row={row}
          width={width}
          selected={offset + index === selectedIndex}
        />
      ))}
    </Box>
  );
}
