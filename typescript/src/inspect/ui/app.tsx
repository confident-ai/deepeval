import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, render, useApp, useInput, useWindowSize } from "ink";
import type { InspectUiOptions } from "@/inspect/model";
import { SpanTree, flattenTrace } from "./span-tree.js";
import { DetailsPane, buildDetailLines } from "./details.js";
import { countSpans, formatCost, formatDuration } from "./styling.js";

const KEY_HINTS =
  "↑↓/jk move  ←→/hl fold  n/p trace  PgUp/PgDn scroll  g/G top/end  q quit";

function Header({
  options,
  traceIndex,
}: {
  options: InspectUiOptions;
  traceIndex: number;
}): React.JSX.Element {
  const { traces, summary, sourcePath } = options;
  const trace = traces[traceIndex]!;
  const spans = countSpans(trace);
  const facts: string[] = [`trace ${traceIndex + 1}/${traces.length}`];
  facts.push(`${spans} span${spans === 1 ? "" : "s"}`);
  if (summary?.testPassed != null || summary?.testFailed != null) {
    facts.push(`${summary.testPassed ?? 0} passed`);
    facts.push(`${summary.testFailed ?? 0} failed`);
  }
  if (summary?.runDuration != null) {
    facts.push(formatDuration(summary.runDuration * 1000));
  }
  if (summary?.evaluationCost != null) {
    facts.push(formatCost(summary.evaluationCost));
  }

  return (
    <Box flexDirection="column">
      <Text wrap="truncate">
        <Text bold color="magenta">
          deepeval inspect
        </Text>
        <Text dimColor>{`  ${sourcePath}`}</Text>
      </Text>
      <Text wrap="truncate" dimColor>
        {facts.join("  ·  ")}
        {trace.caseName ? `  ·  ${trace.caseName}` : ""}
      </Text>
    </Box>
  );
}

function InspectApp(options: InspectUiOptions): React.JSX.Element {
  const { traces } = options;
  const { exit } = useApp();
  const { columns: width, rows: height } = useWindowSize();

  const [traceIndex, setTraceIndex] = useState(0);
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(new Set());
  const [cursor, setCursor] = useState(0);
  const [scroll, setScroll] = useState(0);

  const trace = traces[traceIndex]!;
  const rows = useMemo(
    () => flattenTrace(trace, collapsed),
    [trace, collapsed],
  );

  const safeCursor = Math.min(Math.max(0, cursor), rows.length - 1);
  const selected = rows[safeCursor]!;

  const bodyHeight = Math.max(5, height - 4);
  const treeWidth = Math.max(24, Math.min(52, Math.floor(width * 0.42)));
  const detailsWidth = Math.max(20, width - treeWidth - 3);

  const detailLines = useMemo(
    () => buildDetailLines(selected.span, trace, detailsWidth),
    [selected.span, trace, detailsWidth],
  );

  useEffect(() => setScroll(0), [selected.key, traceIndex]);

  const maxScroll = Math.max(0, detailLines.length - bodyHeight);

  const toggle = (key: string, expand: boolean) => {
    setCollapsed((current) => {
      const next = new Set(current);
      if (expand) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const moveBy = (delta: number) =>
    setCursor((c) =>
      Math.max(
        0,
        Math.min(Math.min(c, rows.length - 1) + delta, rows.length - 1),
      ),
    );

  const expandOrDescend = () => {
    if (!selected.hasChildren) return;
    if (!selected.expanded) toggle(selected.key, true);
    else moveBy(1);
  };

  const collapseOrAscend = () => {
    if (selected.hasChildren && selected.expanded) {
      toggle(selected.key, false);
      return;
    }
    for (let i = safeCursor - 1; i >= 0; i -= 1) {
      if (rows[i]!.depth < selected.depth) {
        setCursor(i);
        return;
      }
    }
  };

  const cycleTrace = (delta: number) => {
    if (traces.length < 2) return;
    setTraceIndex((i) => (i + delta + traces.length) % traces.length);
    setCursor(0);
  };

  const scrollBy = (delta: number) =>
    setScroll((s) => Math.max(0, Math.min(s + delta, maxScroll)));

  useInput((input, key) => {
    if (key.downArrow) return moveBy(1);
    if (key.upArrow) return moveBy(-1);
    if (key.rightArrow) return expandOrDescend();
    if (key.leftArrow) return collapseOrAscend();
    if (key.pageDown) return scrollBy(bodyHeight);
    if (key.pageUp) return scrollBy(-bodyHeight);
    if (key.tab) return cycleTrace(1);
    if (key.return) {
      if (selected.hasChildren) toggle(selected.key, !selected.expanded);
      return;
    }
    if (key.ctrl) {
      if (input === "c") exit();
      else if (input === "d") scrollBy(bodyHeight);
      else if (input === "u") scrollBy(-bodyHeight);
      return;
    }

    // Ink hands a batched keypress over as one string, so each character has to
    // be applied in turn or held keys get dropped.
    for (const character of input) {
      switch (character) {
        case "q":
          exit();
          return;
        case "j":
          moveBy(1);
          break;
        case "k":
          moveBy(-1);
          break;
        case "g":
          setCursor(0);
          break;
        case "G":
          setCursor(rows.length - 1);
          break;
        case "l":
          expandOrDescend();
          break;
        case "h":
          collapseOrAscend();
          break;
        case "n":
          cycleTrace(1);
          break;
        case "p":
          cycleTrace(-1);
          break;
        default:
          break;
      }
    }
  });

  return (
    <Box flexDirection="column" width={width}>
      <Header options={options} traceIndex={traceIndex} />
      <Box flexDirection="row" height={bodyHeight}>
        <SpanTree
          rows={rows}
          selectedIndex={safeCursor}
          width={treeWidth}
          height={bodyHeight}
        />
        <Box
          borderStyle="single"
          borderTop={false}
          borderRight={false}
          borderBottom={false}
          borderDimColor
          paddingLeft={1}
          flexGrow={1}
        >
          <DetailsPane
            lines={detailLines}
            width={detailsWidth}
            height={bodyHeight}
            scroll={Math.min(scroll, maxScroll)}
          />
        </Box>
      </Box>
      <Text wrap="truncate" dimColor>
        {KEY_HINTS}
        {maxScroll > 0
          ? `  ·  ${Math.min(scroll, maxScroll)}/${maxScroll}`
          : ""}
      </Text>
    </Box>
  );
}

export async function mount(options: InspectUiOptions): Promise<void> {
  const instance = render(<InspectApp {...options} />, { exitOnCtrlC: true });
  await instance.waitUntilExit();
}
