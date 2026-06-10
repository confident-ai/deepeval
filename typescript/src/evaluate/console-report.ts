import { TestResult, MetricData } from "./types";

// Palette mirrors Python's console_report (DEEPEVAL_PURPLE / DEEPEVAL_GREEN).
const PURPLE = "\x1b[38;2;106;0;255m";
const GREEN = "\x1b[38;2;25;227;160m";
const RED = "\x1b[31m";
const CYAN = "\x1b[36m";
const YELLOW = "\x1b[33m";
const BOLD = "\x1b[1m";
const RESET = "\x1b[0m";

// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-9;]*m/g;

/** Terminal display width of a single code point (emoji/CJK = 2, combining = 0). */
function charWidth(cp: number): number {
  if (cp === 0x200d || (cp >= 0xfe00 && cp <= 0xfe0f)) return 0; // ZWJ + variation selectors
  if (
    (cp >= 0x1100 && cp <= 0x115f) || // Hangul Jamo
    (cp >= 0x2300 && cp <= 0x23ff) || // misc technical (⏭ …)
    (cp >= 0x2600 && cp <= 0x27bf) || // symbols + dingbats (✅ ❌ ✨ ❤ ⚠ …)
    (cp >= 0x2b00 && cp <= 0x2bff) ||
    (cp >= 0x2e80 && cp <= 0xa4cf) || // CJK
    (cp >= 0xac00 && cp <= 0xd7a3) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0xfe30 && cp <= 0xfe4f) ||
    (cp >= 0xff00 && cp <= 0xff60) ||
    (cp >= 0xffe0 && cp <= 0xffe6) ||
    (cp >= 0x1f000 && cp <= 0x1faff) // emoji (🚀 🎯 🎉 …)
  ) {
    return 2;
  }
  return 1;
}

/** Terminal display width of a string, ignoring ANSI codes, counting wide chars as 2. */
function visLen(s: string): number {
  let w = 0;
  for (const ch of s.replace(ANSI, "")) w += charWidth(ch.codePointAt(0)!);
  return w;
}

function padEndVis(s: string, width: number): string {
  return s + " ".repeat(Math.max(0, width - visLen(s)));
}

/** Truncate to `max` display columns, preserving ANSI codes (which take 0 columns). */
function truncVis(s: string, max: number): string {
  if (visLen(s) <= max) return s;
  let out = "";
  let w = 0;
  let i = 0;
  while (i < s.length) {
    if (s[i] === "\x1b") {
      // eslint-disable-next-line no-control-regex
      const m = s.slice(i).match(/^\x1b\[[0-9;]*m/);
      if (m) {
        out += m[0];
        i += m[0].length;
        continue;
      }
    }
    const cp = s.codePointAt(i)!;
    const ch = String.fromCodePoint(cp);
    const cw = charWidth(cp);
    if (w + cw > max - 1) break;
    out += ch;
    w += cw;
    i += ch.length;
  }
  return out + "…" + RESET;
}

function termWidth(): number {
  return process.stdout.columns || 100;
}

/** A full-width rounded-border panel (rich `Panel`), border optionally colored. */
function panel(lines: string[], borderColor = ""): void {
  const width = termWidth();
  const inner = width - 4; // "│ " + " │"
  const c = borderColor;
  const r = borderColor ? RESET : "";
  const out = [`${c}╭${"─".repeat(width - 2)}╮${r}`];
  for (const line of lines) {
    const clipped = truncVis(line, inner);
    out.push(`${c}│${r} ${padEndVis(clipped, inner)} ${c}│${r}`);
  }
  out.push(`${c}╰${"─".repeat(width - 2)}╯${r}`);
  console.log(out.join("\n"));
}

/** Column widths for a table whose rendered width fills `target`. */
function computeWidths(
  headers: string[],
  rows: string[][],
  target: number,
): number[] {
  const n = headers.length;
  const widths = headers.map((h, i) =>
    Math.max(visLen(h), ...rows.map((row) => visLen(row[i] ?? ""))),
  );
  // rendered width = Σ(w+2) + (n-1) separators = Σw + 3n - 1
  const natural = widths.reduce((a, b) => a + b, 0) + 3 * n - 1;
  const extra = target - natural;
  if (extra > 0) {
    const per = Math.floor(extra / n);
    const rem = extra % n;
    for (let i = 0; i < n; i++) widths[i] += per + (i < rem ? 1 : 0);
  } else if (extra < 0) {
    const idx = widths.indexOf(Math.max(...widths));
    widths[idx] = Math.max(10, widths[idx] + extra);
  }
  return widths;
}

/** Render a rich-style table: heavy header (┃ / ━╇), light body (│). */
function tableLines(
  headers: string[],
  rows: string[][],
  innerWidth: number,
): string[] {
  const widths = computeWidths(headers, rows, innerWidth);
  const row = (cells: string[], sep: string) =>
    cells
      .map((c, i) => ` ${padEndVis(truncVis(c, widths[i]), widths[i])} `)
      .join(sep);
  const header = row(
    headers.map((h) => `${PURPLE}${BOLD}${h}${RESET}`),
    "┃",
  );
  const divider = widths.map((w) => "━".repeat(w + 2)).join("╇");
  return [header, divider, ...rows.map((r) => row(r, "│"))];
}

function metricStatusCell(m: MetricData): string {
  if (m.skipped) return `${YELLOW}${BOLD}SKIP${RESET}`;
  if (m.error) return `${RED}${BOLD}ERROR${RESET}`;
  return m.success ? `${GREEN}${BOLD}PASS${RESET}` : `${RED}${BOLD}FAIL${RESET}`;
}

/**
 * Print the per-test-case panels + an aggregate-metrics panel — the TS
 * counterpart of Python's rich `EvaluationConsoleReport`. Passing cases are
 * shown as a one-line summary panel (`truncatePassing`, default true); failing
 * cases get a full detail panel + metrics table.
 */
export function printResultsTable(
  testResults: TestResult[],
  options: { truncatePassing?: boolean } = {},
): void {
  const truncate = options.truncatePassing ?? true;
  const inner = termWidth() - 4;

  const sorted = [...testResults].sort(
    (a, b) => (a.index ?? Infinity) - (b.index ?? Infinity),
  );

  console.log();
  panel([`${PURPLE}${BOLD}🚀 DeepEval Evaluation Results${RESET}`]);

  for (const tc of sorted) {
    const color = tc.success ? GREEN : RED;
    const icon = tc.success ? "✅" : "❌";
    const metrics = tc.metricsData ?? [];

    if (truncate && tc.success) {
      panel(
        [
          `${color}${BOLD}${icon} ${tc.name} (Passed ${metrics.length} metrics)${RESET}`,
        ],
        color,
      );
      continue;
    }

    const lines = [`${color}${BOLD}${icon} ${tc.name}${RESET}`, ""];
    if (tc.conversational) {
      lines.push(`${CYAN}${BOLD}Conversation Turns${RESET}`);
      for (const turn of tc.turns ?? []) {
        const role = turn.role.charAt(0).toUpperCase() + turn.role.slice(1);
        lines.push(`  ${BOLD}${role}:${RESET} ${turn.content}`);
      }
    } else {
      lines.push(`${CYAN}${BOLD}Input:${RESET} ${tc.input}`);
      lines.push(`${CYAN}${BOLD}Actual Output:${RESET} ${tc.actualOutput}`);
      if (tc.expectedOutput && tc.expectedOutput !== "N/A") {
        lines.push(
          `${CYAN}${BOLD}Expected Output:${RESET} ${tc.expectedOutput}`,
        );
      }
    }
    lines.push("");
    lines.push(`${PURPLE}${BOLD}Metrics${RESET}`);

    const rows = metrics.map((m) => {
      let reason = String(m.reason ?? m.error ?? "N/A");
      if (truncate && m.success && reason.length > 50) {
        reason = reason.slice(0, 47) + "...";
      }
      return [
        metricStatusCell(m),
        m.name,
        m.score != null ? m.score.toFixed(2) : "N/A",
        m.threshold != null ? m.threshold.toFixed(2) : "N/A",
        reason,
      ];
    });
    lines.push(
      ...tableLines(
        ["Status", "Metric", "Score", "Threshold", "Reason"],
        rows,
        inner,
      ),
    );
    panel(lines, color);
  }

  // Aggregate metrics across all test cases (by metric name).
  const agg = new Map<
    string,
    { total: number; passes: number; scoreSum: number; scoreCount: number }
  >();
  for (const tc of sorted) {
    for (const m of tc.metricsData ?? []) {
      const a = agg.get(m.name) ?? {
        total: 0,
        passes: 0,
        scoreSum: 0,
        scoreCount: 0,
      };
      a.total += 1;
      if (m.success) a.passes += 1;
      if (m.score != null) {
        a.scoreSum += m.score;
        a.scoreCount += 1;
      }
      agg.set(m.name, a);
    }
  }
  if (agg.size > 0) {
    const rows = [...agg.entries()].map(([name, a]) => [
      name,
      a.scoreCount > 0 ? (a.scoreSum / a.scoreCount).toFixed(2) : "N/A",
      a.total > 0 ? `${((a.passes / a.total) * 100).toFixed(2)}%` : "N/A",
      String(a.total),
    ]);
    panel(
      [
        `${BOLD}Aggregate Metrics${RESET}`,
        "",
        ...tableLines(
          ["Metric", "Average Score", "Pass Rate", "Total"],
          rows,
          inner,
        ),
      ],
      PURPLE,
    );
  }
  console.log();
}

const SEP = "=".repeat(80);

/** Mirror Python's "No hyperparameters logged" warning (we don't log them yet). */
export function printHyperparametersWarning(): void {
  console.log(
    `\n${BOLD}${YELLOW}⚠ WARNING:${RESET} No hyperparameters logged.\n` +
      `» Log hyperparameters to attribute prompts and models to your test runs.\n\n` +
      SEP,
  );
}

/** Mirror Python's wrap-up completion summary (printed when not posting to Confident AI). */
export function printCompletionSummary(opts: {
  runDuration: number;
  tokenCost: number;
  passed: number;
  failed: number;
}): void {
  const { runDuration, tokenCost, passed, failed } = opts;
  const total = passed + failed;
  const passRate = total ? Math.round((passed / total) * 1000) / 10 : 0;
  const cost = tokenCost ? `${tokenCost} USD` : "None";
  console.log(
    `\n\n${GREEN}✓${RESET} Evaluation completed 🎉! ` +
      `(time taken: ${runDuration.toFixed(2)}s | token cost: ${cost})\n` +
      `» Test Results (${total} total tests):\n` +
      `  » Pass Rate: ${passRate}% | Passed: ${GREEN}${BOLD}${passed}${RESET} | Failed: ${RED}${BOLD}${failed}${RESET}\n\n` +
      `${SEP}\n\n` +
      `» Want to share evals with your team, or a place for your test cases to live? ❤️ 🏡\n` +
      `  » Set ${BOLD}CONFIDENT_API_KEY${RESET} to post test runs to ${PURPLE}Confident AI${RESET}.\n`,
  );
}
