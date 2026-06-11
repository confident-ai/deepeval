import * as fs from "node:fs";
import * as path from "node:path";
import { MultiBar, Presets } from "cli-progress";
import { TestResult, MetricData } from "./types";

/** A MultiBar styled like the rest of the runner (purple filled / dim track). */
export function newProgressMultiBar(): MultiBar {
  const FILLED = "\x1b[38;2;106;0;255m";
  const TRACK = "\x1b[38;2;70;70;82m";
  const ANSI_RESET = "\x1b[0m";
  const BARSIZE = 40;
  return new MultiBar(
    {
      format: "{label} {bar} {percentage}% {duration_formatted}",
      formatBar: (progress, options) => {
        const size = options.barsize ?? BARSIZE;
        const filled = Math.round(progress * size);
        return (
          FILLED +
          "━".repeat(filled) +
          TRACK +
          "━".repeat(size - filled) +
          ANSI_RESET
        );
      },
      hideCursor: true,
      clearOnComplete: true,
      stream: process.stderr,
      barsize: BARSIZE,
    },
    Presets.shades_classic,
  );
}

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

/** Word-wrap plain text to `width` display columns (hard-breaks over-long words). */
function wrapText(s: string, width: number): string[] {
  const out: string[] = [];
  for (const segment of s.split("\n")) {
    let cur = "";
    for (const word of segment.split(/\s+/).filter(Boolean)) {
      let w = word;
      if (cur === "") {
        while (visLen(w) > width) {
          out.push(w.slice(0, width));
          w = w.slice(width);
        }
        cur = w;
      } else if (visLen(cur) + 1 + visLen(word) <= width) {
        cur += " " + word;
      } else {
        out.push(cur);
        while (visLen(w) > width) {
          out.push(w.slice(0, width));
          w = w.slice(width);
        }
        cur = w;
      }
    }
    out.push(cur);
  }
  return out.length ? out : [""];
}

/** A table cell as one-or-more physical lines: short/ANSI cells stay single, long text wraps. */
function wrapCell(c: string, width: number): string[] {
  if (visLen(c) <= width && !c.includes("\n")) return [c];
  return wrapText(c, width);
}

/** A labeled panel line (`Label: value`) wrapped to `inner`, continuations indented. */
function wrapLabeledLine(prefix: string, value: string, inner: number): string[] {
  const indent = visLen(prefix);
  const chunks = wrapText(value, Math.max(10, inner - indent));
  return chunks.map((chunk, i) =>
    i === 0 ? `${prefix}${chunk}` : `${" ".repeat(indent)}${chunk}`,
  );
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
  // A row can span multiple physical lines when a cell (e.g. Reason) wraps.
  const row = (cells: string[], sep: string): string[] => {
    const wrapped = cells.map((c, i) => wrapCell(c, widths[i]));
    const height = Math.max(...wrapped.map((w) => w.length), 1);
    const lines: string[] = [];
    for (let ln = 0; ln < height; ln++) {
      lines.push(
        wrapped
          .map((w, i) => ` ${padEndVis(w[ln] ?? "", widths[i])} `)
          .join(sep),
      );
    }
    return lines;
  };
  const header = row(
    headers.map((h) => `${PURPLE}${BOLD}${h}${RESET}`),
    "┃",
  );
  const divider = widths.map((w) => "━".repeat(w + 2)).join("╇");
  return [...header, divider, ...rows.flatMap((r) => row(r, "│"))];
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
        lines.push(...wrapLabeledLine(`  ${BOLD}${role}:${RESET} `, turn.content, inner));
      }
    } else {
      lines.push(...wrapLabeledLine(`${CYAN}${BOLD}Input:${RESET} `, String(tc.input), inner));
      lines.push(
        ...wrapLabeledLine(`${CYAN}${BOLD}Actual Output:${RESET} `, String(tc.actualOutput), inner),
      );
      if (tc.expectedOutput && tc.expectedOutput !== "N/A") {
        lines.push(
          ...wrapLabeledLine(`${CYAN}${BOLD}Expected Output:${RESET} `, tc.expectedOutput, inner),
        );
      }
    }
    lines.push("");
    lines.push(`${PURPLE}${BOLD}Metrics${RESET}`);

    const rows = metrics.map((m) => {
      // Full reason for shown (failing) cases — the table wraps it. Passing
      // cases are truncated (only relevant when truncatePassing is off).
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

  const aggRows = aggregateRows(sorted);
  if (aggRows.length > 0) {
    panel(
      [
        `${BOLD}Aggregate Metrics${RESET}`,
        "",
        ...tableLines(
          ["Metric", "Average Score", "Pass Rate", "Total"],
          aggRows,
          inner,
        ),
      ],
      PURPLE,
    );
  }
  console.log();
}

/** Aggregate rows `[metric, avgScore, passRate, total]` across all test cases. */
function aggregateRows(testResults: TestResult[]): string[][] {
  const agg = new Map<
    string,
    { total: number; passes: number; scoreSum: number; scoreCount: number }
  >();
  for (const tc of testResults) {
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
  return [...agg.entries()].map(([name, a]) => [
    name,
    a.scoreCount > 0 ? (a.scoreSum / a.scoreCount).toFixed(2) : "N/A",
    a.total > 0 ? `${((a.passes / a.total) * 100).toFixed(2)}%` : "N/A",
    String(a.total),
  ]);
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

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

/**
 * Export the results to a Markdown/MDX file (mirrors Python's
 * `export_to_markdown`). Markdown is valid MDX, so both file types share the
 * same content. Returns the written file path.
 */
export function exportToMarkdown(
  testResults: TestResult[],
  outputDir: string,
  fileType: "md" | "mdx" = "md",
  evaluationName = "evaluation",
): string {
  fs.mkdirSync(outputDir, { recursive: true });
  const d = new Date();
  const ts =
    `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}` +
    `_${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
  const safe = (evaluationName || "evaluation").replace(/\s+/g, "_").toLowerCase();
  const filepath = path.join(outputDir, `${safe}_${ts}.${fileType}`);

  const sorted = [...testResults].sort(
    (a, b) => (a.index ?? Infinity) - (b.index ?? Infinity),
  );
  const md: string[] = ["# 🚀 DeepEval Evaluation Results\n"];

  for (const tc of sorted) {
    md.push(`## ${tc.success ? "✅ PASS" : "❌ FAIL"} - ${tc.name}\n`);
    md.push("<details><summary><b>View Test Case Data</b></summary>\n");
    if (tc.conversational) {
      for (const turn of tc.turns ?? []) {
        const role = turn.role.charAt(0).toUpperCase() + turn.role.slice(1);
        md.push(`- **${role}**: ${turn.content}`);
      }
    } else {
      md.push(`- **Input:** ${tc.input}`);
      md.push(`- **Actual Output:** ${tc.actualOutput}`);
      if (tc.expectedOutput && tc.expectedOutput !== "N/A") {
        md.push(`- **Expected Output:** ${tc.expectedOutput}`);
      }
    }
    md.push("\n</details>\n\n### Metrics\n");
    md.push("| Status | Metric | Score | Threshold | Reason |");
    md.push("|:---:|:---|:---:|:---:|:---|");
    for (const m of tc.metricsData ?? []) {
      const icon = m.skipped
        ? "⏭️ SKIP"
        : m.error
          ? "⚠️ ERROR"
          : m.success
            ? "✅"
            : "❌";
      const score = m.score != null ? m.score.toFixed(2) : "N/A";
      const thresh = m.threshold != null ? m.threshold.toFixed(2) : "N/A";
      const reason = String(m.reason ?? m.error ?? "N/A").replace(
        /\n/g,
        " <br> ",
      );
      md.push(`| ${icon} | **${m.name}** | ${score} | ${thresh} | ${reason} |`);
    }
    md.push("\n---\n");
  }

  const aggRows = aggregateRows(sorted);
  if (aggRows.length > 0) {
    md.push("## Aggregate Metrics\n");
    md.push("| Metric | Average Score | Pass Rate | Total |");
    md.push("|:---|:---:|:---:|:---:|");
    for (const [name, avg, rate, total] of aggRows) {
      md.push(`| **${name}** | ${avg} | ${rate} | ${total} |`);
    }
    md.push("\n---\n");
  }

  fs.writeFileSync(filepath, md.join("\n"), "utf-8");
  console.log(`✅ Markdown report saved to: ${filepath}`);
  return filepath;
}
