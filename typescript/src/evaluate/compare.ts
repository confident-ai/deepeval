import { MultiBar, type SingleBar, Presets } from "cli-progress";
import { ArenaTestCase } from "../test-case";
import { BaseArenaMetric } from "../metrics";
import { MissingTestCaseParamsError } from "../errors";
import {
  DisplayConfig,
  ErrorConfig,
  DEFAULT_DISPLAY_CONFIG,
  DEFAULT_ERROR_CONFIG,
} from "./configs";
import { postExperiment } from "./confident";
import { type ArenaCaseResult } from "./types";

const PURPLE = "\x1b[38;2;106;0;255m";
const GREEN = "\x1b[38;2;25;227;160m";
const TRACK = "\x1b[38;2;70;70;82m";
const BOLD = "\x1b[1m";
const RESET = "\x1b[0m";
const SEP = "=".repeat(80);

export interface CompareOptions {
  name?: string;
  displayConfig?: DisplayConfig;
  errorConfig?: ErrorConfig;
}

/**
 * Compare contestants head-to-head with an arena metric. Runs the metric over
 * each {@link ArenaTestCase}, tallies wins per contestant, prints a summary, and
 * returns `{ contestantName: winCount }`. Posts the results to Confident AI as
 * an experiment when logged in. TS port of Python's `compare()`.
 */
export async function compare(
  testCases: ArenaTestCase[],
  metric: BaseArenaMetric,
  options: CompareOptions = {},
): Promise<Record<string, number>> {
  const display: Required<DisplayConfig> = {
    ...DEFAULT_DISPLAY_CONFIG,
    ...options.displayConfig,
  };
  const errorCfg: Required<ErrorConfig> = {
    ...DEFAULT_ERROR_CONFIG,
    ...options.errorConfig,
  };

  if (display.showIndicator) {
    process.stderr.write(metric.describe() + "\n");
  }

  // A top "🆚 Comparing N contestants" bar + one "🧐 Picking a winner (#i)"
  // bar per case (3 steps each), mirroring Python's rich progress.
  let multibar: MultiBar | null = null;
  let mainBar: SingleBar | null = null;
  let caseBars: SingleBar[] = [];
  if (display.showIndicator && testCases.length > 0) {
    const BARSIZE = 40;
    multibar = new MultiBar(
      {
        format: (_opts, params, payload) => {
          const filled = Math.round(params.progress * BARSIZE);
          const bar =
            PURPLE +
            "━".repeat(filled) +
            TRACK +
            "━".repeat(BARSIZE - filled) +
            RESET;
          const pct = String(Math.round(params.progress * 100)).padStart(3);
          const elapsed = Math.max(
            0,
            Math.floor((Date.now() - params.startTime) / 1000),
          );
          return `${payload.label} ${bar} ${pct}% ${elapsed}s`;
        },
        hideCursor: true,
        clearOnComplete: true,
        stream: process.stderr,
        barsize: BARSIZE,
      },
      Presets.shades_classic,
    );
    mainBar = multibar.create(testCases.length, 0, {
      label: `🆚 Comparing ${testCases.length} contestants sequentially`,
    });
    caseBars = testCases.map((_tc, i) =>
      multibar!.create(3, 0, { label: `    🧐 Picking a winner (#${i + 1})` }),
    );
  }

  const originalShowIndicator = metric.showIndicator;
  metric.showIndicator = false;
  const winners: string[] = [];
  const results: ArenaCaseResult[] = [];
  let totalCost = 0;
  let hasCost = false;
  const startTime = Date.now();
  try {
    for (let i = 0; i < testCases.length; i++) {
      const testCase = testCases[i];
      const caseBar = caseBars[i];
      const caseStart = Date.now();
      let winner: string | null = null;
      try {
        winner = await metric.measure(testCase, {
          onStep: () => caseBar?.increment(),
        });
      } catch (e) {
        if (e instanceof MissingTestCaseParamsError) {
          if (!errorCfg.skipOnMissingParams) {
            if (errorCfg.ignoreErrors) metric.error = (e as Error).message;
            else throw e;
          }
        } else if (errorCfg.ignoreErrors) {
          metric.error = (e as Error).message;
        } else {
          throw e;
        }
      }
      caseBar?.update(3);
      mainBar?.increment();

      if (winner) winners.push(winner);
      if (metric.evaluationCost != null) {
        totalCost += metric.evaluationCost;
        hasCost = true;
      }
      results.push({
        testCase,
        index: i,
        winner,
        reason: metric.reason,
        error: metric.error,
        evaluationModel: metric.evaluationModel,
        evaluationCost: metric.evaluationCost,
        verboseLogs: metric.verboseLogs,
        runDuration: (Date.now() - caseStart) / 1000,
      });
    }
  } finally {
    multibar?.stop();
    metric.showIndicator = originalShowIndicator;
  }
  const runDuration = (Date.now() - startTime) / 1000;

  const counts: Record<string, number> = {};
  for (const w of winners) counts[w] = (counts[w] ?? 0) + 1;

  if (display.printResults) {
    printArenaCompleted(counts, runDuration, winners.length, hasCost ? totalCost : 0);
  }

  // Post to Confident AI as an experiment (no-op unless logged in).
  const { link } = await postExperiment(
    results,
    metric.name,
    options.name ?? "compare()",
  );
  if (display.printResults && !link) {
    printArenaFooter();
  }

  return counts;
}

function printArenaCompleted(
  counts: Record<string, number>,
  runDuration: number,
  total: number,
  tokenCost: number,
): void {
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const breakdown = sorted.length
    ? sorted
        .map(([name, wins]) => `    » ${GREEN}${BOLD}${name}${RESET}: ${wins} wins`)
        .join("\n")
    : "No winners";
  const cost = tokenCost ? `${tokenCost} USD` : "None";
  console.log(
    `\n🎉 Arena completed! (time taken: ${runDuration.toFixed(2)}s | token cost: ${cost})\n` +
      `🏆 Results (${total} total test cases):\n${breakdown}\n`,
  );
}

function printArenaFooter(): void {
  console.log(
    `\n${SEP}\n\n` +
      `» Want to share experiments with your team? ❤️ 🏟️\n` +
      `  » Set ${BOLD}CONFIDENT_API_KEY${RESET} to save arena results on ${PURPLE}Confident AI${RESET}.\n`,
  );
}
