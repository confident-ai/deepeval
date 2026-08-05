import * as fs from "fs";
import * as path from "path";
import { DEEPEVAL_RESULTS_DIR } from "../../constants";
import {
  buildTestCaseEntry,
  postPersistedTestRun,
  type PersistedCase,
} from "../confident";
import { printResultsTable } from "../console-report";
import { EvaluatedCase, MetricData, TestResult } from "../types";

export function getResultsDir(): string | null {
  return process.env[DEEPEVAL_RESULTS_DIR] || null;
}

let workerCaseCount = 0;
function workerFilePath(dir: string): string {
  return path.join(dir, `worker-${process.pid}.jsonl`);
}

export function persistCase(evaluatedCase: EvaluatedCase): void {
  const dir = getResultsDir();
  if (!dir) return;
  const persisted = buildTestCaseEntry(evaluatedCase, workerCaseCount++);
  fs.appendFileSync(workerFilePath(dir), JSON.stringify(persisted) + "\n");
}

export function readPersistedCases(dir: string): PersistedCase[] {
  if (!fs.existsSync(dir)) return [];
  const cases: PersistedCase[] = [];
  for (const file of fs.readdirSync(dir)) {
    if (!file.endsWith(".jsonl")) continue;
    const content = fs.readFileSync(path.join(dir, file), "utf-8");
    for (const line of content.split("\n")) {
      if (line.trim()) cases.push(JSON.parse(line) as PersistedCase);
    }
  }
  return cases;
}

function persistedToTestResult(
  { conversational, entry, metricsData }: PersistedCase,
  index: number,
): TestResult {
  return {
    name: (entry.name as string) ?? `test_case_${index}`,
    success: entry.success as boolean,
    metricsData,
    conversational,
    index,
    input: entry.input as string | undefined,
    actualOutput: entry.actualOutput as string | undefined,
    expectedOutput: entry.expectedOutput as string | undefined,
    context: entry.context as string[] | undefined,
    retrievalContext: entry.retrievalContext as string[] | undefined,
    turns: entry.turns as TestResult["turns"],
  };
}

export interface WrapUpOptions {
  runDuration?: number;
  official?: boolean;
  identifier?: string;
  printResults?: boolean;
}

export async function wrapUpTestRun(
  dir: string,
  options: WrapUpOptions = {},
): Promise<{ link: string | null; testRunId: string | null }> {
  const cases = readPersistedCases(dir);
  if (cases.length === 0) return { link: null, testRunId: null };

  if (options.printResults ?? true) {
    printResultsTable(cases.map(persistedToTestResult));
  }
  return postPersistedTestRun(
    cases,
    options.runDuration ?? 0,
    options.official ?? false,
    false,
    options.identifier,
  );
}

export function _resetWorkerCaseCount(): void {
  workerCaseCount = 0;
}

export type { MetricData };
