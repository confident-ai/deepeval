// Reads persisted test runs into nested trace trees. On disk a trace is a
// `TraceApi`: five flat span buckets whose members point at each other through
// `parentUuid`.

import * as fs from "fs";
import * as path from "path";
import {
  HIDDEN_DIR,
  LATEST_TEST_RUN_FILE,
  DEEPEVAL_RESULTS_FOLDER,
} from "@/constants";
import type { BaseApiSpan } from "@/tracing/api";
import type { InspectSpan, InspectTrace, RunSummary } from "@/inspect/model";

export class InspectLoadError extends Error {}
export class NoTracesError extends InspectLoadError {}

const SPAN_BUCKETS = [
  "baseSpans",
  "agentSpans",
  "llmSpans",
  "retrieverSpans",
  "toolSpans",
] as const;

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Ranked by mtime, so a run copied in under a stale name still sorts first. */
export function findLatestTestRun(folder: string): string {
  if (!fs.existsSync(folder) || !fs.statSync(folder).isDirectory()) {
    throw new InspectLoadError(`Results folder not found: ${folder}`);
  }

  const candidates = fs
    .readdirSync(folder)
    .filter((f) => f.startsWith("test_run_") && f.endsWith(".json"))
    .map((f) => path.join(folder, f))
    .map((f) => ({ file: f, mtime: fs.statSync(f).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime);

  if (candidates.length === 0) {
    throw new InspectLoadError(`No test_run_*.json files found in ${folder}.`);
  }
  return candidates[0]!.file;
}

export function resolveInspectTarget(
  target?: string,
  folderOption?: string,
): string {
  if (target) {
    if (!fs.existsSync(target)) {
      throw new InspectLoadError(`No such file or directory: ${target}`);
    }
    return fs.statSync(target).isDirectory()
      ? findLatestTestRun(target)
      : target;
  }

  const folder = folderOption || process.env[DEEPEVAL_RESULTS_FOLDER];
  if (folder && folder.trim() !== "") return findLatestTestRun(folder.trim());

  const rolling = path.join(process.cwd(), HIDDEN_DIR, LATEST_TEST_RUN_FILE);
  if (fs.existsSync(rolling)) return rolling;

  throw new InspectLoadError(
    `No test run found. Expected ${path.join(HIDDEN_DIR, LATEST_TEST_RUN_FILE)} ` +
      "in the current directory. Run an evaluation first, pass a path " +
      "explicitly, or set DEEPEVAL_RESULTS_FOLDER.",
  );
}

function readJson(file: string): Json {
  let raw: string;
  try {
    raw = fs.readFileSync(file, "utf-8");
  } catch (e) {
    throw new InspectLoadError(
      `Failed to read test run from ${file}: ${(e as Error).message}`,
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    throw new InspectLoadError(
      `${file} is not valid JSON: ${(e as Error).message}`,
    );
  }

  if (!isRecord(parsed)) {
    throw new InspectLoadError(
      `Expected the top-level JSON in ${file} to be an object.`,
    );
  }
  return parsed;
}

/** TypeScript nests the trace under `cases[].entry`; Python puts it on `testCases[]`. */
function* iterateCases(
  data: Json,
): Generator<{ trace: Json; name?: string; passed?: boolean }> {
  for (const group of [data.cases, data.testCases]) {
    if (!Array.isArray(group)) continue;
    for (const item of group) {
      if (!isRecord(item)) continue;
      const holder = isRecord(item.entry) ? item.entry : item;
      if (isRecord(holder.trace)) {
        yield {
          trace: holder.trace,
          name: typeof holder.name === "string" ? holder.name : undefined,
          passed:
            typeof holder.success === "boolean" ? holder.success : undefined,
        };
      }
    }
  }
}

/** Also the compile-time check that `InspectSpan` matches what the tracer writes. */
function toInspectSpan(span: BaseApiSpan): InspectSpan {
  return { ...span, children: [] };
}

function buildSpanTree(spans: Json[]): Map<string, InspectSpan> {
  const byUuid = new Map<string, InspectSpan>();

  for (const raw of spans) {
    const uuid = typeof raw.uuid === "string" ? raw.uuid : undefined;
    if (!uuid || byUuid.has(uuid)) continue;
    byUuid.set(uuid, toInspectSpan(raw as unknown as BaseApiSpan));
  }

  for (const span of byUuid.values()) {
    const parent = span.parentUuid ? byUuid.get(span.parentUuid) : undefined;
    if (parent) parent.children.push(span);
  }

  const byStartTime = (a: InspectSpan, b: InspectSpan) =>
    (a.startTime ?? "").localeCompare(b.startTime ?? "");
  for (const span of byUuid.values()) span.children.sort(byStartTime);

  return byUuid;
}

function parseTrace(raw: Json): InspectTrace {
  const rest: Json = { ...raw };
  const spans: Json[] = [];

  for (const bucket of SPAN_BUCKETS) {
    const value = rest[bucket];
    delete rest[bucket];
    if (!Array.isArray(value)) continue;
    for (const span of value) if (isRecord(span)) spans.push(span);
  }

  const byUuid = buildSpanTree(spans);
  const roots = [...byUuid.values()]
    .filter((s) => !s.parentUuid || !byUuid.has(s.parentUuid))
    .sort((a, b) => (a.startTime ?? "").localeCompare(b.startTime ?? ""));

  return { ...(rest as unknown as InspectTrace), rootSpans: roots };
}

export function loadTestRun(file: string): InspectTrace[] {
  const data = readJson(file);
  const traces: InspectTrace[] = [];

  for (const { trace, name, passed } of iterateCases(data)) {
    traces.push({ ...parseTrace(trace), caseName: name, casePassed: passed });
  }

  if (traces.length === 0) {
    throw new NoTracesError(
      `${file} contains no traces. \`deepeval inspect\` shows trace trees, so ` +
        "runs recorded without tracing have nothing to display.",
    );
  }
  return traces;
}

export function summarizeTestRun(file: string): RunSummary | null {
  try {
    const data = readJson(file);
    const num = (key: string): number | undefined =>
      typeof data[key] === "number" ? (data[key] as number) : undefined;
    return {
      testPassed: num("testPassed"),
      testFailed: num("testFailed"),
      runDuration: num("runDuration"),
      evaluationCost: num("evaluationCost"),
    };
  } catch {
    return null;
  }
}
