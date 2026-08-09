import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  InspectLoadError,
  NoTracesError,
  findLatestTestRun,
  loadTestRun,
  resolveInspectTarget,
  summarizeTestRun,
} from "@/inspect/loader";
import { wrapText, durationMs, metricTally } from "@/inspect/ui/styling";

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-inspect-"));
}

function writeJson(file: string, data: unknown): string {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(data), "utf-8");
  return file;
}

const span = (
  uuid: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({
  uuid,
  name: uuid,
  status: "SUCCESS",
  type: "base",
  traceUuid: "t1",
  startTime: "2026-01-01T00:00:00.000Z",
  endTime: "2026-01-01T00:00:01.000Z",
  ...overrides,
});

const trace = (spans: Record<string, unknown>[]) => ({
  uuid: "t1",
  status: "SUCCESS",
  startTime: "2026-01-01T00:00:00.000Z",
  endTime: "2026-01-01T00:00:05.000Z",
  environment: "development",
  baseSpans: spans.filter((s) => s.type === "base"),
  agentSpans: spans.filter((s) => s.type === "agent"),
  llmSpans: spans.filter((s) => s.type === "llm"),
  retrieverSpans: [],
  toolSpans: [],
});

describe("inspect loader", () => {
  it("rebuilds the span tree from flat buckets", () => {
    const file = writeJson(path.join(tempDir(), "run.json"), {
      cases: [
        {
          entry: {
            name: "case one",
            success: true,
            trace: trace([
              span("root", { type: "agent" }),
              span("child-a", { parentUuid: "root" }),
              span("grandchild", { parentUuid: "child-a", type: "llm" }),
            ]),
          },
        },
      ],
    });

    const traces = loadTestRun(file);
    expect(traces).toHaveLength(1);
    expect(traces[0]!.caseName).toBe("case one");
    expect(traces[0]!.rootSpans.map((s) => s.uuid)).toEqual(["root"]);

    const root = traces[0]!.rootSpans[0]!;
    expect(root.children.map((s) => s.uuid)).toEqual(["child-a"]);
    expect(root.children[0]!.children.map((s) => s.uuid)).toEqual([
      "grandchild",
    ]);
  });

  it("orders siblings by start time", () => {
    const file = writeJson(path.join(tempDir(), "run.json"), {
      cases: [
        {
          entry: {
            trace: trace([
              span("root", { type: "agent" }),
              span("late", {
                parentUuid: "root",
                startTime: "2026-01-01T00:00:03.000Z",
              }),
              span("early", {
                parentUuid: "root",
                startTime: "2026-01-01T00:00:01.000Z",
              }),
            ]),
          },
        },
      ],
    });

    const root = loadTestRun(file)[0]!.rootSpans[0]!;
    expect(root.children.map((s) => s.uuid)).toEqual(["early", "late"]);
  });

  it("treats a span whose parent is missing as a root", () => {
    const file = writeJson(path.join(tempDir(), "run.json"), {
      cases: [
        {
          entry: {
            trace: trace([span("orphan", { parentUuid: "does-not-exist" })]),
          },
        },
      ],
    });

    expect(loadTestRun(file)[0]!.rootSpans.map((s) => s.uuid)).toEqual([
      "orphan",
    ]);
  });

  it("reads the Python layout, where the trace hangs off testCases", () => {
    const file = writeJson(path.join(tempDir(), "run.json"), {
      testCases: [{ name: "py case", trace: trace([span("root")]) }],
    });

    const traces = loadTestRun(file);
    expect(traces[0]!.caseName).toBe("py case");
    expect(traces[0]!.rootSpans).toHaveLength(1);
  });

  it("rejects a run that has no traces", () => {
    const file = writeJson(path.join(tempDir(), "run.json"), {
      cases: [{ entry: { name: "untraced", success: true } }],
    });
    expect(() => loadTestRun(file)).toThrow(NoTracesError);
  });

  it("rejects malformed JSON", () => {
    const file = path.join(tempDir(), "bad.json");
    fs.writeFileSync(file, "{not json", "utf-8");
    expect(() => loadTestRun(file)).toThrow(InspectLoadError);
  });

  it("reads run level totals, tolerating absent fields", () => {
    const dir = tempDir();
    const full = writeJson(path.join(dir, "full.json"), {
      testPassed: 3,
      testFailed: 1,
      runDuration: 12.5,
      evaluationCost: 0.02,
    });
    expect(summarizeTestRun(full)).toEqual({
      testPassed: 3,
      testFailed: 1,
      runDuration: 12.5,
      evaluationCost: 0.02,
    });

    const sparse = writeJson(path.join(dir, "sparse.json"), { cases: [] });
    expect(summarizeTestRun(sparse)).toEqual({
      testPassed: undefined,
      testFailed: undefined,
      runDuration: undefined,
      evaluationCost: undefined,
    });

    expect(summarizeTestRun(path.join(dir, "missing.json"))).toBeNull();
  });
});

describe("inspect target resolution", () => {
  it("picks the most recently modified export in a folder", () => {
    const dir = tempDir();
    const older = writeJson(
      path.join(dir, "test_run_20260101_000000.json"),
      {},
    );
    const newer = writeJson(
      path.join(dir, "test_run_20250101_000000.json"),
      {},
    );
    fs.utimesSync(older, new Date(1), new Date(1));

    expect(findLatestTestRun(dir)).toBe(newer);
    expect(resolveInspectTarget(dir)).toBe(newer);
  });

  it("returns an explicit file untouched", () => {
    const file = writeJson(path.join(tempDir(), "custom.json"), {});
    expect(resolveInspectTarget(file)).toBe(file);
  });

  it("reports a missing path rather than falling back", () => {
    expect(() => resolveInspectTarget("/no/such/run.json")).toThrow(
      InspectLoadError,
    );
  });

  it("errors when a folder holds no exports", () => {
    expect(() => findLatestTestRun(tempDir())).toThrow(InspectLoadError);
  });
});

describe("detail pane text wrapping", () => {
  it("wraps on word boundaries", () => {
    expect(wrapText("the quick brown fox jumps", 10)).toEqual([
      "the quick",
      "brown fox",
      "jumps",
    ]);
  });

  it("keeps indentation so pretty printed JSON stays readable", () => {
    const lines = wrapText('  "content": "alpha beta gamma delta"', 20);
    expect(lines[0]).toBe('  "content": "alpha');
    for (const line of lines.slice(1)) expect(line.startsWith("  ")).toBe(true);
  });

  it("breaks tokens longer than the pane", () => {
    const lines = wrapText("x".repeat(25), 10);
    expect(lines).toEqual(["x".repeat(10), "x".repeat(10), "x".repeat(5)]);
  });

  it("preserves blank lines between paragraphs", () => {
    expect(wrapText("a\n\nb", 10)).toEqual(["a", "", "b"]);
  });
});

describe("inspect formatting helpers", () => {
  it("computes duration from the span timestamps", () => {
    expect(
      durationMs({
        startTime: "2026-01-01T00:00:00.000Z",
        endTime: "2026-01-01T00:00:02.500Z",
      }),
    ).toBe(2500);
    expect(durationMs({})).toBeUndefined();
    expect(
      durationMs({ startTime: "nonsense", endTime: "also" }),
    ).toBeUndefined();
  });

  it("tallies metrics, counting errored ones separately", () => {
    const tally = metricTally({
      rootSpans: [],
      metricsData: [
        { name: "a", success: true },
        { name: "b", success: false },
        { name: "c", success: true, error: "boom" },
      ],
    });
    expect(tally).toEqual({ passed: 1, failed: 1, errored: 1 });
  });
});
