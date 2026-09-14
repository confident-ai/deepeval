import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  MIN_NODE_MAJOR_FOR_SQLITE,
  SqliteUnsupportedError,
  assertSqliteSupported,
  isSqliteSupported,
  nodeMajor,
  normalizeLocalStoreMode,
  resolveLocalStoreMode,
} from "@/sqlite-store/mode";
import {
  DB_FILENAME,
  MIGRATIONS,
  SCHEMA,
  SCHEMA_VERSION,
  TestRunNotFoundError,
  applyMigrations,
  connect,
  formatSource,
  latestRunId,
  listTestRuns,
  loadTestRunPayload,
  parseSource,
  resolveDbPath,
  resolveIncludeRowJson,
  setConfidentTestRunId,
  writeTestRun,
} from "@/sqlite-store/store";
import { exportTestRun, type LocalTestRun } from "@/evaluate/test-run/local";
import {
  loadTestRun,
  resolveInspectTarget,
  runIdFromSource,
  summarizeTestRun,
} from "@/inspect/loader";
import { listStoredRuns } from "@/inspect";
import { settingsSchema } from "@/config/schema";
import {
  DEEPEVAL_LOCAL_STORE,
  DEEPEVAL_RESULTS_FOLDER,
  DEEPEVAL_SQLITE_INCLUDE_ROW_JSON,
} from "@/constants";

// `node:sqlite` exists (unflagged) from Node 22.13 / 23.4, which is enough to
// exercise the store itself; the Node 24 gate is tested with injected versions.
const hasNodeSqlite = (() => {
  try {
    const mod = (
      process as unknown as { getBuiltinModule?: (id: string) => unknown }
    ).getBuiltinModule?.("node:sqlite") as { DatabaseSync?: unknown };
    return typeof mod?.DatabaseSync === "function";
  } catch {
    return false;
  }
})();
const describeWithSqlite = hasNodeSqlite ? describe : describe.skip;

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-sqlite-"));
}

const metric = (name: string, score: number, success = true) => ({
  name,
  threshold: 0.5,
  success,
  score,
  reason: `${name} reason`,
  strictMode: false,
  flaky: false,
  evaluationModel: "gpt-4o-mini",
  evaluationCost: 0.001,
});

const trace = () => ({
  uuid: "trace-1",
  name: "my-agent",
  status: "SUCCESS",
  startTime: "2026-01-01T00:00:00.000Z",
  endTime: "2026-01-01T00:00:03.000Z",
  threadId: "thread-1",
  userId: "user-1",
  environment: "testing",
  input: "hi",
  output: "done",
  tags: ["a", "b"],
  metadata: { k: "v" },
  metricsData: [metric("TraceMetric", 0.7)],
  agentSpans: [
    {
      uuid: "span-root",
      name: "agent",
      type: "agent",
      status: "SUCCESS",
      startTime: "2026-01-01T00:00:00.000Z",
      endTime: "2026-01-01T00:00:03.000Z",
      input: { q: "hi" },
      output: "done",
      metricsData: [metric("TaskCompletion", 0.9)],
    },
  ],
  llmSpans: [
    {
      uuid: "span-llm",
      parentUuid: "span-root",
      name: "openai.chat",
      type: "llm",
      status: "SUCCESS",
      startTime: "2026-01-01T00:00:01.000Z",
      endTime: "2026-01-01T00:00:02.000Z",
      model: "gpt-4o-mini",
      provider: "openai",
      inputTokenCount: 12,
      outputTokenCount: 34,
    },
  ],
  toolSpans: [
    {
      uuid: "span-tool",
      parentUuid: "span-root",
      name: "search",
      type: "tool",
      status: "ERRORED",
      startTime: "2026-01-01T00:00:01.000Z",
      endTime: "2026-01-01T00:00:01.000Z",
      error: "boom",
    },
  ],
  retrieverSpans: [
    {
      uuid: "span-ret",
      parentUuid: "span-root",
      name: "retriever",
      type: "retriever",
      status: "SUCCESS",
      startTime: "2026-01-01T00:00:01.000Z",
      endTime: "2026-01-01T00:00:01.000Z",
    },
  ],
  baseSpans: [
    {
      uuid: "span-base",
      parentUuid: "span-llm",
      name: "helper",
      type: "base",
      status: "SUCCESS",
      startTime: "2026-01-01T00:00:01.000Z",
      endTime: "2026-01-01T00:00:01.000Z",
    },
  ],
});

function makeRun(
  overrides: Partial<LocalTestRun> = {},
  withCases = true,
): LocalTestRun {
  const cases: LocalTestRun["cases"] = withCases
    ? [
        {
          conversational: false,
          datasetAlias: "my-dataset",
          metricsData: [],
          entry: {
            name: "case-0",
            input: "What is 1+1?",
            actualOutput: "2",
            expectedOutput: "2",
            order: 0,
            success: true,
            runDuration: 1.5,
            evaluationCost: 0.01,
            metricsData: [
              metric("AnswerRelevancy", 0.95),
              metric("Faithfulness", 0.3, false),
            ],
            trace: trace(),
          },
        },
        {
          conversational: true,
          metricsData: [],
          entry: {
            name: "conv-0",
            order: 1,
            success: true,
            scenario: "Book a flight",
            expectedOutcome: "Flight booked",
            metricsData: [metric("TurnRelevancy", 0.8)],
            turns: [
              { role: "user", content: "book me a flight", order: 0 },
              { role: "assistant", content: "sure", order: 1 },
            ],
          },
        },
      ]
    : [];
  return {
    link: null,
    savedAt: "2026-09-14T06:00:00.000Z",
    runDuration: 3.25,
    official: false,
    identifier: "base",
    hyperparameters: { model: "x", temperature: 0.2 } as never,
    testPassed: 2,
    testFailed: 0,
    cases,
    ...overrides,
  };
}

function query(dbPath: string, sql: string, ...params: unknown[]): unknown[] {
  const db = connect(dbPath);
  try {
    return db.prepare(sql).all(...(params as never[]));
  } finally {
    db.close();
  }
}

describe("sqlite-store mode / Node gate", () => {
  const savedMode = process.env[DEEPEVAL_LOCAL_STORE];
  afterEach(() => {
    if (savedMode === undefined) delete process.env[DEEPEVAL_LOCAL_STORE];
    else process.env[DEEPEVAL_LOCAL_STORE] = savedMode;
    jest.restoreAllMocks();
  });

  test("nodeMajor parses versions", () => {
    expect(nodeMajor("24.1.0")).toBe(24);
    expect(nodeMajor("v22.13.1".replace(/^v/, ""))).toBe(22);
    expect(nodeMajor("garbage")).toBe(0);
  });

  test("supported from Node 24 onwards", () => {
    expect(isSqliteSupported("23.10.0")).toBe(false);
    expect(isSqliteSupported("24.0.0")).toBe(true);
    expect(isSqliteSupported("25.3.1")).toBe(true);
    expect(MIN_NODE_MAJOR_FOR_SQLITE).toBe(24);
  });

  test("assertSqliteSupported names the version and the fix", () => {
    expect(() => assertSqliteSupported("24.0.0")).not.toThrow();
    expect(() => assertSqliteSupported("20.19.0")).toThrow(
      SqliteUnsupportedError,
    );
    expect(() => assertSqliteSupported("20.19.0")).toThrow(
      /DEEPEVAL_LOCAL_STORE=sqlite requires Node\.js 24.*v20\.19\.0.*set it to json/s,
    );
  });

  test("normalizeLocalStoreMode accepts aliases and rejects junk", () => {
    expect(normalizeLocalStoreMode(undefined)).toBeUndefined();
    expect(normalizeLocalStoreMode("")).toBe("json");
    expect(normalizeLocalStoreMode("JSON")).toBe("json");
    for (const v of ["sqlite", "SQLite", " sqlite3 ", "db"]) {
      expect(normalizeLocalStoreMode(v)).toBe("sqlite");
    }
    expect(normalizeLocalStoreMode("postgres")).toBeUndefined();
  });

  test("resolveLocalStoreMode defaults to json and warns on junk", () => {
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    expect(resolveLocalStoreMode({})).toBe("json");
    expect(resolveLocalStoreMode({ [DEEPEVAL_LOCAL_STORE]: "json" })).toBe(
      "json",
    );
    expect(warn).not.toHaveBeenCalled();
    expect(resolveLocalStoreMode({ [DEEPEVAL_LOCAL_STORE]: "postgres" })).toBe(
      "json",
    );
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]![0]).toMatch(/DEEPEVAL_LOCAL_STORE/);
  });

  test("resolveLocalStoreMode enforces the Node gate for sqlite", () => {
    const env = { [DEEPEVAL_LOCAL_STORE]: "sqlite" };
    if (isSqliteSupported()) {
      expect(resolveLocalStoreMode(env)).toBe("sqlite");
    } else {
      expect(() => resolveLocalStoreMode(env)).toThrow(SqliteUnsupportedError);
    }
  });

  test("settings schema rejects sqlite on unsupported Node, junk always", () => {
    const junk = settingsSchema.safeParse({
      [DEEPEVAL_LOCAL_STORE]: "postgres",
    });
    expect(junk.success).toBe(false);

    const sqlite = settingsSchema.safeParse({
      [DEEPEVAL_LOCAL_STORE]: "sqlite",
    });
    if (isSqliteSupported()) {
      expect(sqlite.success).toBe(true);
      if (sqlite.success)
        expect(sqlite.data.DEEPEVAL_LOCAL_STORE).toBe("sqlite");
    } else {
      expect(sqlite.success).toBe(false);
      if (!sqlite.success) {
        expect(sqlite.error.issues[0]!.message).toMatch(/Node\.js 24/);
      }
    }

    const json = settingsSchema.safeParse({ [DEEPEVAL_LOCAL_STORE]: "JSON" });
    expect(json.success).toBe(true);
    if (json.success) expect(json.data.DEEPEVAL_LOCAL_STORE).toBe("json");
  });
});

describe("sqlite-store paths / source specs", () => {
  test("resolveDbPath honours the results folder, else the cache dir", () => {
    expect(resolveDbPath("/tmp/evals")).toBe(
      path.join("/tmp/evals", DB_FILENAME),
    );
    expect(resolveDbPath(undefined)).toBe(path.join(".deepeval", DB_FILENAME));
    expect(resolveDbPath("   ")).toBe(path.join(".deepeval", DB_FILENAME));
  });

  test("parseSource / formatSource round-trip", () => {
    const db = "/x/deepeval.db";
    expect(parseSource(db)).toEqual({ dbPath: db, runId: null });
    expect(parseSource(`${db}#7`)).toEqual({ dbPath: db, runId: 7 });
    expect(parseSource(`${db}#`)).toEqual({ dbPath: db, runId: null });
    expect(formatSource(db, 7)).toBe(`${db}#7`);
    expect(parseSource("/x/test_run_1.json")).toBeNull();
    expect(parseSource("experiments")).toBeNull();
  });

  test("runIdFromSource", () => {
    expect(runIdFromSource("/x/deepeval.db#3")).toBe("deepeval.db#3");
    expect(runIdFromSource("/x/deepeval.db")).toBe("deepeval.db");
    expect(runIdFromSource("/x/test_run_x.json")).toBe("test_run_x");
  });
});

describeWithSqlite("sqlite-store write / read", () => {
  let dir: string;
  let db: string;
  beforeEach(() => {
    dir = tempDir();
    db = path.join(dir, DB_FILENAME);
  });
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

  test("creates the schema and stamps user_version", () => {
    const nested = path.join(dir, "nested", DB_FILENAME);
    connect(nested).close();
    expect(fs.existsSync(nested)).toBe(true);
    const tables = query(
      nested,
      "SELECT name FROM sqlite_master WHERE type='table'",
    ) as Array<{ name: string }>;
    const names = new Set(tables.map((t) => t.name));
    for (const t of [
      "test_runs",
      "test_cases",
      "traces",
      "spans",
      "metric_data",
    ]) {
      expect(names.has(t)).toBe(true);
    }
    const [{ user_version }] = query(nested, "PRAGMA user_version") as Array<{
      user_version: number;
    }>;
    expect(Number(user_version)).toBe(SCHEMA_VERSION);
  });

  test("rejects a newer schema", () => {
    const c = connect(db);
    c.exec(`PRAGMA user_version = ${SCHEMA_VERSION + 1}`);
    c.close();
    expect(() => connect(db)).toThrow(/newer than this deepeval/);
  });

  test("payload round-trips exactly", () => {
    const run = makeRun();
    const runId = writeTestRun(run, db);
    expect(runId).toBe(1);
    const loaded = loadTestRunPayload(db, runId);
    expect(loaded.runId).toBe(1);
    expect(loaded.payload).toEqual(JSON.parse(JSON.stringify(run)));
    expect(loadTestRunPayload(db).runId).toBe(1);
  });

  const rowJsonCounts = () =>
    ["test_cases", "traces", "spans"].map((t) => {
      const [row] = query(
        db,
        `SELECT count(payload_json) AS n FROM ${t}`,
      ) as Array<{
        n: number | bigint;
      }>;
      return Number(row!.n);
    });

  describe("DEEPEVAL_SQLITE_INCLUDE_ROW_JSON", () => {
    const saved = process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON];
    afterEach(() => {
      if (saved === undefined)
        delete process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON];
      else process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON] = saved;
    });

    test("row JSON is off by default", () => {
      delete process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON];
      expect(resolveIncludeRowJson()).toBe(false);
      const runId = writeTestRun(makeRun(), db);
      expect(rowJsonCounts()).toEqual([0, 0, 0]);
      // Run row always keeps its payload, so reload still works.
      expect(loadTestRunPayload(db, runId).payload).toEqual(
        JSON.parse(JSON.stringify(makeRun())),
      );
      const [{ n }] = query(db, "SELECT count(*) AS n FROM spans") as Array<{
        n: number;
      }>;
      expect(Number(n)).toBeGreaterThan(0);
    });

    test.each(["1", "true", "YES", " on "])("env %j turns them on", (raw) => {
      process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON] = raw;
      expect(resolveIncludeRowJson()).toBe(true);
      writeTestRun(makeRun(), db);
      const counts = rowJsonCounts();
      expect(counts.every((n) => n > 0)).toBe(true);
      const [row] = query(
        db,
        "SELECT uuid, payload_json FROM traces LIMIT 1",
      ) as Array<{
        uuid: string;
        payload_json: string;
      }>;
      expect(JSON.parse(row!.payload_json).uuid).toBe(row!.uuid);
    });

    test.each(["0", "false", "no", ""])("env %j keeps them off", (raw) => {
      process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON] = raw;
      writeTestRun(makeRun(), db);
      expect(rowJsonCounts()).toEqual([0, 0, 0]);
    });

    test("includeRowJson option overrides the env", () => {
      process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON] = "1";
      writeTestRun(makeRun(), db, { includeRowJson: false });
      expect(rowJsonCounts()).toEqual([0, 0, 0]);

      delete process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON];
      writeTestRun(makeRun(), db, { includeRowJson: true });
      expect(rowJsonCounts().every((n) => n > 0)).toBe(true);
    });
  });

  test("confident_test_run_id: NULL when not logged in, filled from the payload when posted", () => {
    const local = writeTestRun(makeRun(), db);
    const posted = writeTestRun(
      makeRun({
        testRunId: "cai_abc123",
        link: "https://app.confident-ai.com/x",
      }),
      db,
    );
    const rows = query(
      db,
      "SELECT id, confident_test_run_id FROM test_runs ORDER BY id",
    ) as Array<{ id: number | bigint; confident_test_run_id: string | null }>;
    expect(rows.map((r) => [Number(r.id), r.confident_test_run_id])).toEqual([
      [local, null],
      [posted, "cai_abc123"],
    ]);
    const summaries = listTestRuns(db);
    expect(summaries.map((r) => r.confident_test_run_id)).toEqual([
      "cai_abc123",
      null,
    ]);
  });

  test("setConfidentTestRunId stamps a stored run", () => {
    const runId = writeTestRun(makeRun(), db);
    setConfidentTestRunId(db, runId, "cai_later");
    const [row] = query(
      db,
      "SELECT confident_test_run_id FROM test_runs",
    ) as Array<{
      confident_test_run_id: string | null;
    }>;
    expect(row!.confident_test_run_id).toBe("cai_later");
    // Payload untouched: the id lives in the column only.
    expect(loadTestRunPayload(db, runId).payload).toEqual(
      JSON.parse(JSON.stringify(makeRun())),
    );
    expect(() => setConfidentTestRunId(db, 999, "x")).toThrow(
      TestRunNotFoundError,
    );
  });

  test("normalized test_runs columns", () => {
    writeTestRun(makeRun(), db);
    const [row] = query(
      db,
      `SELECT created_at, identifier, dataset_alias, test_passed, test_failed,
              run_duration, evaluation_cost, official, hyperparameters_json
       FROM test_runs`,
    ) as Array<Record<string, unknown>>;
    expect(row!.created_at).toBe("2026-09-14T06:00:00.000Z");
    expect(row!.identifier).toBe("base");
    expect(row!.dataset_alias).toBe("my-dataset");
    expect(Number(row!.test_passed)).toBe(2);
    expect(Number(row!.test_failed)).toBe(0);
    expect(row!.run_duration).toBeCloseTo(3.25);
    expect(row!.evaluation_cost).toBeCloseTo(0.01);
    expect(Number(row!.official)).toBe(0);
    expect(JSON.parse(row!.hyperparameters_json as string)).toEqual({
      model: "x",
      temperature: 0.2,
    });
  });

  test("test_cases rows for single-turn and multi-turn kinds", () => {
    writeTestRun(makeRun(), db);
    const rows = query(
      db,
      `SELECT kind, "order", name, input, actual_output, expected_output, success
       FROM test_cases ORDER BY "order"`,
    ) as Array<Record<string, unknown>>;
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      kind: "single-turn",
      name: "case-0",
      input: "What is 1+1?",
      actual_output: "2",
      expected_output: "2",
    });
    expect(Number(rows[0]!.success)).toBe(1);
    expect(rows[1]).toMatchObject({
      kind: "multi-turn",
      name: "conv-0",
      input: "Book a flight",
      actual_output: null,
      expected_output: "Flight booked",
    });
  });

  test("traces and spans are flattened with parent links", () => {
    writeTestRun(makeRun(), db);
    const [t] = query(
      db,
      "SELECT uuid, name, thread_id, user_id, environment, tags_json, test_case_id FROM traces",
    ) as Array<Record<string, unknown>>;
    expect(t).toMatchObject({
      uuid: "trace-1",
      name: "my-agent",
      thread_id: "thread-1",
      user_id: "user-1",
      environment: "testing",
    });
    expect(JSON.parse(t!.tags_json as string)).toEqual(["a", "b"]);
    const [llmCase] = query(
      db,
      "SELECT id FROM test_cases WHERE kind = 'single-turn'",
    ) as Array<{ id: number }>;
    expect(Number(t!.test_case_id)).toBe(Number(llmCase!.id));

    const spans = query(
      db,
      `SELECT uuid, parent_uuid, type, model, provider, input_token_count,
              output_token_count, error, status FROM spans`,
    ) as Array<Record<string, unknown>>;
    const byUuid = new Map(spans.map((s) => [s.uuid as string, s]));
    expect([...byUuid.keys()].sort()).toEqual(
      ["span-base", "span-llm", "span-ret", "span-root", "span-tool"].sort(),
    );
    expect(byUuid.get("span-root")).toMatchObject({
      parent_uuid: null,
      type: "agent",
    });
    expect(byUuid.get("span-llm")).toMatchObject({
      parent_uuid: "span-root",
      type: "llm",
      model: "gpt-4o-mini",
      provider: "openai",
    });
    expect(Number(byUuid.get("span-llm")!.input_token_count)).toBe(12);
    expect(Number(byUuid.get("span-llm")!.output_token_count)).toBe(34);
    expect(byUuid.get("span-tool")).toMatchObject({
      error: "boom",
      status: "ERRORED",
    });
    expect(byUuid.get("span-base")!.parent_uuid).toBe("span-llm");
  });

  test("metrics rows for every owner type", () => {
    const runId = writeTestRun(makeRun(), db);
    const rows = query(
      db,
      `SELECT owner_type, owner_id, name, score, success, reason, evaluation_model,
              evaluation_cost FROM metric_data WHERE test_run_id = ?`,
      runId,
    ) as Array<Record<string, unknown>>;
    const keys = rows.map((r) => `${r.owner_type}:${r.name}`).sort();
    expect(keys).toEqual(
      [
        "test_case:AnswerRelevancy",
        "test_case:Faithfulness",
        "test_case:TurnRelevancy",
        "trace:TraceMetric",
        "span:TaskCompletion",
      ].sort(),
    );
    const faith = rows.find((r) => r.name === "Faithfulness")!;
    expect(faith.score).toBeCloseTo(0.3);
    expect(Number(faith.success)).toBe(0);
    expect(faith.reason).toBe("Faithfulness reason");
    expect(faith.evaluation_model).toBe("gpt-4o-mini");
    expect(faith.evaluation_cost).toBeCloseTo(0.001);

    const spanMetric = rows.find((r) => r.owner_type === "span")!;
    const [span] = query(
      db,
      "SELECT uuid FROM spans WHERE id = ?",
      spanMetric.owner_id,
    ) as Array<{ uuid: string }>;
    expect(span!.uuid).toBe("span-root");
  });

  test("cross-run SQL aggregation", () => {
    for (let i = 0; i < 3; i++) writeTestRun(makeRun(), db);
    const rows = query(
      db,
      `SELECT r.id AS id, avg(m.score) AS avg FROM metric_data m
       JOIN test_runs r ON r.id = m.test_run_id
       WHERE m.name = 'AnswerRelevancy' GROUP BY r.id ORDER BY r.id`,
    ) as Array<{ id: number; avg: number }>;
    expect(rows.map((r) => Number(r.id))).toEqual([1, 2, 3]);
    for (const r of rows) expect(r.avg).toBeCloseTo(0.95);
  });

  test("list / latest / not-found", () => {
    expect(listTestRuns(db)).toEqual([]);
    expect(latestRunId(db)).toBeNull();
    expect(() => loadTestRunPayload(db)).toThrow(TestRunNotFoundError);

    for (const identifier of ["a", "b", "c"]) {
      writeTestRun(makeRun({ identifier }, false), db);
    }
    expect(listTestRuns(db).map((r) => r.identifier)).toEqual(["c", "b", "a"]);
    expect(listTestRuns(db, 1)[0]!.id).toBe(3);
    expect(latestRunId(db)).toBe(3);
    expect(loadTestRunPayload(db).payload.identifier).toBe("c");
    expect(() => loadTestRunPayload(db, 99)).toThrow(TestRunNotFoundError);
  });

  test("inspect loader: JSON and SQLite yield identical traces", () => {
    const run = makeRun();
    const jsonFile = path.join(dir, "test_run_20260914_060000.json");
    fs.writeFileSync(jsonFile, JSON.stringify(run), "utf-8");
    const runId = writeTestRun(run, db);

    const fromJson = loadTestRun(jsonFile);
    const fromDb = loadTestRun(formatSource(db, runId));
    const fromDbLatest = loadTestRun(db);
    expect(fromDb).toEqual(fromJson);
    expect(fromDbLatest).toEqual(fromJson);
    expect(fromDb[0]!.uuid).toBe("trace-1");
    expect(fromDb[0]!.rootSpans.map((s) => s.uuid)).toEqual(["span-root"]);

    expect(summarizeTestRun(formatSource(db, runId))).toEqual({
      testPassed: 2,
      testFailed: 0,
      runDuration: 3.25,
      evaluationCost: undefined,
    });
    expect(() => loadTestRun(`${db}#42`)).toThrow(/No test run with id 42/);
    expect(() => loadTestRun(path.join(dir, "nope.db"))).toThrow(
      /SQLite store not found/,
    );
  });

  test("listStoredRuns prints a table for a db and rejects json sources", () => {
    writeTestRun(makeRun({ identifier: "smoke" }, false), db);
    const log = jest.spyOn(console, "log").mockImplementation(() => {});
    const table = jest.spyOn(console, "table").mockImplementation(() => {});
    try {
      listStoredRuns({ target: db });
      expect(table).toHaveBeenCalledTimes(1);
      const rows = table.mock.calls[0]![0] as Array<Record<string, unknown>>;
      expect(rows).toHaveLength(1);
      expect(rows[0]).toMatchObject({ id: 1, identifier: "smoke", passed: 2 });
      expect(log.mock.calls.some(([m]) => String(m).includes("--run-id"))).toBe(
        true,
      );

      const jsonFile = path.join(dir, "test_run_20260914_060000.json");
      fs.writeFileSync(jsonFile, JSON.stringify(makeRun()), "utf-8");
      expect(() => listStoredRuns({ target: jsonFile })).toThrow(
        /--list only works with a SQLite store/,
      );
    } finally {
      log.mockRestore();
      table.mockRestore();
    }
  });

  describe("resolveInspectTarget", () => {
    const savedMode = process.env[DEEPEVAL_LOCAL_STORE];
    const savedFolder = process.env[DEEPEVAL_RESULTS_FOLDER];
    afterEach(() => {
      if (savedMode === undefined) delete process.env[DEEPEVAL_LOCAL_STORE];
      else process.env[DEEPEVAL_LOCAL_STORE] = savedMode;
      if (savedFolder === undefined)
        delete process.env[DEEPEVAL_RESULTS_FOLDER];
      else process.env[DEEPEVAL_RESULTS_FOLDER] = savedFolder;
    });

    test("db path, db#id, --run-id, and folders", () => {
      writeTestRun(makeRun(), db);
      writeTestRun(makeRun(), db);
      expect(resolveInspectTarget(db)).toBe(db);
      expect(resolveInspectTarget(`${db}#1`)).toBe(`${db}#1`);
      expect(resolveInspectTarget(db, undefined, { runId: 2 })).toBe(`${db}#2`);
      // Folder with only a DB (no JSON) resolves to the DB in json mode too.
      delete process.env[DEEPEVAL_LOCAL_STORE];
      expect(resolveInspectTarget(dir)).toBe(db);
      expect(resolveInspectTarget(dir, undefined, { runId: 1 })).toBe(
        `${db}#1`,
      );
      // Folder with both prefers JSON in json mode, DB in sqlite mode.
      const jsonFile = path.join(dir, "test_run_20260914_060000.json");
      fs.writeFileSync(jsonFile, JSON.stringify(makeRun()), "utf-8");
      expect(resolveInspectTarget(dir)).toBe(jsonFile);
      process.env[DEEPEVAL_LOCAL_STORE] = "sqlite";
      expect(resolveInspectTarget(dir)).toBe(db);
      expect(resolveInspectTarget(undefined, dir)).toBe(db);
      process.env[DEEPEVAL_RESULTS_FOLDER] = dir;
      expect(resolveInspectTarget()).toBe(db);
    });

    test("missing db is a load error", () => {
      expect(() => resolveInspectTarget(path.join(dir, "nope.db"))).toThrow(
        /SQLite store not found/,
      );
    });
  });

  describe("exportTestRun", () => {
    const savedMode = process.env[DEEPEVAL_LOCAL_STORE];
    const savedFolder = process.env[DEEPEVAL_RESULTS_FOLDER];
    afterEach(() => {
      if (savedMode === undefined) delete process.env[DEEPEVAL_LOCAL_STORE];
      else process.env[DEEPEVAL_LOCAL_STORE] = savedMode;
      if (savedFolder === undefined)
        delete process.env[DEEPEVAL_RESULTS_FOLDER];
      else process.env[DEEPEVAL_RESULTS_FOLDER] = savedFolder;
      jest.restoreAllMocks();
    });

    test("json mode writes test_run_*.json and no db", () => {
      process.env[DEEPEVAL_LOCAL_STORE] = "json";
      process.env[DEEPEVAL_RESULTS_FOLDER] = dir;
      const out = exportTestRun(makeRun({}, false));
      expect(out?.mode).toBe("json");
      expect(fs.existsSync(out!.path)).toBe(true);
      expect(path.basename(out!.path)).toMatch(/^test_run_\d{8}_\d{6}\.json$/);
      expect(fs.existsSync(db)).toBe(false);
    });

    test("json mode without a folder is a no-op", () => {
      process.env[DEEPEVAL_LOCAL_STORE] = "json";
      delete process.env[DEEPEVAL_RESULTS_FOLDER];
      expect(exportTestRun(makeRun({}, false))).toBeNull();
    });

    test("sqlite mode appends to deepeval.db and writes no json", () => {
      process.env[DEEPEVAL_LOCAL_STORE] = "sqlite";
      process.env[DEEPEVAL_RESULTS_FOLDER] = dir;
      if (!isSqliteSupported()) {
        // Below Node 24 the request itself is the error, not a storage hiccup.
        expect(() => exportTestRun(makeRun({}, false))).toThrow(
          SqliteUnsupportedError,
        );
        return;
      }
      const first = exportTestRun(makeRun({ identifier: "one" }, false));
      const second = exportTestRun(makeRun({ identifier: "two" }, false));
      expect(first).toEqual({ mode: "sqlite", path: db, runId: 1 });
      expect(second).toEqual({ mode: "sqlite", path: db, runId: 2 });
      expect(fs.readdirSync(dir).filter((f) => f.endsWith(".json"))).toEqual(
        [],
      );
      expect(listTestRuns(db).map((r) => r.identifier)).toEqual(["two", "one"]);
    });
  });
});

// --------------------------------------------------------------------------- //
// Schema parity with the shared `sqlite/schema.sql` at the repo root
// --------------------------------------------------------------------------- //

const SHARED_SCHEMA = path.resolve(__dirname, "../../../sqlite/schema.sql");

/** Strip `--` comments and collapse whitespace so formatting can't fail us. */
function normalizeSql(sql: string): string {
  return sql
    .replace(/--[^\n]*/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

const describeIfRepo = fs.existsSync(SHARED_SCHEMA) ? describe : describe.skip;

const MIGRATIONS_DIR = path.resolve(__dirname, "../../../sqlite/migrations");

function migrationFiles(): string[] {
  return fs
    .readdirSync(MIGRATIONS_DIR)
    .filter((f) => /^\d{4}_.*\.sql$/.test(f))
    .sort();
}

describeIfRepo("sqlite-store schema parity", () => {
  it("embedded SCHEMA matches sqlite/schema.sql", () => {
    const shared = normalizeSql(fs.readFileSync(SHARED_SCHEMA, "utf8"));
    expect(normalizeSql(SCHEMA)).toBe(shared);
  });

  it("MIGRATIONS covers 1..SCHEMA_VERSION and matches sqlite/migrations/", () => {
    const versions = Object.keys(MIGRATIONS)
      .map(Number)
      .sort((a, b) => a - b);
    expect(versions).toEqual(
      Array.from({ length: SCHEMA_VERSION }, (_, i) => i + 1),
    );
    const files = migrationFiles();
    expect(files.map((f) => Number(f.slice(0, 4)))).toEqual(versions);
    for (const file of files) {
      const version = Number(file.slice(0, 4));
      const onDisk = fs.readFileSync(path.join(MIGRATIONS_DIR, file), "utf8");
      expect(normalizeSql(MIGRATIONS[version]!)).toBe(normalizeSql(onDisk));
    }
  });

  (hasNodeSqlite ? it : it.skip)(
    "applying migrations in order equals sqlite/schema.sql",
    () => {
      const { DatabaseSync } = (
        process as unknown as {
          getBuiltinModule: (id: string) => {
            DatabaseSync: new (p: string) => {
              exec(sql: string): void;
              prepare(sql: string): { all(): unknown[] };
              close(): void;
            };
          };
        }
      ).getBuiltinModule("node:sqlite");

      const objects = (scripts: string[]) => {
        const db = new DatabaseSync(":memory:");
        try {
          for (const script of scripts) db.exec(script);
          const rows = db
            .prepare(
              "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name",
            )
            .all() as { type: string; name: string; sql: string }[];
          return rows.map((r) => [r.type, r.name, normalizeSql(r.sql)]);
        } finally {
          db.close();
        }
      };

      const migrated = objects(
        migrationFiles().map((f) =>
          fs.readFileSync(path.join(MIGRATIONS_DIR, f), "utf8"),
        ),
      );
      const snapshot = objects([fs.readFileSync(SHARED_SCHEMA, "utf8")]);
      expect(migrated).toEqual(snapshot);
    },
  );
});

describeWithSqlite("sqlite-store schema upgrade", () => {
  const PROBE = "ALTER TABLE test_runs ADD COLUMN _probe TEXT;";
  type RawDb = Parameters<typeof applyMigrations>[0] & { close(): void };
  const { DatabaseSync } = (
    process as unknown as {
      getBuiltinModule: (id: string) => {
        DatabaseSync: new (p: string) => RawDb;
      };
    }
  ).getBuiltinModule("node:sqlite");

  let dir: string;
  let db: string;

  beforeEach(() => {
    dir = tempDir();
    db = path.join(dir, DB_FILENAME);
  });
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

  // Open without `connect()`: once a file is at v2 the real (v1) `connect()`
  // correctly refuses it, which is itself asserted below.
  const withRaw = <T>(fn: (raw: RawDb) => T): T => {
    const raw = new DatabaseSync(db);
    try {
      return fn(raw);
    } finally {
      raw.close();
    }
  };
  const userVersion = () =>
    withRaw((raw) =>
      Number(
        (
          raw.prepare("PRAGMA user_version").get() as {
            user_version: number | bigint;
          }
        ).user_version,
      ),
    );
  const columns = (table: string) =>
    withRaw(
      (raw) =>
        new Set(
          (
            raw.prepare(`PRAGMA table_info(${table})`).all() as {
              name: string;
            }[]
          ).map((r) => r.name),
        ),
    );
  const upgrade = (
    migrations: Readonly<Record<number, string>>,
    targetVersion = 2,
  ) => withRaw((raw) => applyMigrations(raw, { migrations, targetVersion }));

  test("existing v1 file gets only step 2 and keeps its data", () => {
    writeTestRun(makeRun(), db);
    expect(userVersion()).toBe(SCHEMA_VERSION);
    expect(columns("test_runs").has("_probe")).toBe(false);

    // Step 1 must not re-run: make it explode if the loop starts at 1.
    upgrade({ 1: "SELECT RAISE(ABORT, 'step 1 re-ran');", 2: PROBE });

    expect(userVersion()).toBe(2);
    expect(columns("test_runs").has("_probe")).toBe(true);
    expect(
      withRaw((raw) =>
        raw.prepare("SELECT count(*) AS n FROM test_runs").get(),
      ),
    ).toMatchObject({ n: 1 });
    // ...and today's release now refuses the newer file.
    expect(() => connect(db)).toThrow(/newer than this deepeval/);
  });

  test("second upgrade pass is a no-op", () => {
    const migrations = { ...MIGRATIONS, 2: PROBE };
    connect(db).close(); // v1
    upgrade(migrations);
    // Re-running the ALTER would throw "duplicate column"; a no-op does not.
    upgrade(migrations);
    expect(userVersion()).toBe(2);
  });

  test("fresh file applies every step", () => {
    upgrade({ ...MIGRATIONS, 2: PROBE });
    expect(userVersion()).toBe(2);
    const cols = columns("test_runs");
    expect(cols.has("payload_json")).toBe(true);
    expect(cols.has("_probe")).toBe(true);
  });

  test("a failing step rolls back and leaves the old version", () => {
    connect(db).close(); // v1
    expect(() =>
      upgrade({
        ...MIGRATIONS,
        2: PROBE + "ALTER TABLE no_such_table ADD COLUMN x TEXT;",
      }),
    ).toThrow();
    expect(userVersion()).toBe(1);
    expect(columns("test_runs").has("_probe")).toBe(false);
  });

  test("refuses a file newer than the target", () => {
    connect(db).close(); // v1
    expect(() => upgrade(MIGRATIONS, 0)).toThrow(/newer than this deepeval/);
  });
});
