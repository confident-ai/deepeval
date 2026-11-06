// SQLite storage for deepeval test runs (opt-in via `DEEPEVAL_LOCAL_STORE=sqlite`).
//
// Each finished `deepeval test run` becomes one row in `test_runs`, with its
// test cases, traces, spans and metric scores broken out into their own tables
// so they can be queried across runs with plain SQL:
//
//   sqlite3 .deepeval/deepeval.db \
//     "SELECT r.id, m.name, avg(m.score) FROM metric_data m
//      JOIN test_runs r ON r.id = m.test_run_id GROUP BY r.id, m.name"
//
// The `test_runs` row keeps the exact object the JSON export would have
// written in `payload_json`, so a run reloads losslessly (this is what
// `deepeval inspect` reads). Test case, trace and span rows have the same
// column but it is only filled when `DEEPEVAL_SQLITE_INCLUDE_ROW_JSON=1`, since it
// roughly doubles the database size. The schema is byte-for-byte the one Python's
// `deepeval.sqlite_store` creates, so the two SDKs can share a database.
//
// Built on the runtime's `node:sqlite` — no native addon, no wheels. That
// module only became stable in Node 24, which is why `mode.ts` refuses the
// sqlite mode on older runtimes rather than loading an experimental API.

import * as fs from "fs";
import * as path from "path";
import { createRequire } from "module";
import {
  HIDDEN_DIR,
  DEEPEVAL_RESULTS_FOLDER,
  DEEPEVAL_SQLITE_INCLUDE_ROW_JSON,
} from "@/constants";
import {
  SqliteUnsupportedError,
  sqliteUnsupportedMessage,
} from "@/sqlite-store/mode";
import type { LocalTestRun } from "@/evaluate/test-run/local";

export const DB_FILENAME = "deepeval.db";
export const SCHEMA_VERSION = 1;

const DB_SUFFIXES = new Set([".db", ".sqlite", ".sqlite3"]);
const BUSY_TIMEOUT_MS = 30_000;

const SPAN_BUCKETS: ReadonlyArray<readonly [string, string]> = [
  ["baseSpans", "base"],
  ["agentSpans", "agent"],
  ["llmSpans", "llm"],
  ["retrieverSpans", "retriever"],
  ["toolSpans", "tool"],
];

type Json = Record<string, unknown>;
type SqlValue = string | number | bigint | null;

// `@types/node` in this package predates `node:sqlite`; this is the slice of
// its surface we use. Kept structural so a future types bump needs no edits.
interface NodeSqliteStatement {
  run(...params: SqlValue[]): {
    lastInsertRowid: number | bigint;
    changes: number | bigint;
  };
  get(...params: SqlValue[]): unknown;
  all(...params: SqlValue[]): unknown[];
}
interface NodeSqliteDatabase {
  exec(sql: string): void;
  prepare(sql: string): NodeSqliteStatement;
  close(): void;
}
interface NodeSqliteModule {
  DatabaseSync: new (
    filename: string,
    options?: { timeout?: number },
  ) => NodeSqliteDatabase;
}

/**
 * DDL run once when a database is first opened. The canonical copy lives in
 * `sqlite/schema.sql` at the repo root; `test/test-core/sqlite-store.test.ts`
 * fails if this string drifts from it.
 */
export const SCHEMA = `
CREATE TABLE IF NOT EXISTS test_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,
    identifier          TEXT,
    test_file           TEXT,
    dataset_alias       TEXT,
    dataset_id          TEXT,
    test_passed         INTEGER,
    test_failed         INTEGER,
    run_duration        REAL,
    evaluation_cost     REAL,
    official            INTEGER NOT NULL DEFAULT 0,
    confident_test_run_id TEXT,                -- set once the run is uploaded to Confident AI
    hyperparameters_json TEXT,
    prompts_json        TEXT,
    metrics_scores_json TEXT,
    payload_json        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_test_runs_confident ON test_runs(confident_test_run_id);

CREATE TABLE IF NOT EXISTS test_cases (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id     INTEGER NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    kind            TEXT    NOT NULL,          -- 'single-turn' | 'multi-turn'
    "order"         INTEGER,
    name            TEXT,
    input           TEXT,
    actual_output   TEXT,
    expected_output TEXT,
    success         INTEGER,
    run_duration    REAL,
    evaluation_cost REAL,
    tags_json       TEXT,
    payload_json    TEXT                     -- full object; only with DEEPEVAL_SQLITE_INCLUDE_ROW_JSON=1
);
CREATE INDEX IF NOT EXISTS idx_test_cases_run ON test_cases(test_run_id);

CREATE TABLE IF NOT EXISTS traces (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id   INTEGER NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    test_case_id  INTEGER NOT NULL REFERENCES test_cases(id) ON DELETE CASCADE,
    uuid          TEXT    NOT NULL,
    name          TEXT,
    status        TEXT,
    start_time    TEXT,
    end_time      TEXT,
    thread_id     TEXT,
    user_id       TEXT,
    environment   TEXT,
    input_json    TEXT,
    output_json   TEXT,
    metadata_json TEXT,
    tags_json     TEXT,
    payload_json  TEXT                       -- full object; only with DEEPEVAL_SQLITE_INCLUDE_ROW_JSON=1
);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(test_run_id);
CREATE INDEX IF NOT EXISTS idx_traces_case ON traces(test_case_id);

CREATE TABLE IF NOT EXISTS spans (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id           INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    uuid               TEXT    NOT NULL,
    parent_uuid        TEXT,
    type               TEXT    NOT NULL,
    name               TEXT,
    status             TEXT,
    start_time         TEXT,
    end_time           TEXT,
    error              TEXT,
    model              TEXT,
    provider           TEXT,
    input_token_count  REAL,
    output_token_count REAL,
    input_json         TEXT,
    output_json        TEXT,
    payload_json       TEXT                  -- full object; only with DEEPEVAL_SQLITE_INCLUDE_ROW_JSON=1
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_parent ON spans(trace_id, parent_uuid);

CREATE TABLE IF NOT EXISTS metric_data (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id      INTEGER NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    owner_type       TEXT    NOT NULL,
    owner_id         INTEGER NOT NULL,
    name             TEXT    NOT NULL,
    score            REAL,
    threshold        REAL,
    success          INTEGER,
    flaky            INTEGER NOT NULL DEFAULT 0,
    strict_mode      INTEGER NOT NULL DEFAULT 0,
    reason           TEXT,
    error            TEXT,
    evaluation_model TEXT,
    evaluation_cost  REAL,
    input_tokens     INTEGER,
    output_tokens    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_metric_data_run_name ON metric_data(test_run_id, name);
CREATE INDEX IF NOT EXISTS idx_metric_data_owner ON metric_data(owner_type, owner_id);
`;

/**
 * Ordered schema migrations keyed by the `PRAGMA user_version` each brings the
 * database *to*. `ensureSchema` applies every entry above the file's current
 * version, in order, inside one transaction. Each value is the embedded copy
 * of `sqlite/migrations/<NNNN>_*.sql`; parity tests keep them in sync. Never
 * edit a shipped entry: add the next one and bump `SCHEMA_VERSION`.
 */
export const MIGRATIONS: Readonly<Record<number, string>> = Object.freeze({
  1: SCHEMA, // 0001_initial.sql
});

for (let v = 1; v <= SCHEMA_VERSION; v++) {
  if (!(v in MIGRATIONS)) {
    throw new Error(
      `deepeval/sqlite-store: MIGRATIONS is missing step ${v} (SCHEMA_VERSION=${SCHEMA_VERSION}).`,
    );
  }
}

/**
 * `DEEPEVAL_SQLITE_INCLUDE_ROW_JSON` -> boolean (default false). Controls whether
 * `test_cases`, `traces` and `spans` rows also carry their full serialized
 * object in `payload_json`. The `test_runs` row always does, since that is
 * what `loadTestRunPayload` / `deepeval inspect` read back.
 */
export function resolveIncludeRowJson(): boolean {
  const raw = (process.env[DEEPEVAL_SQLITE_INCLUDE_ROW_JSON] ?? "")
    .trim()
    .toLowerCase();
  return ["1", "true", "yes", "y", "on"].includes(raw);
}

export interface WriteTestRunOptions {
  /** Store the full JSON object on child rows too. Default: `DEEPEVAL_SQLITE_INCLUDE_ROW_JSON`. */
  includeRowJson?: boolean;
}

// --------------------------------------------------------------------------- //
// Paths / source specs
// --------------------------------------------------------------------------- //

/**
 * Where `deepeval.db` lives: inside `DEEPEVAL_RESULTS_FOLDER` when set so it
 * sits next to where the user asked for results, else the hidden cache dir.
 */
export function resolveDbPath(
  resultsFolder: string | undefined = process.env[DEEPEVAL_RESULTS_FOLDER],
): string {
  const folder = resultsFolder && resultsFolder.trim();
  return path.join(folder || HIDDEN_DIR, DB_FILENAME);
}

export function isDbPath(source: string): boolean {
  const withoutRun = source.split("#", 1)[0] ?? source;
  return DB_SUFFIXES.has(path.extname(withoutRun).toLowerCase());
}

/** `<db_path>#<run_id>`: how `deepeval inspect` addresses a run in a DB. */
export function formatSource(dbPath: string, runId: number): string {
  return `${dbPath}#${runId}`;
}

/**
 * Split `<db_path>[#<run_id>]`. Returns `null` when `source` is not a DB, so
 * callers can fall through to the JSON loader.
 */
export function parseSource(
  source: string,
): { dbPath: string; runId: number | null } | null {
  if (!isDbPath(source)) return null;
  const hash = source.lastIndexOf("#");
  if (hash === -1) return { dbPath: source, runId: null };
  const runPart = source.slice(hash + 1).trim();
  return {
    dbPath: source.slice(0, hash),
    runId: /^\d+$/.test(runPart) ? Number.parseInt(runPart, 10) : null,
  };
}

// --------------------------------------------------------------------------- //
// Connection / schema
// --------------------------------------------------------------------------- //

let sqliteModule: NodeSqliteModule | undefined;

function loadNodeSqlite(): NodeSqliteModule {
  if (sqliteModule) return sqliteModule;
  // `process.getBuiltinModule` (Node 22.3+) sidesteps bundlers that would
  // otherwise try to resolve `node:sqlite` at build time.
  const getBuiltin = (
    process as unknown as {
      getBuiltinModule?: (id: string) => unknown;
    }
  ).getBuiltinModule;
  let mod: unknown;
  try {
    mod = getBuiltin
      ? getBuiltin("node:sqlite")
      : createRequire(__filename)("node:sqlite");
  } catch (e) {
    // Node < 22.5 has no `node:sqlite` at all; point at the real fix.
    throw new SqliteUnsupportedError(
      `${sqliteUnsupportedMessage()} (${(e as Error).message})`,
    );
  }
  if (!mod || typeof (mod as NodeSqliteModule).DatabaseSync !== "function") {
    throw new SqliteUnsupportedError(sqliteUnsupportedMessage());
  }
  sqliteModule = mod as NodeSqliteModule;
  return sqliteModule;
}

/** Open (creating if needed) `dbPath` with the schema in place. Callers must `close()`. */
export function connect(dbPath: string): NodeSqliteDatabase {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const { DatabaseSync } = loadNodeSqlite();
  const db = new DatabaseSync(dbPath, { timeout: BUSY_TIMEOUT_MS });
  try {
    db.exec("PRAGMA foreign_keys = ON");
    try {
      db.exec("PRAGMA journal_mode = WAL");
    } catch {
      // Network filesystems and some bind mounts refuse WAL; the default
      // rollback journal works everywhere.
    }
    ensureSchema(db);
  } catch (e) {
    db.close();
    throw e;
  }
  return db;
}

function ensureSchema(db: NodeSqliteDatabase): void {
  applyMigrations(db, {
    migrations: MIGRATIONS,
    targetVersion: SCHEMA_VERSION,
  });
}

export interface ApplyMigrationsOptions {
  /** Registry keyed by the version each script brings the database to. */
  migrations: Readonly<Record<number, string>>;
  /** Version to upgrade the file to (usually `SCHEMA_VERSION`). */
  targetVersion: number;
}

/**
 * Bring `db` from its current `PRAGMA user_version` to `targetVersion` by
 * running each pending migration in order, then stamp the version, all in one
 * transaction. No-op when already current; throws when the file is newer than
 * `targetVersion`. Exported for tests; `connect()` calls it with the real
 * `MIGRATIONS` / `SCHEMA_VERSION`.
 */
export function applyMigrations(
  db: NodeSqliteDatabase,
  { migrations, targetVersion }: ApplyMigrationsOptions,
): void {
  const row = db.prepare("PRAGMA user_version").get() as
    | { user_version: number | bigint }
    | undefined;
  const version = Number(row?.user_version ?? 0);
  if (version === targetVersion) return;
  if (version > targetVersion) {
    throw new Error(
      `deepeval.db schema version ${version} is newer than this deepeval ` +
        `supports (${targetVersion}); please upgrade deepeval.`,
    );
  }
  // Every pending step plus the version stamp land atomically; a crash midway
  // leaves the file at its old version for the next open to retry.
  db.exec("BEGIN");
  try {
    for (let step = version + 1; step <= targetVersion; step++) {
      const script = migrations[step];
      if (script === undefined) {
        throw new Error(
          `deepeval/sqlite-store: no migration for schema version ${step}.`,
        );
      }
      db.exec(script);
    }
    db.exec(`PRAGMA user_version = ${targetVersion}`);
    db.exec("COMMIT");
  } catch (e) {
    db.exec("ROLLBACK");
    throw e;
  }
}

// --------------------------------------------------------------------------- //
// Write
// --------------------------------------------------------------------------- //

function isRecord(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function dumps(value: unknown): string | null {
  return value === undefined || value === null ? null : JSON.stringify(value);
}

/** Free-text column: strings as-is, anything else as JSON. */
function text(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  return typeof value === "string" ? value : JSON.stringify(value);
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function bool(value: unknown): number | null {
  return value === undefined || value === null ? null : value ? 1 : 0;
}

function str(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function rowId(result: { lastInsertRowid: number | bigint }): number {
  return Number(result.lastInsertRowid);
}

/** Child-row `payload_json`: the full object, or NULL when opted out. */
function payloadOf(obj: Json, include: boolean): string | null {
  return include ? JSON.stringify(obj) : null;
}

/**
 * Persist `run` and everything inside it; returns the new run id.
 *
 * One transaction, so a crash mid-write never leaves a half-inserted run.
 * Throws on storage failure; callers should catch so persistence problems
 * never fail the evaluation itself.
 *
 * `includeRowJson` (default: `DEEPEVAL_SQLITE_INCLUDE_ROW_JSON`) also stores the
 * full JSON object on every test case, trace and span row. The run row
 * always keeps its payload regardless.
 */
export function writeTestRun(
  run: LocalTestRun,
  dbPath: string,
  options: WriteTestRunOptions = {},
): number {
  const includeRowJson = options.includeRowJson ?? resolveIncludeRowJson();
  // Round-trip through JSON so the normalized columns are derived from the
  // exact payload we store (drops `undefined`, normalizes Dates, etc.).
  const payload = JSON.parse(JSON.stringify(run)) as Json;

  const db = connect(dbPath);
  try {
    db.exec("BEGIN");
    try {
      const runId = insertTestRun(db, payload);
      const cases = Array.isArray(payload.cases) ? payload.cases : [];
      for (const persisted of cases) {
        if (!isRecord(persisted)) continue;
        const entry = isRecord(persisted.entry) ? persisted.entry : persisted;
        const kind =
          persisted.conversational === true ? "multi-turn" : "single-turn";
        insertTestCase(db, runId, kind, entry, persisted, includeRowJson);
      }
      db.exec("COMMIT");
      return runId;
    } catch (e) {
      db.exec("ROLLBACK");
      throw e;
    }
  } finally {
    db.close();
  }
}

function sumEvaluationCost(payload: Json): number | null {
  const cases = Array.isArray(payload.cases) ? payload.cases : [];
  let total: number | null = null;
  for (const persisted of cases) {
    if (!isRecord(persisted)) continue;
    const entry = isRecord(persisted.entry) ? persisted.entry : persisted;
    const cost = num(entry.evaluationCost);
    if (cost !== null) total = (total ?? 0) + cost;
  }
  return total;
}

function firstCaseField(payload: Json, key: string): string | null {
  const cases = Array.isArray(payload.cases) ? payload.cases : [];
  for (const persisted of cases) {
    if (isRecord(persisted) && typeof persisted[key] === "string") {
      return persisted[key] as string;
    }
  }
  return null;
}

function insertTestRun(db: NodeSqliteDatabase, payload: Json): number {
  const result = db
    .prepare(
      `INSERT INTO test_runs (
         created_at, identifier, test_file, dataset_alias, dataset_id,
         test_passed, test_failed, run_duration, evaluation_cost, official,
         confident_test_run_id, hyperparameters_json, prompts_json,
         metrics_scores_json, payload_json
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      str(payload.savedAt) ?? new Date().toISOString(),
      str(payload.identifier),
      null,
      firstCaseField(payload, "datasetAlias"),
      firstCaseField(payload, "datasetId"),
      num(payload.testPassed),
      num(payload.testFailed),
      num(payload.runDuration),
      sumEvaluationCost(payload),
      payload.official ? 1 : 0,
      // `wrapUpTestRun` posts before exporting, so a logged-in run already
      // carries the id Confident AI assigned.
      str(payload.testRunId),
      dumps(payload.hyperparameters),
      null,
      null,
      JSON.stringify(payload),
    );
  return rowId(result);
}

function insertTestCase(
  db: NodeSqliteDatabase,
  runId: number,
  kind: "single-turn" | "multi-turn",
  entry: Json,
  persisted: Json,
  includeRowJson: boolean,
): number {
  const conversational = kind === "multi-turn";
  const result = db
    .prepare(
      `INSERT INTO test_cases (
         test_run_id, kind, "order", name, input, actual_output,
         expected_output, success, run_duration, evaluation_cost,
         tags_json, payload_json
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      runId,
      kind,
      num(entry.order),
      str(entry.name),
      text(conversational ? entry.scenario : entry.input),
      conversational ? null : text(entry.actualOutput),
      text(conversational ? entry.expectedOutcome : entry.expectedOutput),
      bool(entry.success),
      num(entry.runDuration),
      num(entry.evaluationCost),
      dumps(entry.tags),
      payloadOf(persisted, includeRowJson),
    );
  const caseId = rowId(result);

  insertMetrics(db, runId, "test_case", caseId, entry.metricsData);
  if (isRecord(entry.trace)) {
    insertTrace(db, runId, caseId, entry.trace, includeRowJson);
  }
  return caseId;
}

function insertTrace(
  db: NodeSqliteDatabase,
  runId: number,
  caseId: number,
  trace: Json,
  includeRowJson: boolean,
): number {
  const result = db
    .prepare(
      `INSERT INTO traces (
         test_run_id, test_case_id, uuid, name, status, start_time,
         end_time, thread_id, user_id, environment, input_json,
         output_json, metadata_json, tags_json, payload_json
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      runId,
      caseId,
      str(trace.uuid) ?? "",
      str(trace.name),
      str(trace.status),
      str(trace.startTime),
      str(trace.endTime),
      str(trace.threadId),
      str(trace.userId),
      str(trace.environment),
      dumps(trace.input),
      dumps(trace.output),
      dumps(trace.metadata),
      dumps(trace.tags),
      payloadOf(trace, includeRowJson),
    );
  const traceId = rowId(result);

  insertMetrics(db, runId, "trace", traceId, trace.metricsData);
  for (const [bucket, defaultType] of SPAN_BUCKETS) {
    const spans = trace[bucket];
    if (!Array.isArray(spans)) continue;
    for (const span of spans) {
      if (isRecord(span)) {
        insertSpan(db, runId, traceId, defaultType, span, includeRowJson);
      }
    }
  }
  return traceId;
}

function insertSpan(
  db: NodeSqliteDatabase,
  runId: number,
  traceId: number,
  defaultType: string,
  span: Json,
  includeRowJson: boolean,
): number {
  const result = db
    .prepare(
      `INSERT INTO spans (
         trace_id, uuid, parent_uuid, type, name, status, start_time,
         end_time, error, model, provider, input_token_count,
         output_token_count, input_json, output_json, payload_json
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      traceId,
      str(span.uuid) ?? "",
      str(span.parentUuid),
      str(span.type) ?? defaultType,
      str(span.name),
      str(span.status),
      str(span.startTime),
      str(span.endTime),
      str(span.error),
      str(span.model),
      str(span.provider),
      num(span.inputTokenCount),
      num(span.outputTokenCount),
      dumps(span.input),
      dumps(span.output),
      payloadOf(span, includeRowJson),
    );
  const spanId = rowId(result);
  insertMetrics(db, runId, "span", spanId, span.metricsData);
  return spanId;
}

function insertMetrics(
  db: NodeSqliteDatabase,
  runId: number,
  ownerType: "test_case" | "trace" | "span",
  ownerId: number,
  metrics: unknown,
): void {
  if (!Array.isArray(metrics)) return;
  const stmt = db.prepare(
    `INSERT INTO metric_data (
       test_run_id, owner_type, owner_id, name, score, threshold,
       success, flaky, strict_mode, reason, error, evaluation_model,
       evaluation_cost, input_tokens, output_tokens
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  );
  for (const m of metrics) {
    if (!isRecord(m) || typeof m.name !== "string" || !m.name) continue;
    stmt.run(
      runId,
      ownerType,
      ownerId,
      m.name,
      num(m.score),
      num(m.threshold),
      bool(m.success),
      m.flaky ? 1 : 0,
      m.strictMode ? 1 : 0,
      str(m.reason),
      str(m.error),
      str(m.evaluationModel),
      num(m.evaluationCost),
      num(m.inputTokenCount),
      num(m.outputTokenCount),
    );
  }
}

// --------------------------------------------------------------------------- //
// Read
// --------------------------------------------------------------------------- //

export interface TestRunSummaryRow {
  id: number;
  created_at: string;
  identifier: string | null;
  test_file: string | null;
  dataset_alias: string | null;
  test_passed: number | null;
  test_failed: number | null;
  run_duration: number | null;
  evaluation_cost: number | null;
  official: number;
  confident_test_run_id: string | null;
}

/** Newest-first summaries of stored runs (summary columns only, no payload_json). */
/**
 * Record the Confident AI test run id for a stored run, e.g. after a later
 * `deepeval view` upload. Throws `TestRunNotFoundError` if `runId` is absent.
 */
export function setConfidentTestRunId(
  dbPath: string,
  runId: number,
  confidentTestRunId: string,
): void {
  const db = connect(dbPath);
  try {
    const result = db
      .prepare("UPDATE test_runs SET confident_test_run_id = ? WHERE id = ?")
      .run(confidentTestRunId, runId);
    if (Number(result.changes) === 0) {
      throw new TestRunNotFoundError(
        `${dbPath} has no test run with id ${runId}.`,
      );
    }
  } finally {
    db.close();
  }
}

export function listTestRuns(dbPath: string, limit = 20): TestRunSummaryRow[] {
  if (!fs.existsSync(dbPath)) return [];
  const db = connect(dbPath);
  try {
    const rows = db
      .prepare(
        `SELECT id, created_at, identifier, test_file, dataset_alias,
                test_passed, test_failed, run_duration, evaluation_cost, official,
                confident_test_run_id
         FROM test_runs ORDER BY id DESC LIMIT ?`,
      )
      .all(limit) as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      ...(r as unknown as TestRunSummaryRow),
      id: Number(r.id),
      official: Number(r.official ?? 0),
    }));
  } finally {
    db.close();
  }
}

export function latestRunId(dbPath: string): number | null {
  if (!fs.existsSync(dbPath)) return null;
  const db = connect(dbPath);
  try {
    const row = db.prepare("SELECT max(id) AS id FROM test_runs").get() as
      | { id: number | bigint | null }
      | undefined;
    return row?.id === null || row?.id === undefined ? null : Number(row.id);
  } finally {
    db.close();
  }
}

export class TestRunNotFoundError extends Error {}

/**
 * `{ runId, payload }` where `payload` is the original `LocalTestRun` (same
 * shape as a `test_run_*.json` export). Latest run when `runId` is null.
 *
 * Throws `TestRunNotFoundError` if the DB is missing, empty, or lacks `runId`.
 */
export function loadTestRunPayload(
  dbPath: string,
  runId: number | null = null,
): { runId: number; payload: LocalTestRun } {
  if (!fs.existsSync(dbPath)) {
    throw new TestRunNotFoundError(`SQLite store not found: ${dbPath}`);
  }
  const db = connect(dbPath);
  try {
    const row = (
      runId === null
        ? db
            .prepare(
              "SELECT id, payload_json FROM test_runs ORDER BY id DESC LIMIT 1",
            )
            .get()
        : db
            .prepare("SELECT id, payload_json FROM test_runs WHERE id = ?")
            .get(runId)
    ) as { id: number | bigint; payload_json: string } | undefined;
    if (!row) {
      throw new TestRunNotFoundError(
        runId === null
          ? `${dbPath} contains no test runs yet.`
          : `No test run with id ${runId} in ${dbPath}.`,
      );
    }
    return {
      runId: Number(row.id),
      payload: JSON.parse(row.payload_json) as LocalTestRun,
    };
  } finally {
    db.close();
  }
}
