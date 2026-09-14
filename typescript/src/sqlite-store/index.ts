// SQLite local store for finished test runs. Opt in with
// `DEEPEVAL_LOCAL_STORE=sqlite` (Node 24+); see `store.ts` for the schema.

export {
  LOCAL_STORE_JSON,
  LOCAL_STORE_SQLITE,
  MIN_NODE_MAJOR_FOR_SQLITE,
  SqliteUnsupportedError,
  assertSqliteSupported,
  isSqliteSupported,
  nodeMajor,
  normalizeLocalStoreMode,
  resolveLocalStoreMode,
  sqliteUnsupportedMessage,
  type LocalStoreMode,
} from "@/sqlite-store/mode";
export {
  DB_FILENAME,
  MIGRATIONS,
  SCHEMA,
  SCHEMA_VERSION,
  TestRunNotFoundError,
  connect,
  formatSource,
  isDbPath,
  latestRunId,
  listTestRuns,
  loadTestRunPayload,
  parseSource,
  resolveDbPath,
  resolveIncludeRowJson,
  setConfidentTestRunId,
  writeTestRun,
  type TestRunSummaryRow,
  type WriteTestRunOptions,
} from "@/sqlite-store/store";
