-- SQLite schema for deepeval's local backend store (`DEEPEVAL_LOCAL_STORE=sqlite`).
--
-- This file is the FULL CURRENT schema: what you get by applying every file in
-- sqlite/migrations/ in order to an empty database. Both SDKs embed the
-- migrations, and a parity test in each fails if they drift:
--   Python:     deepeval/sqlite_store/store.py        (_MIGRATIONS, SCHEMA_SQL)
--   TypeScript: typescript/src/sqlite-store/store.ts  (MIGRATIONS, SCHEMA)
-- A further test checks that migrations-applied-in-order == this file, and the
-- docs page (docs/content/docs/evaluation-local-backend-storage.mdx) is checked
-- against this file too.
--
-- To change the schema: add sqlite/migrations/NNNN_<slug>.sql (NNNN = the
-- user_version it brings the database to), mirror it into both SDKs' migration
-- registries, bump SCHEMA_VERSION, then update this file to the new full
-- schema. Never edit a migration that has shipped.
--
-- The `test_runs` row always keeps the complete serialized run in `payload_json`
-- (that is what `inspect` reloads). `test_cases`, `traces` and `spans` have the
-- same column but it is only filled when DEEPEVAL_SQLITE_INCLUDE_ROW_JSON=1, because it
-- roughly doubles the file size. `metric_data` (one row per MetricData result)
-- has no payload. All other columns are a promoted subset for filtering and
-- joining without parsing JSON.

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
