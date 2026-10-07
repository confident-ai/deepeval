-- Migration 0001: initial schema (brings PRAGMA user_version to 1).
--
-- Applied by both SDKs when they open a deepeval.db whose user_version is 0.
-- Never edit a migration after it has shipped; add the next NNNN_<slug>.sql
-- and update sqlite/schema.sql to match the result of applying them all.

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
