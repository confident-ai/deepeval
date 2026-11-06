"""SQLite storage for deepeval test runs (opt-in via `DEEPEVAL_LOCAL_STORE=sqlite`).

Every finished `evaluate()` / `evals_iterator()` / `deepeval test run` call
becomes one row in `test_runs`, with its test cases, traces, spans and metric
scores broken out into their own tables so they can be queried across runs
with plain SQL:

    sqlite3 .deepeval/deepeval.db \\
      "SELECT r.id, m.name, avg(m.score) FROM metric_data m
       JOIN test_runs r ON r.id = m.test_run_id GROUP BY r.id, m.name"

The `test_runs` row keeps the exact pydantic dump of the whole run in
`payload_json`, so the full `TestRun` can be reloaded losslessly (this is
what `deepeval inspect` uses). Test case, trace and span rows have the same
column but it is only filled when `DEEPEVAL_SQLITE_INCLUDE_ROW_JSON=1`, since it
roughly doubles the database size. Only the standard-library `sqlite3` module is
used: no wheels, no compiled extensions, works on any Python that can run
deepeval.

Portability notes:
  * The schema uses nothing newer than SQLite 3.7 (no `RETURNING`, strict
    tables or JSON functions) so it runs on the oldest bundled SQLite we
    still see in the wild (Ubuntu 20.04 system Python ships 3.31).
  * WAL journal mode is attempted for better concurrent-read behaviour but
    silently falls back to the default rollback journal on filesystems
    that do not support it (NFS/SMB, some Docker bind mounts).
  * Connections are always closed explicitly: on Windows an open handle
    blocks deleting or renaming the `.db` file.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from types import MappingProxyType
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from deepeval.constants import HIDDEN_DIR
from deepeval.test_run.test_run import TestRun, TestRunEncoder

DB_FILENAME = "deepeval.db"
SCHEMA_VERSION = 1

_DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_CONNECT_TIMEOUT_S = 30.0

_SPAN_BUCKETS: Tuple[Tuple[str, str], ...] = (
    ("baseSpans", "base"),
    ("agentSpans", "agent"),
    ("llmSpans", "llm"),
    ("retrieverSpans", "retriever"),
    ("toolSpans", "tool"),
)

_SCHEMA = """
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
"""

# Public alias. The canonical copy lives in `sqlite/schema.sql` at the repo root;
# tests/test_core/test_sqlite_store.py fails if this string drifts from it.
SCHEMA_SQL = _SCHEMA

# Ordered schema migrations, keyed by the `PRAGMA user_version` each one
# brings the database *to*. `_ensure_schema` applies every entry above the
# file's current version, in order, inside one transaction. Each value is the
# embedded copy of `sqlite/migrations/<NNNN>_*.sql`; parity tests keep them in
# sync. Never edit a shipped entry: add the next one and bump SCHEMA_VERSION.
_MIGRATIONS: Dict[int, str] = {
    1: _SCHEMA,  # 0001_initial.sql
}
MIGRATIONS: Mapping[int, str] = MappingProxyType(_MIGRATIONS)

if set(_MIGRATIONS) != set(range(1, SCHEMA_VERSION + 1)):  # pragma: no cover
    raise RuntimeError(
        "deepeval.sqlite_store: _MIGRATIONS must have exactly one entry for "
        f"every version 1..{SCHEMA_VERSION}, got {sorted(_MIGRATIONS)}."
    )

INCLUDE_ROW_JSON_ENV_VAR = "DEEPEVAL_SQLITE_INCLUDE_ROW_JSON"


def resolve_include_row_json() -> bool:
    """`DEEPEVAL_SQLITE_INCLUDE_ROW_JSON` -> bool (default False).

    Controls whether `test_cases`, `traces` and `spans` rows also carry their
    full serialized object in `payload_json`. The `test_runs` row always does,
    since that is what `load_test_run` / `deepeval inspect` read back.
    """
    raw = (os.getenv(INCLUDE_ROW_JSON_ENV_VAR) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Paths / source specs
# --------------------------------------------------------------------------- #


def resolve_db_path(
    results_folder: Optional[str] = None,
    results_subfolder: Optional[str] = None,
) -> Path:
    """Where `deepeval.db` lives.

    `results_folder` (or `DEEPEVAL_RESULTS_FOLDER`) wins when set so the DB
    sits next to where the user asked for results; otherwise it goes in the
    hidden cache dir (`.deepeval/` by default).
    """
    folder = results_folder or os.getenv("DEEPEVAL_RESULTS_FOLDER")
    if folder:
        base = Path(folder)
        if results_subfolder:
            base = base / results_subfolder
    else:
        base = Path(HIDDEN_DIR)
    return base / DB_FILENAME


def is_db_path(path: "str | Path") -> bool:
    return Path(str(path).split("#", 1)[0]).suffix.lower() in _DB_SUFFIXES


def format_source(db_path: "str | Path", run_id: int) -> str:
    """`<db_path>#<run_id>`: the string `deepeval inspect` uses to address a
    run inside a DB, and what shows up in the TUI header."""
    return f"{db_path}#{run_id}"


def parse_source(source: "str | Path") -> Optional[Tuple[Path, Optional[int]]]:
    """Split `<db_path>[#<run_id>]` into its parts.

    Returns `None` when `source` does not look like a SQLite DB, so callers
    can fall through to the JSON loader.
    """
    s = str(source)
    if not is_db_path(s):
        return None
    if "#" in s:
        path_part, _, run_part = s.rpartition("#")
        run_part = run_part.strip()
        if run_part.isdigit():
            return Path(path_part), int(run_part)
        return Path(path_part), None
    return Path(s), None


# --------------------------------------------------------------------------- #
# Connection / schema
# --------------------------------------------------------------------------- #


def connect(db_path: "str | Path") -> sqlite3.Connection:
    """Open (creating if needed) `db_path` and make sure the schema exists.

    Callers must `close()` the connection; prefer `contextlib.closing`.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=_CONNECT_TIMEOUT_S)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:  # pragma: no cover - fs dependent
            # Network filesystems and some bind mounts refuse WAL; the
            # default rollback journal works everywhere.
            pass
        _ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    if version > SCHEMA_VERSION:
        raise sqlite3.OperationalError(
            f"deepeval.db schema version {version} is newer than this "
            f"deepeval supports ({SCHEMA_VERSION}); please upgrade deepeval."
        )
    # `executescript` issues a COMMIT first, so drive the transaction by hand:
    # every pending step plus the version stamp land atomically, and a crash
    # midway leaves the file at its old version for the next open to retry.
    conn.execute("BEGIN")
    try:
        for step in range(version + 1, SCHEMA_VERSION + 1):
            for statement in _split_statements(_MIGRATIONS[step]):
                conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _split_statements(script: str) -> List[str]:
    """Split a migration script into statements (`--` comments stripped).

    Migration files are plain DDL without string literals containing `;`,
    so splitting on `;` is sufficient and keeps everything in one transaction
    (`sqlite3.Cursor.executescript` would commit first).
    """
    body = "\n".join(line.split("--", 1)[0] for line in script.splitlines())
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #


def write_test_run(
    test_run: TestRun,
    db_path: "str | Path",
    include_row_json: Optional[bool] = None,
) -> int:
    """Persist `test_run` and everything inside it. Returns the new run id.

    Runs as a single transaction so a crash mid-write never leaves a
    half-inserted run. Raises `sqlite3.Error` on failure; callers should
    catch it so a storage problem never fails the evaluation itself.

    `include_row_json` (default: `DEEPEVAL_SQLITE_INCLUDE_ROW_JSON`) also stores the
    full JSON object on every test case, trace and span row. The run row
    always keeps its payload regardless.
    """
    if include_row_json is None:
        include_row_json = resolve_include_row_json()
    payload = _to_plain_json(test_run)

    conn = connect(db_path)
    try:
        with conn:
            run_id = _insert_test_run(conn, payload)
            for kind, key in (
                ("single-turn", "testCases"),
                ("multi-turn", "conversationalTestCases"),
            ):
                for case in payload.get(key) or []:
                    _insert_test_case(
                        conn, run_id, kind, case, include_row_json
                    )
        return run_id
    finally:
        conn.close()


def set_confident_test_run_id(
    db_path: "str | Path", run_id: int, confident_test_run_id: str
) -> None:
    """Record the Confident AI test run id for a stored run.

    Called after a successful upload so `WHERE confident_test_run_id IS NULL`
    finds runs that only exist locally. Raises `LookupError` if `run_id` is
    not in the database.
    """
    conn = connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE test_runs SET confident_test_run_id = ? WHERE id = ?",
                (str(confident_test_run_id), int(run_id)),
            )
            if cur.rowcount == 0:
                raise LookupError(
                    f"{db_path} has no test run with id {run_id}."
                )
    finally:
        conn.close()


def _to_plain_json(test_run: TestRun) -> Dict[str, Any]:
    """`TestRun` -> pure-JSON dict identical to what the JSON store writes."""
    try:
        body = test_run.model_dump(by_alias=True, exclude_none=True)
    except AttributeError:  # pragma: no cover - pydantic v1
        body = test_run.dict(by_alias=True, exclude_none=True)
    # Round-trip through the same encoder as the JSON store so enums, sets,
    # datetimes etc. end up as plain JSON scalars in both the payload column
    # and the normalized columns.
    return json.loads(json.dumps(body, cls=TestRunEncoder))


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _text(value: Any) -> Optional[str]:
    """Free-text column: strings are stored as-is, anything else as JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _bool(value: Any) -> Optional[int]:
    return None if value is None else int(bool(value))


def _payload(obj: Dict[str, Any], include: bool) -> Optional[str]:
    """Child-row `payload_json`: the full object, or NULL when opted out."""
    return json.dumps(obj, ensure_ascii=False) if include else None


def _insert_test_run(conn: sqlite3.Connection, payload: Dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO test_runs (
            created_at, identifier, test_file, dataset_alias, dataset_id,
            test_passed, test_failed, run_duration, evaluation_cost, official,
            confident_test_run_id, hyperparameters_json, prompts_json,
            metrics_scores_json, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            payload.get("identifier"),
            payload.get("testFile"),
            payload.get("datasetAlias"),
            payload.get("datasetId"),
            payload.get("testPassed"),
            payload.get("testFailed"),
            payload.get("runDuration"),
            payload.get("evaluationCost"),
            int(bool(payload.get("official", False))),
            None,  # confident_test_run_id: set by set_confident_test_run_id()
            _dumps(payload.get("hyperparameters")),
            _dumps(payload.get("prompts")),
            _dumps(payload.get("metricsScores")),
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    return int(cur.lastrowid)


def _insert_test_case(
    conn: sqlite3.Connection,
    run_id: int,
    kind: str,
    case: Dict[str, Any],
    include_row_json: bool,
) -> int:
    if kind == "multi-turn":
        input_text = _text(case.get("scenario"))
        actual_output = None
        expected_output = _text(case.get("expectedOutcome"))
    else:
        input_text = _text(case.get("input"))
        actual_output = _text(case.get("actualOutput"))
        expected_output = _text(case.get("expectedOutput"))

    cur = conn.execute(
        """
        INSERT INTO test_cases (
            test_run_id, kind, "order", name, input, actual_output,
            expected_output, success, run_duration, evaluation_cost,
            tags_json, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            kind,
            case.get("order"),
            case.get("name"),
            input_text,
            actual_output,
            expected_output,
            _bool(case.get("success")),
            case.get("runDuration"),
            case.get("evaluationCost"),
            _dumps(case.get("tags")),
            _payload(case, include_row_json),
        ),
    )
    case_id = int(cur.lastrowid)

    _insert_metrics(conn, run_id, "test_case", case_id, case.get("metricsData"))

    trace = case.get("trace")
    if isinstance(trace, dict):
        _insert_trace(conn, run_id, case_id, trace, include_row_json)

    return case_id


def _insert_trace(
    conn: sqlite3.Connection,
    run_id: int,
    case_id: int,
    trace: Dict[str, Any],
    include_row_json: bool,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO traces (
            test_run_id, test_case_id, uuid, name, status, start_time,
            end_time, thread_id, user_id, environment, input_json,
            output_json, metadata_json, tags_json, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            case_id,
            trace.get("uuid"),
            trace.get("name"),
            trace.get("status"),
            trace.get("startTime"),
            trace.get("endTime"),
            trace.get("threadId"),
            trace.get("userId"),
            trace.get("environment"),
            _dumps(trace.get("input")),
            _dumps(trace.get("output")),
            _dumps(trace.get("metadata")),
            _dumps(trace.get("tags")),
            _payload(trace, include_row_json),
        ),
    )
    trace_id = int(cur.lastrowid)

    _insert_metrics(conn, run_id, "trace", trace_id, trace.get("metricsData"))

    for bucket, default_type in _SPAN_BUCKETS:
        for span in trace.get(bucket) or []:
            if isinstance(span, dict):
                _insert_span(
                    conn, run_id, trace_id, default_type, span, include_row_json
                )

    return trace_id


def _insert_span(
    conn: sqlite3.Connection,
    run_id: int,
    trace_id: int,
    default_type: str,
    span: Dict[str, Any],
    include_row_json: bool,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO spans (
            trace_id, uuid, parent_uuid, type, name, status, start_time,
            end_time, error, model, provider, input_token_count,
            output_token_count, input_json, output_json, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            span.get("uuid"),
            span.get("parentUuid"),
            span.get("type") or default_type,
            span.get("name"),
            span.get("status"),
            span.get("startTime"),
            span.get("endTime"),
            span.get("error"),
            span.get("model"),
            span.get("provider"),
            span.get("inputTokenCount"),
            span.get("outputTokenCount"),
            _dumps(span.get("input")),
            _dumps(span.get("output")),
            _payload(span, include_row_json),
        ),
    )
    span_id = int(cur.lastrowid)
    _insert_metrics(conn, run_id, "span", span_id, span.get("metricsData"))
    return span_id


def _insert_metrics(
    conn: sqlite3.Connection,
    run_id: int,
    owner_type: str,
    owner_id: int,
    metrics: Optional[Iterable[Dict[str, Any]]],
) -> None:
    rows = []
    for m in metrics or []:
        if not isinstance(m, dict) or not m.get("name"):
            continue
        rows.append(
            (
                run_id,
                owner_type,
                owner_id,
                m.get("name"),
                m.get("score"),
                m.get("threshold"),
                _bool(m.get("success")),
                int(bool(m.get("flaky", False))),
                int(bool(m.get("strictMode", False))),
                m.get("reason"),
                m.get("error"),
                m.get("evaluationModel"),
                m.get("evaluationCost"),
                m.get("inputTokenCount"),
                m.get("outputTokenCount"),
            )
        )
    if rows:
        conn.executemany(
            """
            INSERT INTO metric_data (
                test_run_id, owner_type, owner_id, name, score, threshold,
                success, flaky, strict_mode, reason, error, evaluation_model,
                evaluation_cost, input_tokens, output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #


def list_test_runs(
    db_path: "str | Path", limit: int = 20
) -> List[Dict[str, Any]]:
    """Newest-first summaries of stored runs (summary columns only, no payload_json)."""
    if not Path(db_path).is_file():
        return []
    conn = connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, identifier, test_file, dataset_alias,
                   test_passed, test_failed, run_duration, evaluation_cost,
                   official, confident_test_run_id
            FROM test_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def latest_run_id(db_path: "str | Path") -> Optional[int]:
    if not Path(db_path).is_file():
        return None
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT max(id) FROM test_runs").fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()


def load_test_run_payload(
    db_path: "str | Path", run_id: Optional[int] = None
) -> Tuple[int, Dict[str, Any]]:
    """Return `(run_id, payload)` where `payload` is the original `TestRun`
    dump (same shape as a `test_run_*.json` file). Latest run when
    `run_id` is `None`.

    Raises `FileNotFoundError` if the DB is missing, `LookupError` if the
    run id does not exist (or the DB is empty).
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite store not found: {path}")

    conn = connect(path)
    try:
        if run_id is None:
            row = conn.execute(
                "SELECT id, payload_json FROM test_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise LookupError(f"{path} contains no test runs yet.")
        else:
            row = conn.execute(
                "SELECT id, payload_json FROM test_runs WHERE id = ?",
                (int(run_id),),
            ).fetchone()
            if row is None:
                raise LookupError(f"No test run with id {run_id} in {path}.")
        return int(row[0]), json.loads(row[1])
    finally:
        conn.close()


def load_test_run(
    db_path: "str | Path", run_id: Optional[int] = None
) -> Tuple[int, TestRun]:
    """Like `load_test_run_payload` but validates back into a `TestRun`."""
    rid, payload = load_test_run_payload(db_path, run_id)
    return rid, TestRun.model_validate(payload)
