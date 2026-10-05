"""SQLite storage for deepeval test runs (opt-in via `DEEPEVAL_LOCAL_STORE=sqlite`).

See `deepeval.sqlite_store.store` for the schema and implementation notes.
"""

from deepeval.sqlite_store.store import (
    DB_FILENAME,
    INCLUDE_ROW_JSON_ENV_VAR,
    MIGRATIONS,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    connect,
    format_source,
    is_db_path,
    latest_run_id,
    list_test_runs,
    load_test_run,
    load_test_run_payload,
    parse_source,
    resolve_db_path,
    resolve_include_row_json,
    set_confident_test_run_id,
    write_test_run,
)

__all__ = [
    "DB_FILENAME",
    "INCLUDE_ROW_JSON_ENV_VAR",
    "MIGRATIONS",
    "SCHEMA_SQL",
    "SCHEMA_VERSION",
    "connect",
    "format_source",
    "is_db_path",
    "latest_run_id",
    "list_test_runs",
    "load_test_run",
    "load_test_run_payload",
    "parse_source",
    "resolve_db_path",
    "resolve_include_row_json",
    "set_confident_test_run_id",
    "write_test_run",
]
