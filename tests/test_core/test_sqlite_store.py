"""Unit + integration tests for deepeval/sqlite_store."""

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Optional

import pytest

from deepeval import sqlite_store
from deepeval.evaluate.local_store import (
    LOCAL_STORE_JSON,
    LOCAL_STORE_SQLITE,
    resolve_local_store_mode,
    write_test_run as write_json_test_run,
)
from deepeval.inspect.loader import (
    InspectLoadError,
    load_test_run,
    run_id_from_path,
    summarize_test_run,
)
from deepeval.test_run.api import (
    ConversationalApiTestCase,
    LLMApiTestCase,
    TurnApi,
)
from deepeval.test_run.test_run import (
    TestRun as _TestRun,
    TestRunEncoder,
    TestRunManager as _TestRunManager,
)
from deepeval.tracing.api import (
    BaseApiSpan,
    MetricData,
    SpanApiType,
    TraceApi,
    TraceSpanApiStatus,
)


def _metric(name: str, score: float, success: bool = True) -> MetricData:
    return MetricData(
        name=name,
        threshold=0.5,
        success=success,
        score=score,
        reason=f"{name} reason",
        evaluationModel="gpt-4o-mini",
        evaluationCost=0.001,
        inputTokenCount=10,
        outputTokenCount=5,
    )


def _trace(uuid: str = "trace-1") -> TraceApi:
    root = BaseApiSpan(
        uuid="span-root",
        name="agent",
        status=TraceSpanApiStatus.SUCCESS,
        type=SpanApiType.AGENT,
        startTime="2026-01-01T00:00:00",
        endTime="2026-01-01T00:00:03",
        input={"q": "hi"},
        output="done",
        metricsData=[_metric("TaskCompletion", 0.9)],
    )
    llm = BaseApiSpan(
        uuid="span-llm",
        name="openai.chat",
        status=TraceSpanApiStatus.SUCCESS,
        type=SpanApiType.LLM,
        parentUuid="span-root",
        startTime="2026-01-01T00:00:01",
        endTime="2026-01-01T00:00:02",
        model="gpt-4o-mini",
        provider="openai",
        inputTokenCount=12,
        outputTokenCount=34,
    )
    tool = BaseApiSpan(
        uuid="span-tool",
        name="search",
        status=TraceSpanApiStatus.ERRORED,
        type=SpanApiType.TOOL,
        parentUuid="span-root",
        startTime="2026-01-01T00:00:01",
        endTime="2026-01-01T00:00:01",
        error="boom",
    )
    retriever = BaseApiSpan(
        uuid="span-ret",
        name="retriever",
        status=TraceSpanApiStatus.SUCCESS,
        type=SpanApiType.RETRIEVER,
        parentUuid="span-root",
        startTime="2026-01-01T00:00:01",
        endTime="2026-01-01T00:00:01",
        topK=3,
    )
    base = BaseApiSpan(
        uuid="span-base",
        name="helper",
        status=TraceSpanApiStatus.SUCCESS,
        type=SpanApiType.BASE,
        parentUuid="span-llm",
        startTime="2026-01-01T00:00:01",
        endTime="2026-01-01T00:00:01",
    )
    return TraceApi(
        uuid=uuid,
        name="my-agent",
        startTime="2026-01-01T00:00:00",
        endTime="2026-01-01T00:00:03",
        threadId="thread-1",
        userId="user-1",
        environment="testing",
        input="hi",
        output="done",
        tags=["a", "b"],
        metadata={"k": "v"},
        agentSpans=[root],
        llmSpans=[llm],
        toolSpans=[tool],
        retrieverSpans=[retriever],
        baseSpans=[base],
        metricsData=[_metric("TraceMetric", 0.7)],
    )


def _make_test_run(
    hyperparameters: Optional[dict] = None,
    identifier: Optional[str] = None,
    with_cases: bool = True,
) -> _TestRun:
    run = _TestRun(
        identifier=identifier,
        testFile="tests/test_foo.py",
        testCases=[],
        metricsScores=[],
        hyperparameters=hyperparameters,
        testPassed=None,
        testFailed=None,
    )
    if not with_cases:
        return run

    llm_case = LLMApiTestCase(
        name="case-0",
        input="What is 1+1?",
        actualOutput="2",
        expectedOutput="2",
        order=0,
        tags=["math"],
        success=True,
        runDuration=1.5,
        evaluationCost=0.01,
        metricsData=[
            _metric("AnswerRelevancy", 0.95),
            _metric("Faithfulness", 0.3, success=False),
        ],
        trace=_trace(),
    )
    conv_case = ConversationalApiTestCase(
        name="conv-0",
        success=True,
        order=1,
        scenario="Book a flight",
        expectedOutcome="Flight booked",
        metricsData=[_metric("TurnRelevancy", 0.8)],
        turns=[
            TurnApi(role="user", content="book me a flight", order=0),
            TurnApi(role="assistant", content="sure", order=1),
        ],
    )
    run.add_test_case(llm_case)
    run.add_test_case(conv_case)
    run.test_passed = 2
    run.test_failed = 0
    run.run_duration = 3.25
    return run


def _q(db_path: Path, sql: str, *params):
    with closing(sqlite3.connect(str(db_path))) as conn:
        return conn.execute(sql, params).fetchall()


# --------------------------------------------------------------------------- #


class TestResolveLocalStoreMode:
    def test_default_is_json(self, monkeypatch):
        monkeypatch.delenv("DEEPEVAL_LOCAL_STORE", raising=False)
        assert resolve_local_store_mode() == LOCAL_STORE_JSON

    @pytest.mark.parametrize("raw", ["sqlite", "SQLite", " sqlite3 ", "db"])
    def test_sqlite_aliases(self, monkeypatch, raw):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", raw)
        assert resolve_local_store_mode() == LOCAL_STORE_SQLITE

    def test_unknown_falls_back_to_json(self, monkeypatch, capsys):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "postgres")
        assert resolve_local_store_mode() == LOCAL_STORE_JSON
        assert "DEEPEVAL_LOCAL_STORE" in capsys.readouterr().err


class TestResolveDbPath:
    def test_defaults_to_hidden_dir(self, monkeypatch):
        monkeypatch.delenv("DEEPEVAL_RESULTS_FOLDER", raising=False)
        from deepeval.constants import HIDDEN_DIR

        assert (
            sqlite_store.resolve_db_path() == Path(HIDDEN_DIR) / "deepeval.db"
        )

    def test_results_folder_and_subfolder(self, tmp_path: Path):
        got = sqlite_store.resolve_db_path(str(tmp_path), "sweep")
        assert got == tmp_path / "sweep" / "deepeval.db"

    def test_env_var_fallback(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_RESULTS_FOLDER", str(tmp_path))
        assert sqlite_store.resolve_db_path() == tmp_path / "deepeval.db"


class TestSourceSpec:
    def test_parse_and_format(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        assert sqlite_store.parse_source(db) == (db, None)
        assert sqlite_store.parse_source(f"{db}#7") == (db, 7)
        assert sqlite_store.parse_source(f"{db}#") == (db, None)
        assert sqlite_store.format_source(db, 7) == f"{db}#7"

    def test_non_db_paths_are_none(self, tmp_path: Path):
        assert sqlite_store.parse_source(tmp_path / "test_run_1.json") is None
        assert sqlite_store.parse_source("experiments") is None

    def test_run_id_from_path(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        assert run_id_from_path(f"{db}#3") == "deepeval.db#3"
        assert run_id_from_path(db) == "deepeval.db"
        assert run_id_from_path(tmp_path / "test_run_x.json") == "test_run_x"


class TestSchema:
    def test_creates_tables_and_user_version(self, tmp_path: Path):
        db = tmp_path / "nested" / "deepeval.db"
        with closing(sqlite_store.connect(db)):
            pass
        assert db.is_file()
        tables = {
            r[0]
            for r in _q(db, "SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "test_runs",
            "test_cases",
            "traces",
            "spans",
            "metric_data",
        } <= tables
        assert (
            _q(db, "PRAGMA user_version")[0][0] == sqlite_store.SCHEMA_VERSION
        )

    def test_reopen_is_idempotent(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(_make_test_run(with_cases=False), db)
        sqlite_store.write_test_run(_make_test_run(with_cases=False), db)
        assert _q(db, "SELECT count(*) FROM test_runs")[0][0] == 2

    def test_newer_schema_is_rejected(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        with closing(sqlite3.connect(str(db))) as conn:
            conn.execute(
                f"PRAGMA user_version = {sqlite_store.SCHEMA_VERSION + 1}"
            )
        with pytest.raises(sqlite3.OperationalError):
            sqlite_store.connect(db)


_PROBE_MIGRATION = "ALTER TABLE test_runs ADD COLUMN _probe TEXT;"


def _columns(db: Path, table: str) -> set:
    with closing(sqlite3.connect(str(db))) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _user_version(db: Path) -> int:
    with closing(sqlite3.connect(str(db))) as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


class TestSchemaUpgrade:
    """Simulate a future release by patching SCHEMA_VERSION=2 + one ALTER step."""

    @pytest.fixture
    def future_release(self, monkeypatch):
        """Patch the module the way a release with one more migration would."""

        def activate():
            store = sqlite_store.store
            monkeypatch.setattr(store, "SCHEMA_VERSION", 2)
            monkeypatch.setitem(store._MIGRATIONS, 2, _PROBE_MIGRATION)
            return store

        return activate

    def test_existing_v1_file_gets_only_step_2(
        self, tmp_path: Path, future_release
    ):
        db = tmp_path / "deepeval.db"
        # A genuine v1 file with data, written by "today's" release.
        sqlite_store.write_test_run(_make_test_run(with_cases=False), db)
        assert _user_version(db) == 1
        assert "_probe" not in _columns(db, "test_runs")

        store = future_release()
        # Step 1 must NOT re-run (harmless here thanks to IF NOT EXISTS, but
        # the loop must start at version + 1). Prove it by making step 1 blow
        # up if executed.
        store._MIGRATIONS[1] = "SELECT RAISE(ABORT, 'step 1 re-ran');"
        try:
            sqlite_store.connect(db).close()
        finally:
            store._MIGRATIONS[1] = sqlite_store.SCHEMA_SQL

        assert _user_version(db) == 2
        assert "_probe" in _columns(db, "test_runs")
        # Existing data survives the in-place upgrade.
        assert len(sqlite_store.list_test_runs(db)) == 1

    def test_second_connect_is_a_noop(self, tmp_path: Path, future_release):
        db = tmp_path / "deepeval.db"
        future_release()
        sqlite_store.connect(db).close()
        assert _user_version(db) == 2
        # A second ALTER of the same column would fail if the loop re-ran.
        sqlite_store.connect(db).close()
        assert _user_version(db) == 2

    def test_fresh_file_applies_all_steps(self, tmp_path: Path, future_release):
        db = tmp_path / "deepeval.db"
        future_release()
        sqlite_store.connect(db).close()
        assert _user_version(db) == 2
        cols = _columns(db, "test_runs")
        assert {"id", "payload_json", "_probe"} <= cols

    def test_failed_step_rolls_back_and_keeps_old_version(
        self, tmp_path: Path, future_release
    ):
        db = tmp_path / "deepeval.db"
        sqlite_store.connect(db).close()  # v1 file
        store = future_release()
        store._MIGRATIONS[2] = (
            "ALTER TABLE test_runs ADD COLUMN _probe TEXT;"
            "ALTER TABLE no_such_table ADD COLUMN x TEXT;"
        )
        with pytest.raises(sqlite3.OperationalError):
            sqlite_store.connect(db)
        assert _user_version(db) == 1
        assert "_probe" not in _columns(db, "test_runs")


class TestWriteTestRun:
    def test_round_trip_payload_equals_model_dump(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        run = _make_test_run(
            hyperparameters={"model": "x", "t": 0.2}, identifier="base"
        )

        run_id = sqlite_store.write_test_run(run, db)
        assert run_id == 1

        loaded_id, payload = sqlite_store.load_test_run_payload(db, run_id)
        assert loaded_id == run_id
        expected = json.loads(
            json.dumps(
                run.model_dump(by_alias=True, exclude_none=True),
                cls=TestRunEncoder,
            )
        )
        assert payload == expected

        _, reloaded = sqlite_store.load_test_run(db, run_id)
        assert reloaded.identifier == "base"
        assert reloaded.hyperparameters == {"model": "x", "t": 0.2}
        assert len(reloaded.test_cases) == 1
        assert len(reloaded.conversational_test_cases) == 1
        assert reloaded.test_cases[0].trace.uuid == "trace-1"

    @staticmethod
    def _row_json_counts(db: Path):
        return tuple(
            _q(db, f"SELECT count(payload_json) FROM {t}")[0][0]
            for t in ("test_cases", "traces", "spans")
        )

    def test_row_json_off_by_default(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv(sqlite_store.INCLUDE_ROW_JSON_ENV_VAR, raising=False)
        db = tmp_path / "deepeval.db"
        run_id = sqlite_store.write_test_run(_make_test_run(), db)

        assert self._row_json_counts(db) == (0, 0, 0)
        # The run row always keeps its payload: reload still works.
        assert sqlite_store.load_test_run(db, run_id)[1].test_cases
        # Promoted columns are unaffected.
        assert _q(db, "SELECT count(*) FROM spans")[0][0] > 0

    @pytest.mark.parametrize("raw", ["1", "true", "YES", " on "])
    def test_row_json_via_env(self, tmp_path: Path, monkeypatch, raw):
        monkeypatch.setenv(sqlite_store.INCLUDE_ROW_JSON_ENV_VAR, raw)
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(_make_test_run(), db)

        cases, traces, spans = self._row_json_counts(db)
        assert cases == _q(db, "SELECT count(*) FROM test_cases")[0][0] > 0
        assert traces == _q(db, "SELECT count(*) FROM traces")[0][0] > 0
        assert spans == _q(db, "SELECT count(*) FROM spans")[0][0] > 0
        uuid, payload = _q(db, "SELECT uuid, payload_json FROM traces LIMIT 1")[
            0
        ]
        assert json.loads(payload)["uuid"] == uuid

    @pytest.mark.parametrize("raw", ["0", "false", "no", ""])
    def test_row_json_env_falsy(self, tmp_path: Path, monkeypatch, raw):
        monkeypatch.setenv(sqlite_store.INCLUDE_ROW_JSON_ENV_VAR, raw)
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(_make_test_run(), db)
        assert self._row_json_counts(db) == (0, 0, 0)

    def test_row_json_argument_overrides_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv(sqlite_store.INCLUDE_ROW_JSON_ENV_VAR, "1")
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(
            _make_test_run(), db, include_row_json=False
        )
        assert self._row_json_counts(db) == (0, 0, 0)

        monkeypatch.delenv(sqlite_store.INCLUDE_ROW_JSON_ENV_VAR)
        sqlite_store.write_test_run(_make_test_run(), db, include_row_json=True)
        assert all(n > 0 for n in self._row_json_counts(db))

    def test_confident_test_run_id_null_until_set(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        run_id = sqlite_store.write_test_run(_make_test_run(), db)
        assert _q(db, "SELECT confident_test_run_id FROM test_runs")[0] == (
            None,
        )
        assert (
            sqlite_store.list_test_runs(db)[0]["confident_test_run_id"] is None
        )

        sqlite_store.set_confident_test_run_id(db, run_id, "cai_abc123")
        assert _q(db, "SELECT confident_test_run_id FROM test_runs")[0] == (
            "cai_abc123",
        )
        assert (
            sqlite_store.list_test_runs(db)[0]["confident_test_run_id"]
            == "cai_abc123"
        )
        # Payload is untouched: the id lives in the column only.
        _, payload = sqlite_store.load_test_run_payload(db, run_id)
        assert "confident_test_run_id" not in payload

    def test_set_confident_test_run_id_unknown_run(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(_make_test_run(), db)
        with pytest.raises(LookupError):
            sqlite_store.set_confident_test_run_id(db, 999, "cai_x")

    def test_normalized_test_run_columns(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(
            _make_test_run(hyperparameters={"t": 1}, identifier="ident"), db
        )
        (row,) = _q(
            db,
            "SELECT identifier, test_file, test_passed, test_failed, "
            "run_duration, hyperparameters_json, created_at FROM test_runs",
        )
        assert row[0] == "ident"
        assert row[1] == "tests/test_foo.py"
        assert row[2] == 2 and row[3] == 0
        assert row[4] == pytest.approx(3.25)
        assert json.loads(row[5]) == {"t": 1}
        assert row[6]  # ISO timestamp

    def test_test_cases_rows(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(_make_test_run(), db)
        rows = _q(
            db,
            'SELECT kind, "order", name, input, actual_output, expected_output, '
            'success, tags_json FROM test_cases ORDER BY "order"',
        )
        assert rows[0][:7] == (
            "single-turn",
            0,
            "case-0",
            "What is 1+1?",
            "2",
            "2",
            1,
        )
        assert json.loads(rows[0][7]) == ["math"]
        assert rows[1][:6] == (
            "multi-turn",
            1,
            "conv-0",
            "Book a flight",
            None,
            "Flight booked",
        )

    def test_traces_and_spans_flattened(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(_make_test_run(), db)

        (trace,) = _q(
            db,
            "SELECT uuid, name, thread_id, user_id, environment, tags_json, "
            "test_case_id FROM traces",
        )
        assert trace[:5] == (
            "trace-1",
            "my-agent",
            "thread-1",
            "user-1",
            "testing",
        )
        assert json.loads(trace[5]) == ["a", "b"]
        (case_id,) = _q(
            db, "SELECT id FROM test_cases WHERE kind='single-turn'"
        )[0]
        assert trace[6] == case_id

        spans = _q(
            db,
            "SELECT uuid, parent_uuid, type, model, provider, "
            "input_token_count, output_token_count, error, status FROM spans "
            "ORDER BY uuid",
        )
        by_uuid = {s[0]: s for s in spans}
        assert set(by_uuid) == {
            "span-root",
            "span-llm",
            "span-tool",
            "span-ret",
            "span-base",
        }
        assert by_uuid["span-root"][1] is None
        assert by_uuid["span-root"][2] == "agent"
        assert by_uuid["span-llm"][1:7] == (
            "span-root",
            "llm",
            "gpt-4o-mini",
            "openai",
            12.0,
            34.0,
        )
        assert by_uuid["span-tool"][7:] == ("boom", "ERRORED")
        assert by_uuid["span-base"][1] == "span-llm"

    def test_metrics_rows_for_all_owner_types(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        run_id = sqlite_store.write_test_run(_make_test_run(), db)

        rows = _q(
            db,
            "SELECT owner_type, name, score, success, reason, evaluation_model, "
            "evaluation_cost, input_tokens, output_tokens FROM metric_data "
            "WHERE test_run_id = ? ORDER BY owner_type, name",
            run_id,
        )
        by_key = {(r[0], r[1]): r for r in rows}
        assert set(by_key) == {
            ("test_case", "AnswerRelevancy"),
            ("test_case", "Faithfulness"),
            ("test_case", "TurnRelevancy"),
            ("trace", "TraceMetric"),
            ("span", "TaskCompletion"),
        }
        faith = by_key[("test_case", "Faithfulness")]
        assert faith[2] == pytest.approx(0.3)
        assert faith[3] == 0
        assert faith[4] == "Faithfulness reason"
        assert faith[5] == "gpt-4o-mini"
        assert faith[6] == pytest.approx(0.001)
        assert faith[7:] == (10, 5)

        # owner_id points at the right span row
        (span_owner_id,) = _q(
            db, "SELECT owner_id FROM metric_data WHERE owner_type='span'"
        )[0]
        (span_uuid,) = _q(
            db, "SELECT uuid FROM spans WHERE id = ?", span_owner_id
        )[0]
        assert span_uuid == "span-root"

    def test_cross_run_sql_query(self, tmp_path: Path):
        """The whole point: aggregate a metric across runs with plain SQL."""
        db = tmp_path / "deepeval.db"
        for _ in range(3):
            sqlite_store.write_test_run(_make_test_run(), db)
        rows = _q(
            db,
            "SELECT r.id, avg(m.score) FROM metric_data m "
            "JOIN test_runs r ON r.id = m.test_run_id "
            "WHERE m.name = 'AnswerRelevancy' GROUP BY r.id ORDER BY r.id",
        )
        assert [r[0] for r in rows] == [1, 2, 3]
        assert all(r[1] == pytest.approx(0.95) for r in rows)

    def test_concurrent_writes(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        n = 8
        errors = []

        def worker(i: int):
            try:
                sqlite_store.write_test_run(
                    _make_test_run(hyperparameters={"i": i}, with_cases=False),
                    db,
                )
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert _q(db, "SELECT count(*) FROM test_runs")[0][0] == n
        seen = {
            json.loads(r[0])["i"]
            for r in _q(db, "SELECT hyperparameters_json FROM test_runs")
        }
        assert seen == set(range(n))


class TestRead:
    def test_list_and_latest(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        assert sqlite_store.list_test_runs(db) == []
        assert sqlite_store.latest_run_id(db) is None

        for ident in ("a", "b", "c"):
            sqlite_store.write_test_run(
                _make_test_run(identifier=ident, with_cases=False), db
            )

        runs = sqlite_store.list_test_runs(db)
        assert [r["identifier"] for r in runs] == ["c", "b", "a"]
        assert sqlite_store.latest_run_id(db) == 3
        assert sqlite_store.list_test_runs(db, limit=1)[0]["id"] == 3

        latest_id, payload = sqlite_store.load_test_run_payload(db)
        assert latest_id == 3 and payload["identifier"] == "c"

    def test_missing_db_and_run(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        with pytest.raises(FileNotFoundError):
            sqlite_store.load_test_run_payload(db)
        sqlite_store.write_test_run(_make_test_run(with_cases=False), db)
        with pytest.raises(LookupError):
            sqlite_store.load_test_run_payload(db, 99)


class TestInspectLoaderParity:
    def test_traces_identical_from_json_and_sqlite(self, tmp_path: Path):
        run = _make_test_run()
        json_path = write_json_test_run(tmp_path / "json", run)
        db = tmp_path / "deepeval.db"
        run_id = sqlite_store.write_test_run(run, db)

        from_json = load_test_run(json_path)
        from_db = load_test_run(sqlite_store.format_source(db, run_id))
        from_db_latest = load_test_run(db)

        dump = lambda traces: [t.model_dump() for t in traces]  # noqa: E731
        assert dump(from_json) == dump(from_db) == dump(from_db_latest)
        assert from_db[0].uuid == "trace-1"
        assert {s.uuid for s in from_db[0].root_spans} == {"span-root"}

    def test_summary_from_sqlite(self, tmp_path: Path):
        db = tmp_path / "deepeval.db"
        run_id = sqlite_store.write_test_run(_make_test_run(), db)
        summary = summarize_test_run(sqlite_store.format_source(db, run_id))
        assert summary == {
            "test_passed": 2,
            "test_failed": 0,
            "run_duration": 3.25,
            "evaluation_cost": pytest.approx(0.01),
        }

    def test_bad_sources_raise_inspect_load_error(self, tmp_path: Path):
        with pytest.raises(InspectLoadError):
            load_test_run(tmp_path / "nope.db")
        db = tmp_path / "deepeval.db"
        sqlite_store.write_test_run(_make_test_run(with_cases=False), db)
        with pytest.raises(InspectLoadError):
            load_test_run(f"{db}#42")


class TestTestRunManagerIntegration:
    def test_json_mode_is_unchanged(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "json")
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(results_folder=str(tmp_path))

        mgr.save_test_run_locally()

        assert len(list(tmp_path.glob("test_run_*.json"))) == 1
        assert not (tmp_path / "deepeval.db").exists()
        assert mgr.last_saved_run_id is None

    def test_sqlite_mode_writes_db_not_json(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "sqlite")
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(results_folder=str(tmp_path))

        mgr.save_test_run_locally()

        db = tmp_path / "deepeval.db"
        assert db.is_file()
        assert list(tmp_path.glob("test_run_*.json")) == []
        assert mgr.last_saved_path == db
        assert mgr.last_saved_run_id == 1
        assert _q(db, "SELECT count(*) FROM test_runs")[0][0] == 1

    def test_sqlite_mode_subfolder_and_env_fallback(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "sqlite")
        monkeypatch.setenv("DEEPEVAL_RESULTS_FOLDER", str(tmp_path))

        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.save_test_run_locally()
        assert (tmp_path / "deepeval.db").is_file()

        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(
            results_folder=str(tmp_path), results_subfolder="sweep"
        )
        mgr.save_test_run_locally()
        assert (tmp_path / "sweep" / "deepeval.db").is_file()

    def test_sqlite_mode_appends_across_runs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "sqlite")
        for temp in [0.0, 0.4, 0.8]:
            mgr = _TestRunManager()
            mgr.set_test_run(
                _make_test_run(
                    hyperparameters={"temperature": temp}, with_cases=False
                )
            )
            mgr.configure_local_store(results_folder=str(tmp_path))
            mgr.save_test_run_locally()

        db = tmp_path / "deepeval.db"
        temps = sorted(
            json.loads(r[0])["temperature"]
            for r in _q(db, "SELECT hyperparameters_json FROM test_runs")
        )
        assert temps == [0.0, 0.4, 0.8]
        assert mgr.last_saved_run_id == 3

    def test_confident_id_stamped_after_upload(
        self, tmp_path: Path, monkeypatch
    ):
        """Logged-in + sqlite: the row is written first, then stamped with the
        id Confident AI returned, so upload failures never lose the local copy.
        """
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "sqlite")
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(results_folder=str(tmp_path))
        mgr.save_test_run_locally()
        db = tmp_path / "deepeval.db"
        assert _q(db, "SELECT confident_test_run_id FROM test_runs")[0] == (
            None,
        )

        mgr._record_confident_test_run_id("cai_from_post")
        assert _q(db, "SELECT confident_test_run_id FROM test_runs")[0] == (
            "cai_from_post",
        )

    def test_confident_id_noop_in_json_mode(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "json")
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(results_folder=str(tmp_path))
        mgr.save_test_run_locally()
        assert mgr.last_saved_run_id is None
        mgr._record_confident_test_run_id("cai_x")  # must not raise
        assert not (tmp_path / "deepeval.db").exists()

    def test_confident_id_failure_is_warning(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "sqlite")
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(results_folder=str(tmp_path))
        mgr.save_test_run_locally()
        mgr.last_saved_run_id = 999  # row that does not exist
        mgr._record_confident_test_run_id("cai_x")  # must not raise
        assert (
            "could not record Confident AI test run id"
            in capsys.readouterr().err
        )

    def test_read_only_env_is_noop(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "sqlite")
        monkeypatch.setattr(
            "deepeval.test_run.test_run.is_read_only_env", lambda: True
        )
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(results_folder=str(tmp_path))
        mgr.save_test_run_locally()
        assert not (tmp_path / "deepeval.db").exists()
        assert mgr.last_saved_path is None

    def test_storage_failure_is_a_warning_not_an_error(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "sqlite")

        def boom(*_a, **_k):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(sqlite_store, "write_test_run", boom)
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(with_cases=False))
        mgr.configure_local_store(results_folder=str(tmp_path))

        mgr.save_test_run_locally()  # must not raise

        assert "disk I/O error" in capsys.readouterr().err
        assert mgr.last_saved_path is None


# --------------------------------------------------------------------------- #
# Schema parity with the shared `sqlite/schema.sql`
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_SCHEMA = _REPO_ROOT / "sqlite" / "schema.sql"
_MIGRATIONS_DIR = _REPO_ROOT / "sqlite" / "migrations"
_DOCS_PAGE = (
    _REPO_ROOT
    / "docs"
    / "content"
    / "docs"
    / "evaluation-local-backend-storage.mdx"
)


def _normalize_sql(sql: str) -> str:
    """Strip `--` comments and collapse whitespace so formatting can't fail us."""
    import re

    sql = re.sub(r"--[^\n]*", "", sql)
    return re.sub(r"\s+", " ", sql).strip()


@pytest.mark.skipif(
    not _SHARED_SCHEMA.exists(), reason="repo checkout only (sqlite/schema.sql)"
)
class TestSchemaParity:
    def test_embedded_schema_matches_shared_file(self):
        shared = _normalize_sql(_SHARED_SCHEMA.read_text(encoding="utf-8"))
        embedded = _normalize_sql(sqlite_store.SCHEMA_SQL)
        assert embedded == shared, (
            "deepeval/sqlite_store/store.py::_SCHEMA has drifted from "
            "sqlite/schema.sql; update both (and SCHEMA_VERSION) together."
        )

    def test_each_migration_matches_its_shared_file(self):
        """`_MIGRATIONS[n]` is the embedded copy of `sqlite/migrations/000n_*.sql`."""
        assert set(sqlite_store.MIGRATIONS) == set(
            range(1, sqlite_store.SCHEMA_VERSION + 1)
        )
        for version, embedded in sqlite_store.MIGRATIONS.items():
            matches = sorted(_MIGRATIONS_DIR.glob(f"{version:04d}_*.sql"))
            assert len(matches) == 1, (
                f"expected exactly one sqlite/migrations/{version:04d}_*.sql, "
                f"got {[m.name for m in matches]}"
            )
            assert _normalize_sql(embedded) == _normalize_sql(
                matches[0].read_text(encoding="utf-8")
            ), f"_MIGRATIONS[{version}] has drifted from {matches[0].name}"

    def test_no_orphan_migration_files(self):
        on_disk = sorted(
            int(p.name[:4])
            for p in _MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql")
        )
        assert on_disk == sorted(sqlite_store.MIGRATIONS)

    def test_migrations_in_order_equal_full_schema(self):
        """Applying every migration to an empty DB == applying schema.sql."""

        def objects(sql_scripts):
            with closing(sqlite3.connect(":memory:")) as conn:
                for script in sql_scripts:
                    conn.executescript(script)
                rows = conn.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE sql IS NOT NULL ORDER BY type, name"
                ).fetchall()
            return [(t, n, _normalize_sql(s)) for t, n, s in rows]

        migrated = objects(
            p.read_text(encoding="utf-8")
            for p in sorted(_MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql"))
        )
        snapshot = objects([_SHARED_SCHEMA.read_text(encoding="utf-8")])
        assert migrated == snapshot, (
            "sqlite/schema.sql is stale: it must equal the result of applying "
            "sqlite/migrations/*.sql in order."
        )

    def test_shared_schema_is_valid_sqlite(self):
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.executescript(_SHARED_SCHEMA.read_text(encoding="utf-8"))
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert tables >= {
            "test_runs",
            "test_cases",
            "traces",
            "spans",
            "metric_data",
        }

    @pytest.mark.skipif(
        not _DOCS_PAGE.exists(), reason="docs page not in this checkout"
    )
    def test_docs_ddl_matches_shared_file(self):
        """Every CREATE TABLE shown in the docs equals the real one.

        The docs drop `IF NOT EXISTS`, omit the indexes and add `--` column
        comments, so compare table by table after normalising those away.
        """
        import re

        def tables(sql: str):
            out = {}
            for m in re.finditer(
                r"CREATE TABLE (?:IF NOT EXISTS )?(\w+)\s*\((.*?)\);", sql, re.S
            ):
                out[m.group(1)] = _normalize_sql(m.group(2))
            return out

        shared = tables(_SHARED_SCHEMA.read_text(encoding="utf-8"))
        doc_sql = "\n".join(
            re.findall(r"```sql\n(.*?)```", _DOCS_PAGE.read_text("utf-8"), re.S)
        )
        documented = tables(doc_sql)
        assert set(documented) == set(shared)
        for name, body in shared.items():
            assert documented[name] == body, f"docs DDL for `{name}` is stale"
