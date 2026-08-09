import json
import re
from pathlib import Path

import pytest

import deepeval.telemetry as telemetry
from deepeval.telemetry import (
    Entrypoint,
    Event,
    EventProperties,
    FlushReason,
    Outcome,
    Runtime,
    TelemetryKey,
    UserStatus,
)
from deepeval.telemetry import context as context_mod
from deepeval.telemetry import identity as identity_mod


class FakeBackend:
    """Stands in for PostHog so tests assert on the payload, not the wire."""

    def __init__(self):
        self.events = []

    def capture(self, anonymous_id, event, properties):
        self.events.append((anonymous_id, event, properties))

    def flush(self):
        pass

    def only(self):
        assert len(self.events) == 1, f"expected 1 event, got {self.events}"
        return self.events[0][2]


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
    monkeypatch.setenv("DEEPEVAL_HOME", str(tmp_path / "home"))
    identity_mod.reset_cache_for_testing()
    context_mod.reset_for_testing()

    fake = FakeBackend()
    telemetry.set_backend(fake)
    yield fake

    identity_mod.reset_cache_for_testing()
    context_mod.reset_for_testing()


class TestIdentity:
    def test_id_is_written_to_the_home_directory(self, backend, tmp_path):
        unique_id = telemetry.get_unique_id()

        assert unique_id
        stored = (tmp_path / "home" / telemetry.TELEMETRY_DATA_FILE).read_text()
        assert f"{TelemetryKey.ID.value}={unique_id}" in stored

    def test_id_survives_a_change_of_working_directory(
        self, backend, tmp_path, monkeypatch
    ):
        """The whole point of the move: a second project folder is the same
        user, where the old CWD-relative store minted a new id."""
        first = telemetry.get_unique_id()

        identity_mod.reset_cache_for_testing()
        monkeypatch.chdir(tmp_path)

        assert telemetry.get_unique_id() == first

    def test_a_legacy_cwd_store_is_migrated_once(
        self, backend, tmp_path, monkeypatch
    ):
        legacy_dir = tmp_path / "project" / ".deepeval"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / telemetry.TELEMETRY_DATA_FILE).write_text(
            f"{TelemetryKey.ID.value}=legacy-id-1234\n"
        )
        monkeypatch.chdir(tmp_path / "project")
        identity_mod.reset_cache_for_testing()

        assert telemetry.get_unique_id() == "legacy-id-1234"

    def test_only_the_first_event_of_a_fresh_install_reports_new(self, backend):
        """Regression: `get_status()` used to run before `get_unique_id()` had
        written the flag, so the first two events both said `new`."""
        assert telemetry.get_identity().status is UserStatus.NEW

        identity_mod.reset_cache_for_testing()
        assert telemetry.get_identity().status is UserStatus.OLD

    def test_an_email_is_stored_locally_but_never_transmitted(self, backend):
        """The privacy page says no PII, so only the boolean goes out."""
        telemetry.set_logged_in_with("someone@example.com")

        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            pass

        props = backend.only()
        assert props["user.logged_in"] is True
        assert "someone@example.com" not in json.dumps(props)
        assert telemetry.get_logged_in_with() == "someone@example.com"

    def test_legacy_per_feature_keys_are_not_reported_as_new(
        self, backend, tmp_path
    ):
        """Pre-v2 wrote one key per feature; reading only the new list made
        every existing user look like a first-time user of everything."""
        store = tmp_path / "home" / telemetry.TELEMETRY_DATA_FILE
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(
            f"{TelemetryKey.ID.value}=abc\n" "DEEPEVAL_EVALUATION_STATUS=old\n"
        )
        identity_mod.reset_cache_for_testing()

        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            pass

        assert backend.only()["feature.status"] == UserStatus.OLD.value
        assert "DEEPEVAL_EVALUATION_STATUS" not in store.read_text()


class TestEvaluationEvent:
    def test_a_run_emits_one_event_carrying_its_totals(self, backend):
        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            for _ in range(5):
                telemetry.record_test_case()
                for name in ("AnswerRelevancy", "Faithfulness", "Bias"):
                    telemetry.record_metric(
                        name, async_mode=True, in_component=False
                    )

        props = backend.only()
        assert backend.events[0][1] is Event.EVALUATION
        assert props["eval.entrypoint"] == Entrypoint.EVALUATE.value
        assert props["eval.test_case_count"] == 5
        assert props["eval.metric_runs"] == 15
        assert props["eval.metrics_count"] == 3
        assert props["eval.outcome"] == Outcome.COMPLETED.value

    def test_every_event_carries_the_schema_version(self, backend):
        with telemetry.capture_evaluation_run(Entrypoint.COMPARE):
            pass

        assert backend.only()["telemetry.schema_version"] == 2

    def test_nested_runs_attribute_metrics_to_the_innermost(self, backend):
        with telemetry.capture_evaluation_run(Entrypoint.PYTEST):
            with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
                telemetry.record_metric(
                    "Bias", async_mode=False, in_component=False
                )

        inner, outer = backend.events[0][2], backend.events[1][2]
        assert inner["eval.entrypoint"] == Entrypoint.EVALUATE.value
        assert inner["eval.metric_runs"] == 1
        assert outer["eval.entrypoint"] == Entrypoint.PYTEST.value
        assert outer["eval.metric_runs"] == 0

    def test_no_vendor_reserved_keys_are_emitted(self, backend):
        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            telemetry.record_test_case()

        assert [k for k in backend.only() if k.startswith("$")] == []


class TestRunId:
    def test_each_run_gets_its_own_id(self, backend):
        for _ in range(2):
            with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
                pass

        first, second = (event[2]["eval.run_id"] for event in backend.events)
        assert first and second and first != second

    def test_processes_sharing_an_id_report_as_one_run(self, backend):
        """What makes `pytest -n` one run rather than one per worker: the
        workers cannot share counters, only the id."""
        shared = "11111111-2222-3333-4444-555555555555"

        for _ in range(4):
            with telemetry.capture_evaluation_run(
                Entrypoint.PYTEST, run_id=shared
            ):
                telemetry.record_test_case()

        assert len(backend.events) == 4
        assert {event[2]["eval.run_id"] for event in backend.events} == {shared}
        assert sum(e[2]["eval.test_case_count"] for e in backend.events) == 4

    def test_an_empty_scope_can_stay_silent(self, backend):
        """A pytest session with no deepeval tests is not an evaluation, and
        the plugin loads in every suite on the machine."""
        with telemetry.capture_evaluation_run(
            Entrypoint.PYTEST, skip_if_empty=True
        ):
            pass

        assert backend.events == []

    def test_a_scope_with_work_still_reports_when_skipping_empties(
        self, backend
    ):
        with telemetry.capture_evaluation_run(
            Entrypoint.PYTEST, skip_if_empty=True
        ):
            telemetry.record_metric(
                "Bias", async_mode=False, in_component=False
            )

        assert backend.only()["eval.metric_runs"] == 1

    def test_skipping_empties_never_swallows_an_exception(self, backend):
        """`skip_if_empty` is evaluated in a `finally`, where an early return
        would discard the user's exception."""
        with pytest.raises(ValueError):
            with telemetry.capture_evaluation_run(
                Entrypoint.PYTEST, skip_if_empty=True
            ):
                raise ValueError("boom")

        assert backend.events == []


class TestOutcome:
    def test_a_failed_run_reports_the_exception_class(self, backend):
        with pytest.raises(ValueError):
            with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
                raise ValueError("sk-live-secret and a user prompt")

        props = backend.only()
        assert props["eval.outcome"] == Outcome.ERRORED.value
        assert props["eval.error_type"] == "ValueError"

    def test_an_exception_message_never_reaches_the_payload(self, backend):
        with pytest.raises(RuntimeError):
            with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
                raise RuntimeError("sk-live-secret and a user prompt")

        assert "sk-live-secret" not in json.dumps(backend.only())

    def test_ctrl_c_is_distinguishable_from_a_crash(self, backend):
        with pytest.raises(KeyboardInterrupt):
            with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
                raise KeyboardInterrupt

        assert backend.only()["eval.outcome"] == Outcome.INTERRUPTED.value


def shipped_model(class_name: str, model_name: str):
    """Stand-in for a model class the library ships.

    `judge.provider` is bounded by where the class is defined, so a stub has to
    claim a `deepeval.` module to be treated as one of ours.
    """
    model_class = type(class_name, (), {"name": model_name})
    model_class.__module__ = "deepeval.models.llms.openai_model"
    return model_class()


class TestJudgeModel:
    def test_a_known_model_is_reported(self, backend):
        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            telemetry.record_metric(
                "GEval",
                async_mode=False,
                in_component=False,
                model=shipped_model("OpenAIModel", "gpt-4o"),
            )

        props = backend.only()
        assert props["judge.provider"] == "OpenAIModel"
        assert props["judge.model"] == "gpt-4o"

    def test_a_self_hosted_model_name_cannot_leak(self, backend):
        class AcmeUnderwritingLLM:
            name = "acme-internal-underwriting-v3"

            def get_model_name(self):
                return self.name

        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            telemetry.record_metric(
                "GEval",
                async_mode=False,
                in_component=False,
                model=AcmeUnderwritingLLM(),
            )

        props = backend.only()
        assert props["judge.provider"] == "custom"
        assert props["judge.model"] == "other"
        assert "acme" not in json.dumps(props)

    def test_an_unknown_name_from_a_known_provider_becomes_other(self, backend):
        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            telemetry.record_metric(
                "GEval",
                async_mode=False,
                in_component=False,
                model=shipped_model("OpenAIModel", "gpt-internal-finetune-42"),
            )

        props = backend.only()
        assert props["judge.provider"] == "OpenAIModel"
        assert props["judge.model"] == "other"

    def test_a_subclass_of_a_shipped_model_is_not_treated_as_ours(
        self, backend
    ):
        """Subclassing is how a user-defined class would otherwise inherit a
        provider it did not write."""

        class InternalJudge(type(shipped_model("OpenAIModel", "gpt-4o"))):
            name = "gpt-4o"

        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            telemetry.record_metric(
                "GEval",
                async_mode=False,
                in_component=False,
                model=InternalJudge(),
            )

        assert backend.only()["judge.provider"] == "custom"


class TestStandaloneMetrics:
    def test_bare_measures_become_one_event_at_exit(self, backend):
        for _ in range(10):
            telemetry.record_metric(
                "AnswerRelevancy", async_mode=False, in_component=False
            )
        assert backend.events == []

        telemetry.flush_standalone_metrics()

        props = backend.only()
        assert props["eval.entrypoint"] == Entrypoint.STANDALONE.value
        assert props["eval.metric_runs"] == 10
        assert props["eval.flush_reason"] == FlushReason.MANUAL.value
        # Same shape as every other Evaluation event, so the counts can be
        # summed across entrypoints without special-casing standalone.
        assert props["eval.test_case_count"] == 0
        assert props["eval.golden_count"] == 0
        assert props["tracing.traced"] is False
        assert props["tracing.trace_count"] == 0

    def test_partial_flushes_sum_to_the_true_total(self, backend):
        for _ in range(120):
            telemetry.record_metric(
                "Bias", async_mode=False, in_component=False
            )
        telemetry.flush_standalone_metrics()

        totals = sum(p["eval.metric_runs"] for _, _, p in backend.events)
        assert totals == 120
        # Threshold flushes are partial sessions, so they must be marked as
        # such: counting these events would overstate the session count.
        assert any(
            p["eval.flush_reason"] == FlushReason.THRESHOLD.value
            for _, _, p in backend.events
        )

    def test_metrics_inside_a_run_do_not_reach_the_standalone_path(
        self, backend
    ):
        with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
            telemetry.record_metric(
                "Bias", async_mode=False, in_component=False
            )
        telemetry.flush_standalone_metrics()

        assert len(backend.events) == 1


class TestIntegrations:
    def test_an_integration_is_reported_once_per_process(self, backend):
        from deepeval.tracing.integrations import Integration

        for _ in range(3):
            with telemetry.capture_tracing_integration(Integration.LANGCHAIN):
                pass

        props = backend.only()
        assert backend.events[0][1] is Event.INTEGRATION_INSTALLED
        assert props["tracing.integration"] == Integration.LANGCHAIN.value


class TestOptOut:
    def test_nothing_is_emitted_when_opted_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
        monkeypatch.setenv("DEEPEVAL_HOME", str(tmp_path / "home"))
        identity_mod.reset_cache_for_testing()

        from deepeval.config.settings import reset_settings

        reset_settings()

        fake = FakeBackend()
        telemetry.set_backend(fake)
        try:
            with telemetry.capture_evaluation_run(Entrypoint.EVALUATE):
                telemetry.record_test_case()
            assert fake.events == []
            assert telemetry.get_unique_id() == "telemetry-opted-out"
        finally:
            monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
            reset_settings()
            identity_mod.reset_cache_for_testing()


class TestTypeSafety:
    def test_posthog_is_imported_and_called_in_exactly_one_place(self):
        """The guard that keeps the vendor swappable and stops capture sprawl.

        A new contributor adding a capture elsewhere is how the old sprawl
        happened, and this is also what keeps a vendor swap to one class.
        """
        root = Path(__file__).resolve().parents[2] / "deepeval"
        imports = re.compile(r"^\s*(from posthog import|import posthog)", re.M)
        calls = re.compile(r"\.capture\(\s*\n?\s*distinct_id=")

        importers, callers = [], []
        for path in root.rglob("*.py"):
            source = path.read_text()
            if imports.search(source):
                importers.append(path.relative_to(root).as_posix())
            if calls.search(source):
                callers.append(path.relative_to(root).as_posix())

        assert importers == ["telemetry/client.py"], importers
        assert callers == ["telemetry/client.py"], callers

    def test_every_property_key_is_namespaced(self):
        from deepeval.telemetry.properties import Prop

        assert all("." in prop.value for prop in Prop)
        assert all(prop.value.islower() for prop in Prop)

    def test_every_dataclass_field_maps_to_a_property_key(self):
        from dataclasses import fields

        from deepeval.telemetry.properties import _FIELD_TO_PROP

        assert {f.name for f in fields(EventProperties)} == set(_FIELD_TO_PROP)

    def test_every_registered_command_reports_itself(self, backend):
        """No list to drift: a command is valid because the CLI dispatches it.

        Registering `deepeval whatever` is the only step needed for it to be
        reported as `whatever`.
        """
        import typer

        from deepeval.cli.main import app

        dispatch_table = typer.main.get_command(app).commands
        assert {"test", "login", "view"} <= set(dispatch_table)

        for name in dispatch_table:
            backend.events.clear()
            telemetry.capture_cli_command(name, dispatch_table)
            assert backend.only()["cli.command"] == name

    def test_an_unregistered_command_falls_back_rather_than_escaping(
        self, backend
    ):
        telemetry.capture_cli_command("not-a-command", {"view": object()})
        assert backend.only()["cli.command"] == "unknown"

        backend.events.clear()
        telemetry.capture_cli_command(None, {"view": object()})
        assert backend.only()["cli.command"] == "unknown"

    def test_event_names_are_the_known_set(self):
        """Catches an accidental rename before it silently forks a series."""
        assert {event.value for event in Event} == {
            "Evaluation",
            "Synthesizer",
            "Conversation Simulator",
            "Benchmark",
            "Integration Installed",
            "CLI Command",
            "Login Prompt Shown",
            "Login",
        }


class TestRuntime:
    def test_github_actions_is_detected(self, monkeypatch):
        from deepeval.telemetry.runtime import detect_runtime

        detect_runtime.cache_clear()
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        try:
            assert detect_runtime() is Runtime.CI_GITHUB
        finally:
            detect_runtime.cache_clear()

    def test_a_plain_ci_variable_falls_back_to_ci_other(self, monkeypatch):
        from deepeval.telemetry.runtime import detect_runtime

        detect_runtime.cache_clear()
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("GITLAB_CI", raising=False)
        monkeypatch.setenv("CI", "1")
        try:
            assert detect_runtime() is Runtime.CI_OTHER
        finally:
            detect_runtime.cache_clear()

    def test_a_terminal_alone_does_not_mean_interactive(self, monkeypatch):
        """`python script.py` from a shell has a tty on stdin but is a script.

        Testing for a tty labelled every laptop run interactive and left
        `script` meaning little more than "stdin was piped".
        """
        import sys as sys_mod

        from deepeval.telemetry import runtime as runtime_mod

        for var in (
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            *runtime_mod._CI_VENDOR_VARS,
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(runtime_mod, "_in_container", lambda: False)
        monkeypatch.delattr(sys_mod, "ps1", raising=False)

        runtime_mod.detect_runtime.cache_clear()
        try:
            assert runtime_mod.detect_runtime() is Runtime.SCRIPT
        finally:
            runtime_mod.detect_runtime.cache_clear()

    def test_a_repl_prompt_is_interactive(self, monkeypatch):
        import sys as sys_mod

        from deepeval.telemetry import runtime as runtime_mod

        for var in (
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            *runtime_mod._CI_VENDOR_VARS,
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(runtime_mod, "_in_container", lambda: False)
        monkeypatch.setattr(sys_mod, "ps1", ">>> ", raising=False)

        runtime_mod.detect_runtime.cache_clear()
        try:
            assert runtime_mod.detect_runtime() is Runtime.INTERACTIVE
        finally:
            runtime_mod.detect_runtime.cache_clear()
