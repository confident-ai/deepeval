// Port of `tests/test_core/test_telemetry.py`, plus two guards Python has no
// need for: that every property key matches the artifact generated from
// Python's enums, and that `posthog-node` stays confined to one file.

import * as fs from "fs";
import * as os from "os";
import * as path from "path";

import { resetSettings } from "@/config/settings";
import { OpenAIModel } from "@/models";
import * as telemetry from "@/telemetry";
import * as context from "@/telemetry/context";
import * as identity from "@/telemetry/identity";
import * as runtime from "@/telemetry/runtime";
import {
  Entrypoint,
  Event,
  FlushReason,
  Outcome,
  Runtime,
  TelemetryKey,
  UserStatus,
  type EventProperties,
  type PropValue,
} from "@/telemetry";
import { Integration } from "@/tracing/integrations";

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

interface Captured {
  anonymousId: string;
  event: Event;
  properties: Record<string, PropValue>;
}

/** Stands in for PostHog so tests assert on the payload, not the wire. */
class FakeBackend implements telemetry.TelemetryBackend {
  readonly events: Captured[] = [];

  capture(
    anonymousId: string,
    event: Event,
    properties: Record<string, PropValue>,
  ): void {
    this.events.push({ anonymousId, event, properties });
  }

  flush(): void {}

  only(): Record<string, PropValue> {
    expect(this.events).toHaveLength(1);
    return this.events[0].properties;
  }
}

let backend: FakeBackend;
let home: string;
let originalCwd: string;
const savedEnv: Record<string, string | undefined> = {};

function setEnv(name: string, value: string | undefined): void {
  if (!(name in savedEnv)) savedEnv[name] = process.env[name];
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}

function resetTelemetryState(): void {
  identity.resetCacheForTesting();
  context.resetForTesting();
  runtime.resetCacheForTesting();
}

beforeEach(() => {
  originalCwd = process.cwd();
  home = fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-telemetry-"));
  setEnv("DEEPEVAL_TELEMETRY_OPT_OUT", undefined);
  setEnv("DEEPEVAL_TELEMETRY_ENABLED", undefined);
  setEnv("DEEPEVAL_HOME", path.join(home, "home"));
  resetSettings();
  resetTelemetryState();

  backend = new FakeBackend();
  telemetry.setBackend(backend);
});

afterEach(() => {
  process.chdir(originalCwd);
  telemetry.setBackend(null);
  for (const [name, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
    delete savedEnv[name];
  }
  resetSettings();
  resetTelemetryState();
  fs.rmSync(home, { recursive: true, force: true });
});

function storePath(): string {
  return path.join(home, "home", telemetry.TELEMETRY_DATA_FILE);
}

describe("identity", () => {
  it("writes the id to the home directory", () => {
    const uniqueId = telemetry.getUniqueId();

    expect(uniqueId).toBeTruthy();
    expect(fs.readFileSync(storePath(), "utf-8")).toContain(
      `${TelemetryKey.ID}=${uniqueId}`,
    );
  });

  it("keeps the id across a change of working directory", () => {
    // The point of the home directory: a second project folder is one user.
    const first = telemetry.getUniqueId();

    identity.resetCacheForTesting();
    process.chdir(home);

    expect(telemetry.getUniqueId()).toBe(first);
  });

  it("migrates a legacy cwd store once", () => {
    const project = path.join(home, "project");
    fs.mkdirSync(path.join(project, ".deepeval"), { recursive: true });
    fs.writeFileSync(
      path.join(project, ".deepeval", telemetry.TELEMETRY_DATA_FILE),
      `${TelemetryKey.ID}=legacy-id-1234\n`,
    );
    process.chdir(project);
    identity.resetCacheForTesting();

    expect(telemetry.getUniqueId()).toBe("legacy-id-1234");
    expect(fs.readFileSync(storePath(), "utf-8")).toContain("legacy-id-1234");
  });

  it("reports new for the first event of a fresh install only", () => {
    // Regression: the first two events of an install both said `new`.
    expect(telemetry.getIdentity().status).toBe(UserStatus.NEW);

    identity.resetCacheForTesting();
    expect(telemetry.getIdentity().status).toBe(UserStatus.OLD);
  });

  it("stores an email locally but never transmits it", () => {
    // The privacy page says no PII, so only the boolean goes out.
    telemetry.setLoggedInWith("someone@example.com");

    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {});

    const props = backend.only();
    expect(props["user.logged_in"]).toBe(true);
    expect(JSON.stringify(props)).not.toContain("someone@example.com");
    expect(telemetry.getLoggedInWith()).toBe("someone@example.com");
  });

  it("does not report a legacy per-feature key as new", () => {
    // Pre-v2 wrote one key per feature, which the new list has to adopt.
    fs.mkdirSync(path.dirname(storePath()), { recursive: true });
    fs.writeFileSync(
      storePath(),
      `${TelemetryKey.ID}=abc\nDEEPEVAL_EVALUATION_STATUS=old\n`,
    );
    identity.resetCacheForTesting();

    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {});

    expect(backend.only()["feature.status"]).toBe(UserStatus.OLD);
    expect(fs.readFileSync(storePath(), "utf-8")).not.toContain(
      "DEEPEVAL_EVALUATION_STATUS",
    );
  });
});

describe("the Evaluation event", () => {
  it("emits one event per run carrying its totals", () => {
    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      for (let i = 0; i < 5; i += 1) {
        telemetry.recordTestCase();
        for (const name of ["AnswerRelevancy", "Faithfulness", "Bias"]) {
          telemetry.recordMetric(name, {
            asyncMode: true,
            inComponent: false,
          });
        }
      }
    });

    const props = backend.only();
    expect(backend.events[0].event).toBe(Event.EVALUATION);
    expect(props["eval.entrypoint"]).toBe(Entrypoint.EVALUATE);
    expect(props["eval.test_case_count"]).toBe(5);
    expect(props["eval.metric_runs"]).toBe(15);
    expect(props["eval.metrics_count"]).toBe(3);
    expect(props["eval.outcome"]).toBe(Outcome.COMPLETED);
  });

  it("stamps every event with the schema version and the language", () => {
    telemetry.captureEvaluationRun(Entrypoint.COMPARE, () => {});

    const props = backend.only();
    expect(props["telemetry.schema_version"]).toBe(2);
    // Absent means Python, so this must always be present here.
    expect(props["sdk.language"]).toBe("typescript");
  });

  it("attributes a metric to the innermost of nested runs", () => {
    telemetry.captureEvaluationRun(Entrypoint.VITEST, () => {
      telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
        telemetry.recordMetric("Bias", {
          asyncMode: false,
          inComponent: false,
        });
      });
    });

    const [inner, outer] = backend.events.map((event) => event.properties);
    expect(inner["eval.entrypoint"]).toBe(Entrypoint.EVALUATE);
    expect(inner["eval.metric_runs"]).toBe(1);
    expect(outer["eval.entrypoint"]).toBe(Entrypoint.VITEST);
    expect(outer["eval.metric_runs"]).toBe(0);
  });

  it("keeps counting metrics of a run that awaits", async () => {
    // An `AsyncLocalStorage` scope only survives an `await` if the whole body
    // is inside the callback.
    await telemetry.captureEvaluationRun(Entrypoint.EVALUATE, async () => {
      await Promise.all(
        ["Bias", "Toxicity"].map(async (name) => {
          await new Promise((resolve) => setTimeout(resolve, 1));
          telemetry.recordMetric(name, { asyncMode: true, inComponent: false });
        }),
      );
    });

    expect(backend.only()["eval.metric_runs"]).toBe(2);
  });

  it("emits no vendor-reserved keys", () => {
    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordTestCase();
    });

    expect(
      Object.keys(backend.only()).filter((key) => key.startsWith("$")),
    ).toEqual([]);
  });
});

describe("run ids", () => {
  it("gives each run its own id", () => {
    for (let i = 0; i < 2; i += 1) {
      telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {});
    }

    const [first, second] = backend.events.map(
      (event) => event.properties["eval.run_id"],
    );
    expect(first).toBeTruthy();
    expect(second).toBeTruthy();
    expect(first).not.toBe(second);
  });

  it("reports processes sharing an id as one run", () => {
    // What makes a multi-worker Vitest session read as one run.
    const shared = "11111111-2222-3333-4444-555555555555";

    for (let i = 0; i < 4; i += 1) {
      telemetry.captureEvaluationRun(
        Entrypoint.VITEST,
        () => {
          telemetry.recordTestCase();
        },
        { runId: shared },
      );
    }

    expect(backend.events).toHaveLength(4);
    expect(
      new Set(backend.events.map((event) => event.properties["eval.run_id"])),
    ).toEqual(new Set([shared]));
    expect(
      backend.events.reduce(
        (total, event) =>
          total + (event.properties["eval.test_case_count"] as number),
        0,
      ),
    ).toBe(4);
  });

  it("lets an empty scope stay silent", () => {
    // The setup file loads in every suite, deepeval tests or not.
    telemetry.captureEvaluationRun(Entrypoint.VITEST, () => {}, {
      skipIfEmpty: true,
    });

    expect(backend.events).toEqual([]);
  });

  it("still reports a scope with work when skipping empties", () => {
    telemetry.captureEvaluationRun(
      Entrypoint.VITEST,
      () => {
        telemetry.recordMetric("Bias", {
          asyncMode: false,
          inComponent: false,
        });
      },
      { skipIfEmpty: true },
    );

    expect(backend.only()["eval.metric_runs"]).toBe(1);
  });

  it("never swallows an exception when skipping empties", () => {
    expect(() =>
      telemetry.captureEvaluationRun(
        Entrypoint.VITEST,
        () => {
          throw new Error("boom");
        },
        { skipIfEmpty: true },
      ),
    ).toThrow("boom");

    expect(backend.events).toEqual([]);
  });

  it("emits once from a run opened and closed by separate hooks", () => {
    // The Vitest path: two hooks, so no callback for ALS to wrap.
    const run = telemetry.beginEvaluationRun(Entrypoint.VITEST);
    telemetry.recordTestCase();
    run.finish();
    run.finish();

    expect(backend.only()["eval.test_case_count"]).toBe(1);
  });
});

describe("outcomes", () => {
  it("reports the error class of a failed run", () => {
    expect(() =>
      telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
        throw new TypeError("sk-live-secret and a user prompt");
      }),
    ).toThrow(TypeError);

    const props = backend.only();
    expect(props["eval.outcome"]).toBe(Outcome.ERRORED);
    expect(props["eval.error_type"]).toBe("TypeError");
  });

  it("never puts an exception message in the payload", async () => {
    await expect(
      telemetry.captureEvaluationRun(Entrypoint.EVALUATE, async () => {
        throw new Error("sk-live-secret and a user prompt");
      }),
    ).rejects.toThrow("sk-live-secret");

    expect(JSON.stringify(backend.only())).not.toContain("sk-live-secret");
  });
});

describe("the judge model", () => {
  it("reports a known model", () => {
    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordMetric("GEval", {
        asyncMode: false,
        inComponent: false,
        model: new OpenAIModel({ apiKey: "sk-test", model: "gpt-4o" }),
      });
    });

    const props = backend.only();
    expect(props["judge.provider"]).toBe("OpenAIModel");
    expect(props["judge.model"]).toBe("gpt-4o");
  });

  it("cannot leak a self-hosted model name", () => {
    class AcmeUnderwritingLLM {
      modelName = "acme-internal-underwriting-v3";
    }

    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordMetric("GEval", {
        asyncMode: false,
        inComponent: false,
        model: new AcmeUnderwritingLLM(),
      });
    });

    const props = backend.only();
    expect(props["judge.provider"]).toBe("custom");
    expect(props["judge.model"]).toBe("other");
    expect(JSON.stringify(props)).not.toContain("acme");
  });

  it("turns an unknown name from a known provider into other", () => {
    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordMetric("GEval", {
        asyncMode: false,
        inComponent: false,
        model: new OpenAIModel({
          apiKey: "sk-test",
          model: "gpt-internal-finetune-42",
        }),
      });
    });

    const props = backend.only();
    expect(props["judge.provider"]).toBe("OpenAIModel");
    expect(props["judge.model"]).toBe("other");
  });

  it("does not treat a subclass of a shipped model as ours", () => {
    // A subclass would otherwise inherit a provider it did not write.
    class InternalJudge extends OpenAIModel {}

    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordMetric("GEval", {
        asyncMode: false,
        inComponent: false,
        model: new InternalJudge({ apiKey: "sk-test", model: "gpt-4o" }),
      });
    });

    expect(backend.only()["judge.provider"]).toBe("custom");
  });
});

describe("standalone metrics", () => {
  it("turns bare measures into one event at flush", () => {
    for (let i = 0; i < 10; i += 1) {
      telemetry.recordMetric("AnswerRelevancy", {
        asyncMode: false,
        inComponent: false,
      });
    }
    expect(backend.events).toEqual([]);

    telemetry.flushStandaloneMetrics();

    const props = backend.only();
    expect(props["eval.entrypoint"]).toBe(Entrypoint.STANDALONE);
    expect(props["eval.metric_runs"]).toBe(10);
    expect(props["eval.flush_reason"]).toBe(FlushReason.MANUAL);
    // Same shape as every other Evaluation event, so counts can be summed.
    expect(props["eval.test_case_count"]).toBe(0);
    expect(props["eval.golden_count"]).toBe(0);
    expect(props["tracing.traced"]).toBe(false);
    expect(props["tracing.trace_count"]).toBe(0);
  });

  it("makes partial flushes sum to the true total", () => {
    for (let i = 0; i < 120; i += 1) {
      telemetry.recordMetric("Bias", { asyncMode: false, inComponent: false });
    }
    telemetry.flushStandaloneMetrics();

    const total = backend.events.reduce(
      (sum, event) => sum + (event.properties["eval.metric_runs"] as number),
      0,
    );
    expect(total).toBe(120);
    // Threshold flushes are partial sessions and must be marked as such.
    expect(
      backend.events.some(
        (event) =>
          event.properties["eval.flush_reason"] === FlushReason.THRESHOLD,
      ),
    ).toBe(true);
  });

  it("keeps metrics inside a run off the standalone path", () => {
    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordMetric("Bias", { asyncMode: false, inComponent: false });
    });
    telemetry.flushStandaloneMetrics();

    expect(backend.events).toHaveLength(1);
  });
});

describe("integrations", () => {
  it("reports an integration once per process", () => {
    for (let i = 0; i < 3; i += 1) {
      telemetry.recordTracingIntegration(Integration.LANGCHAIN);
    }

    const props = backend.only();
    expect(backend.events[0].event).toBe(Event.INTEGRATION_INSTALLED);
    expect(props["tracing.integration"]).toBe(Integration.LANGCHAIN);
  });
});

describe("opting out", () => {
  it("emits nothing", () => {
    setEnv("DEEPEVAL_TELEMETRY_OPT_OUT", "1");
    resetSettings();
    identity.resetCacheForTesting();

    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordTestCase();
    });

    expect(backend.events).toEqual([]);
    expect(telemetry.getUniqueId()).toBe("telemetry-opted-out");
  });

  it("honours the deprecated enabled flag being off", () => {
    setEnv("DEEPEVAL_TELEMETRY_ENABLED", "0");
    resetSettings();
    identity.resetCacheForTesting();

    telemetry.captureEvaluationRun(Entrypoint.EVALUATE, () => {
      telemetry.recordTestCase();
    });

    expect(backend.events).toEqual([]);
  });
});

describe("the CLI command dimension", () => {
  it("reports every registered command", () => {
    // A command is valid because the CLI dispatches it, so there is no list
    // that can drift.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { buildProgram } = require("@/cli/index") as {
      buildProgram: () => { commands: { name(): string }[] };
    };
    const names = buildProgram().commands.map((command) => command.name());

    for (const name of ["test", "login", "view"]) {
      expect(names).toContain(name);
    }

    for (const name of names) {
      backend.events.length = 0;
      telemetry.captureCliCommand(name, names);
      expect(backend.only()["cli.command"]).toBe(name);
    }
  });

  it("falls back rather than letting an unregistered name escape", () => {
    telemetry.captureCliCommand("not-a-command", ["view"]);
    expect(backend.only()["cli.command"]).toBe("unknown");

    backend.events.length = 0;
    telemetry.captureCliCommand(undefined, ["view"]);
    expect(backend.only()["cli.command"]).toBe("unknown");
  });
});

describe("the runtime dimension", () => {
  it("detects GitHub Actions", () => {
    setEnv("GITHUB_ACTIONS", "true");
    runtime.resetCacheForTesting();

    expect(runtime.detectRuntime()).toBe(Runtime.CI_GITHUB);
  });

  it("falls back to ci_other for a plain CI variable", () => {
    setEnv("GITHUB_ACTIONS", undefined);
    setEnv("GITLAB_CI", undefined);
    setEnv("CI", "1");
    runtime.resetCacheForTesting();

    expect(runtime.detectRuntime()).toBe(Runtime.CI_OTHER);
  });
});

// Generated from Python's enums by `scripts/compile_telemetry_vocabulary.py`. A
// key that differs between the SDKs forks a PostHog series silently rather than
// failing, so it has to fail here instead.
describe("parity with Python's wire vocabulary", () => {
  const vocabulary = JSON.parse(
    fs.readFileSync(path.join(__dirname, "telemetry-vocabulary.json"), "utf-8"),
  ) as Record<string, string[] | Record<string, string> | number | string>;

  function values(enumObject: Record<string, string>): string[] {
    return Object.values(enumObject).sort();
  }

  it.each([
    ["events", telemetry.Event],
    ["entrypoints", telemetry.Entrypoint],
    ["features", telemetry.Feature],
    ["integrations", Integration],
    ["props", telemetry.Prop],
    ["languages", telemetry.Language],
    ["runtimes", telemetry.Runtime],
    ["userStatuses", telemetry.UserStatus],
    ["outcomes", telemetry.Outcome],
    ["turnKinds", telemetry.TurnKind],
    ["flushReasons", telemetry.FlushReason],
    ["loginPromptSurfaces", telemetry.LoginPromptSurface],
    ["loginOutcomes", telemetry.LoginOutcome],
    ["loginMethods", telemetry.LoginMethod],
    ["telemetryKeys", telemetry.TelemetryKey],
  ])("matches Python's %s", (name, enumObject) => {
    expect(values(enumObject as Record<string, string>)).toEqual(
      vocabulary[name],
    );
  });

  it("maps every payload field to the same key Python does", () => {
    const expected = vocabulary.propsByField as Record<string, string>;
    // Field names are camelCase by design; only the wire keys have to match.
    expect(new Set(Object.values(telemetry.FIELD_TO_PROP))).toEqual(
      new Set(Object.values(expected)),
    );
    expect(Object.keys(telemetry.FIELD_TO_PROP)).toHaveLength(
      Object.keys(expected).length,
    );
  });

  it("agrees on the schema version, the run id variable and the sentinels", () => {
    expect(telemetry.TELEMETRY_SCHEMA_VERSION).toBe(vocabulary.schemaVersion);
    expect(telemetry.TELEMETRY_RUN_ID_ENV_VAR).toBe(vocabulary.runIdEnvVar);
    expect(telemetry.TELEMETRY_DATA_FILE).toBe(vocabulary.identityFileName);

    const sentinels = vocabulary.sentinels as Record<string, string>;
    expect(telemetry.CUSTOM_PROVIDER).toBe(sentinels.customProvider);
    expect(telemetry.UNKNOWN_MODEL).toBe(sentinels.unknownModel);
    expect(telemetry.UNKNOWN_CLI_COMMAND).toBe(sentinels.unknownCliCommand);
  });

  it("keeps every property key lowercase and namespaced", () => {
    for (const key of Object.values(telemetry.Prop)) {
      expect(key).toContain(".");
      expect(key).toBe(key.toLowerCase());
    }
  });

  it("stamps the same base properties Python does", () => {
    const emitted = new Set(
      Object.keys(telemetry.toRecord(telemetry.baseProperties())),
    );
    for (const key of [
      "telemetry.schema_version",
      "sdk.language",
      "deepeval.version",
      "runtime.kind",
      "user.status",
      "user.unique_id",
      "user.logged_in",
    ]) {
      expect(emitted).toContain(key);
    }
  });
});

describe("vendor containment", () => {
  function sourceFiles(): string[] {
    const root = path.join(REPO_ROOT, "typescript", "src");
    const found: string[] = [];
    const walk = (dir: string): void => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (/\.(ts|mts|cts)$/.test(entry.name)) found.push(full);
      }
    };
    walk(root);
    return found;
  }

  it("imports and calls posthog in exactly one place", () => {
    // Keeps the vendor swappable and stops capture sites sprawling.
    const imports =
      /(from\s+["']posthog-node["']|require\(["']posthog-node["']\))/;
    const calls = /\.capture\(\{\s*distinctId/;

    const importers: string[] = [];
    const callers: string[] = [];
    for (const file of sourceFiles()) {
      const source = fs.readFileSync(file, "utf-8");
      const relative = path
        .relative(path.join(REPO_ROOT, "typescript", "src"), file)
        .split(path.sep)
        .join("/");
      if (imports.test(source)) importers.push(relative);
      if (calls.test(source)) callers.push(relative);
    }

    expect(importers).toEqual(["telemetry/client.ts"]);
    expect(callers).toEqual(["telemetry/client.ts"]);
  });
});

describe("the payload type", () => {
  it("drops absent fields rather than sending nulls", () => {
    const properties: EventProperties = {
      entrypoint: Entrypoint.EVALUATE,
      testCaseCount: 0,
    };

    expect(telemetry.toRecord(properties)).toEqual({
      "eval.entrypoint": "evaluate",
      "eval.test_case_count": 0,
    });
  });
});
