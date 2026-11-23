import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { parseSetupResult, readSetupResult } from "../src/result.js";

const validResult = {
  status: "completed",
  changedFiles: ["evals/evaluate.py"],
  sdks: ["deepeval-python"],
  levels: ["test-case", "span", "trace"],
  datasetSource: "Existing support examples",
  metrics: ["Answer relevancy", "Tool correctness", "Task completion"],
  rerunCommand: "python evals/evaluate.py",
  testRunId: "run-123",
};

describe("setup result validation", () => {
  it("accepts a complete structured result", () => {
    expect(parseSetupResult(validResult)).toEqual({
      ...validResult,
      shape: "component-level",
    });
  });

  it("requires a run ID for completed cloud setup", () => {
    expect(() =>
      parseSetupResult({ ...validResult, testRunId: undefined }),
    ).toThrow("completed setup");
  });

  it("allows a completed local setup without a run ID", () => {
    expect(
      parseSetupResult(
        {
          status: "completed",
          changedFiles: validResult.changedFiles,
          sdks: validResult.sdks,
          levels: validResult.levels,
          datasetSource: validResult.datasetSource,
          metrics: validResult.metrics,
          rerunCommand: validResult.rerunCommand,
        },
        false,
      ),
    ).toMatchObject({
      status: "completed",
    });
  });

  it("rejects a suite that evaluated no component", () => {
    expect(() =>
      parseSetupResult({ ...validResult, levels: ["test-case", "trace"] }),
    ).toThrow("must report the span level");
  });

  /** Task completion lives on the trace, so a component suite always has one. */
  it("rejects a component suite that scored nothing run-wide", () => {
    expect(() =>
      parseSetupResult({ ...validResult, levels: ["test-case", "span"] }),
    ).toThrow("must report the trace level");
  });

  it("accepts a black-box run that had no spans to score", () => {
    expect(
      parseSetupResult({
        ...validResult,
        shape: "black-box",
        levels: ["test-case"],
        rerunCommand: ".venv/bin/python evals/evaluate_app.py",
      }),
    ).toMatchObject({ shape: "black-box" });
  });

  it("rejects a black-box run that claims spans", () => {
    expect(() =>
      parseSetupResult({ ...validResult, shape: "black-box" }),
    ).toThrow("cannot report the span level");
  });

  it("defaults to the component-level shape", () => {
    expect(parseSetupResult(validResult).shape).toBe("component-level");
  });

  it("lets a failure report the levels it never reached", () => {
    expect(
      parseSetupResult({
        ...validResult,
        status: "failed",
        levels: ["test-case"],
        testRunId: undefined,
        errors: ["No component could be observed."],
      }),
    ).toMatchObject({ status: "failed" });
  });

  it("requires errors for failures and rejects unknown fields", () => {
    expect(() =>
      parseSetupResult({
        ...validResult,
        status: "failed",
        testRunId: undefined,
      }),
    ).toThrow("failed setup");
    expect(() =>
      parseSetupResult({ ...validResult, apiKey: "secret" }),
    ).toThrow();
  });
});

const writeResultFile = async (value: unknown): Promise<string> => {
  const directory = await mkdtemp(join(tmpdir(), "confident-result-"));
  const path = join(directory, "result.json");
  await writeFile(
    path,
    typeof value === "string" ? value : JSON.stringify(value),
    "utf8",
  );
  return path;
};

describe("reading a setup result file", () => {
  it("reads a result that meets the contract", async () => {
    const outcome = await readSetupResult(await writeResultFile(validResult));
    expect(outcome).toMatchObject({
      ok: true,
      result: { status: "completed" },
    });
  });

  /**
   * The suite has already run and spent judge money by now, so a rejected file
   * must still surrender the test run and the command that reruns it.
   */
  it("keeps the test run and rerun command from a rejected result", async () => {
    const outcome = await readSetupResult(
      await writeResultFile({ ...validResult, levels: ["test-case"] }),
    );
    expect(outcome.ok).toBe(false);
    if (outcome.ok) return;
    expect(outcome.rejected.issues.join("\n")).toContain(
      "must report the span level",
    );
    expect(outcome.rejected.salvaged).toMatchObject({
      rerunCommand: validResult.rerunCommand,
      testRunId: validResult.testRunId,
    });
  });

  it("reports unreadable JSON without salvaging anything", async () => {
    const outcome = await readSetupResult(await writeResultFile("{not json"));
    expect(outcome.ok).toBe(false);
    if (outcome.ok) return;
    expect(outcome.rejected.issues).toHaveLength(1);
    expect(outcome.rejected.salvaged).toEqual({});
  });
});
