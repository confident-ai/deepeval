import { describe, expect, it } from "vitest";

import {
  banner,
  localWizardSteps,
  stepHeading,
  stepRoadmap,
  WIZARD_STEP_COUNT,
} from "../src/theme.js";

/** Built at runtime because a literal escape trips `no-control-regex`. */
const ansiPattern = new RegExp(`${String.fromCharCode(27)}\\[[\\d;]*m`, "g");

describe("welcome banner", () => {
  it("draws DeepEval's big_money-ne wordmark when it fits", () => {
    const rows = banner(120).split("\n");
    expect(rows[0]).toContain("the evals cloud platform");
    expect(rows).toContain(
      "  /$$$$$$                       /$$$$$$  /$$       /$$                       /$$            /$$$$$$  /$$$$$$",
    );
    expect(banner(120)).toContain("sign in → project");
  });

  it("draws DeepEval's own wordmark when Confident AI is declined", () => {
    const local = banner(120, false);
    expect(local).toContain("Welcome to");
    expect(local).toContain("DeepEval");
    expect(local.split("\n")).toContain(
      "|_______/  \\_______/ \\_______/| $$____/ |________/ \\_/    \\_______/|__/",
    );
    expect(local).toContain("judge model → method → run");
    expect(local).not.toContain("the evals cloud platform");
    expect(local).not.toContain("Confident");
    expect(banner(70, false)).toContain("DEEPEVAL");
  });

  it("keeps the money type at narrower widths instead of another font", () => {
    for (const columns of [108, 84, 60, 18]) {
      const art = banner(columns)
        .split("\n")
        .map((row) => row.replaceAll(ansiPattern, ""))
        .filter((row) => row.includes("$"));
      expect(art.length).toBeGreaterThan(0);
      expect(art.every((row) => row.length <= columns)).toBe(true);
    }
    expect(banner(17)).toContain("CONFIDENT AI");
  });

  it("names every step in the roadmap", () => {
    expect(stepRoadmap().split(" → ")).toHaveLength(WIZARD_STEP_COUNT);
    expect(stepRoadmap(localWizardSteps).split(" → ")).toEqual([
      "judge model",
      "method",
      "run",
    ]);
  });
});

describe("wizard steps", () => {
  it("labels position and title without repeating what remains", () => {
    expect(WIZARD_STEP_COUNT).toBe(6);
    expect(stepHeading(2)).toContain("Step 2/6");
    expect(stepHeading(2)).toContain("Choose project");
    expect(stepHeading(2)).not.toContain("left");
  });

  it("accepts a title that depends on the chosen setup mode", () => {
    expect(stepHeading(6, "Launch the detected agent")).toContain(
      "Step 6/6  Launch the detected agent",
    );
    expect(stepHeading(3, undefined, localWizardSteps)).toContain(
      "Step 3/3  Run the evaluation",
    );
  });
});
