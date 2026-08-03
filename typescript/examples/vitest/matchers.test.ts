import { describe, it, expect } from "vitest";
import "deepeval/vitest";
import { LLMTestCase, metrics } from "deepeval";

class FakeMetric extends metrics.BaseMetric {
  private readonly label: string;
  private readonly ok: boolean;

  constructor(label: string, ok: boolean) {
    super(0.5);
    this.label = label;
    this.ok = ok;
  }

  get name(): string {
    return this.label;
  }

  isSuccessful(): boolean {
    return this.success ?? false;
  }

  measure(): number {
    this.score = this.ok ? 0.9 : 0.1;
    this.success = this.ok;
    return this.score;
  }
}

const testCase = () =>
  new LLMTestCase({ input: "What is 2+2?", actualOutput: "4" });

describe("deepeval Vitest matcher", () => {
  it("passes a single passing metric", async () => {
    await expect(testCase()).toPass([new FakeMetric("Pass", true)]);
  });

  it("passes when every metric in the list passes", async () => {
    await expect(testCase()).toPass([
      new FakeMetric("A", true),
      new FakeMetric("B", true),
    ]);
  });

  it("fails a failing metric (via .not it passes)", async () => {
    await expect(testCase()).not.toPass([new FakeMetric("Fail", false)]);
  });

  it("fails when any metric in the list fails", async () => {
    await expect(testCase()).not.toPass([
      new FakeMetric("A", true),
      new FakeMetric("B", false),
    ]);
  });
});
