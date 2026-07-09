import { describe, it, expect } from "vitest";
import "deepeval/vitest";
import { assertTest, LLMTestCase, metrics } from "deepeval";

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

describe("deepeval Vitest matchers", () => {
  it("toPassMetric passes a passing metric", async () => {
    await expect(testCase()).toPassMetric(new FakeMetric("Pass", true));
  });

  it("toPassMetric fails a failing metric (via .not it passes)", async () => {
    await expect(testCase()).not.toPassMetric(new FakeMetric("Fail", false));
  });

  it("toPassAll passes when all metrics pass", async () => {
    await expect(testCase()).toPassAll([
      new FakeMetric("A", true),
      new FakeMetric("B", true),
    ]);
  });

  it("assertTest also works", async () => {
    await assertTest(testCase(), [new FakeMetric("Pass", true)]);
  });
});
