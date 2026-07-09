import { it } from "vitest";
import { assertTest, Golden, LLMTestCase, metrics } from "deepeval";
import { observe } from "deepeval/tracing";

class FakeMetric extends metrics.BaseMetric {
  private readonly ok: boolean;
  constructor(ok: boolean) {
    super(0.5);
    this.ok = ok;
  }
  get name(): string {
    return "TraceFake";
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

const llmApp = observe({
  metrics: [new FakeMetric(true)],
  fn: async (query: string) => `answer to ${query}`,
});

it("explicit assertTest works via the CLI-injected env", async () => {
  const tc = new LLMTestCase({ input: "q", actualOutput: "a" });
  await assertTest(tc, [new FakeMetric(true)]);
});

it("trace-scoped assertTest evaluates the observed trace", async () => {
  const golden = new Golden({ input: "What is 2+2?", expectedOutput: "4" });
  await llmApp(golden.input);
  await assertTest({ golden });
});
