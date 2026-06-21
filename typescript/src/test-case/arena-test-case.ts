import { LLMTestCase } from "./llm-test-case";

// Mirrors deepeval/test_case/arena_test_case.py.

export class Contestant {
  name: string;
  testCase: LLMTestCase;
  hyperparameters?: Record<string, unknown>;

  constructor(params: {
    name: string;
    testCase: LLMTestCase;
    hyperparameters?: Record<string, unknown>;
  }) {
    this.name = params.name;
    this.testCase = params.testCase;
    this.hyperparameters = params.hyperparameters;
  }
}

/**
 * A set of contestants answering the SAME `input` (and sharing the same
 * `expectedOutput`), to be compared head-to-head by an arena metric.
 */
export class ArenaTestCase {
  contestants: Contestant[];
  multimodal: boolean = false;

  constructor(params: { contestants: Contestant[]; multimodal?: boolean }) {
    this.contestants = params.contestants;
    this.multimodal = params.multimodal ?? false;
    this.validate();
  }

  private validate(): void {
    if (!this.contestants || this.contestants.length === 0) {
      throw new TypeError("'contestants' must not be empty");
    }
    const names = this.contestants.map((c) => c.name);
    if (new Set(names).size !== names.length) {
      throw new TypeError("All contestant names must be unique.");
    }
    const cases = this.contestants.map((c) => c.testCase);
    const refInput = cases[0].input;
    if (cases.slice(1).some((c) => c.input !== refInput)) {
      throw new TypeError("All contestants must have the same 'input'.");
    }
    const refExpected = cases[0].expectedOutput;
    if (cases.slice(1).some((c) => c.expectedOutput !== refExpected)) {
      throw new TypeError(
        "All contestants must have the same 'expectedOutput'.",
      );
    }
  }
}
