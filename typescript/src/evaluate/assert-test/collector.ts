import { EvaluatedCase } from "../types";
import { getIsRunningDeepEval } from "../../utils";
import { persistCase } from "./test-run-utils";

class ResultCollector {
  private cases: EvaluatedCase[] = [];

  record(evaluatedCase: EvaluatedCase): void {
    if (getIsRunningDeepEval()) {
      this.cases.push(evaluatedCase);
      persistCase(evaluatedCase);
    }
  }

  add(evaluatedCase: EvaluatedCase): void {
    this.cases.push(evaluatedCase);
  }

  getCases(): EvaluatedCase[] {
    return [...this.cases];
  }

  get size(): number {
    return this.cases.length;
  }

  reset(): void {
    this.cases = [];
  }
}

export const globalResultCollector = new ResultCollector();
