// Central error types, mirroring deepeval/errors.py.

export class DeepEvalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DeepEvalError";
  }
}

/**
 * Raised when a metric is run on a test case that is missing required params.
 * The evaluate() runner can treat this specially (skip) vs. a hard error.
 */
export class MissingTestCaseParamsError extends DeepEvalError {
  constructor(message: string) {
    super(message);
    this.name = "MissingTestCaseParamsError";
  }
}
