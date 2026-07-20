import * as fs from "fs";
import {
  DEEPEVAL_RESULTS_DIR,
  DEEPEVAL_RUNNING,
  DEEPEVAL_OFFICIAL,
  DEEPEVAL_IDENTIFIER,
} from "../../constants.js";
import { createTestRunResultsDir } from "../../utils.js";
import { wrapUpTestRun } from "../../evaluate/assert-test/index.js";

export default function setup() {
  let dir = process.env[DEEPEVAL_RESULTS_DIR];
  if (!dir) {
    dir = createTestRunResultsDir();
    process.env[DEEPEVAL_RESULTS_DIR] = dir;
  } else {
    fs.mkdirSync(dir, { recursive: true });
  }
  process.env[DEEPEVAL_RUNNING] = "1";

  const start = Date.now();
  const resultsDir = dir;

  return async () => {
    const runDuration = (Date.now() - start) / 1000;
    await wrapUpTestRun(resultsDir, {
      runDuration,
      official: process.env[DEEPEVAL_OFFICIAL] === "1",
      identifier: process.env[DEEPEVAL_IDENTIFIER] || undefined,
    });
    try {
      fs.rmSync(resultsDir, { recursive: true, force: true });
    } catch {
      // best-effort cleanup
    }
  };
}
