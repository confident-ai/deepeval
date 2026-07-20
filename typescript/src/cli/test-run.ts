import * as path from "path";
import {
  DEEPEVAL_RUNNING,
  DEEPEVAL_RESULTS_DIR,
  DEEPEVAL_OFFICIAL,
  DEEPEVAL_IDENTIFIER,
} from "../constants";
import { createTestRunResultsDir } from "../utils";

export interface TestRunOptions {
  paths: string[];
  official?: boolean;
  identifier?: string;
}

export async function runTest(opts: TestRunOptions): Promise<number> {
  const resultsDir = createTestRunResultsDir();

  const env: Record<string, string> = {
    [DEEPEVAL_RUNNING]: "1",
    [DEEPEVAL_RESULTS_DIR]: resultsDir,
  };
  if (opts.official) env[DEEPEVAL_OFFICIAL] = "1";
  if (opts.identifier) env[DEEPEVAL_IDENTIFIER] = opts.identifier;
  Object.assign(process.env, env);

  const setupFile = path.join(__dirname, "../integrations/vitest/index.mjs");
  const globalSetupFile = path.join(
    __dirname,
    "../integrations/vitest/global-setup.mjs",
  );

  const { startVitest } = await import("vitest/node");
  const vitest = await startVitest(
    "test",
    opts.paths,
    { watch: false, run: true },
    {
      test: {
        setupFiles: [setupFile],
        globalSetup: [globalSetupFile],
        env,
        testTimeout: 120_000,
        hookTimeout: 120_000,
      },
    },
  );

  if (!vitest) return 1;
  const failed = vitest.state.getCountOfFailedTests() > 0;
  await vitest.close();
  return failed ? 1 : 0;
}
