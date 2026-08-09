import * as path from "path";
import {
  DEEPEVAL_DISPLAY,
  DEEPEVAL_IDENTIFIER,
  DEEPEVAL_MAX_CONCURRENT,
  DEEPEVAL_OFFICIAL,
  DEEPEVAL_RESULTS_DIR,
  DEEPEVAL_RUNNING,
  DEEPEVAL_VERBOSE_MODE,
  ENABLE_DEEPEVAL_CACHE,
  IGNORE_DEEPEVAL_ERRORS,
  SKIP_DEEPEVAL_MISSING_PARAMS,
} from "@/constants";
import { type TestRunDisplay } from "@/env-flags";
import { createTestRunResultsDir } from "@/utils";

export interface TestRunOptions {
  paths: string[];
  official?: boolean;
  identifier?: string;
  ignoreErrors?: boolean;
  skipOnMissingParams?: boolean;
  verbose?: boolean;
  display?: TestRunDisplay;
  maxConcurrent?: number;
  useCache?: boolean;
}

export async function runTest(opts: TestRunOptions): Promise<number> {
  const resultsDir = createTestRunResultsDir();

  const env: Record<string, string> = {
    [DEEPEVAL_RUNNING]: "1",
    [DEEPEVAL_RESULTS_DIR]: resultsDir,
  };
  if (opts.official) env[DEEPEVAL_OFFICIAL] = "1";
  if (opts.identifier) env[DEEPEVAL_IDENTIFIER] = opts.identifier;
  if (opts.ignoreErrors) env[IGNORE_DEEPEVAL_ERRORS] = "1";
  if (opts.skipOnMissingParams) env[SKIP_DEEPEVAL_MISSING_PARAMS] = "1";
  if (opts.verbose) env[DEEPEVAL_VERBOSE_MODE] = "1";
  if (opts.useCache) env[ENABLE_DEEPEVAL_CACHE] = "1";
  if (opts.display) env[DEEPEVAL_DISPLAY] = opts.display;
  if (opts.maxConcurrent != null) {
    env[DEEPEVAL_MAX_CONCURRENT] = String(opts.maxConcurrent);
  }
  Object.assign(process.env, env);

  const setupFile = path.join(__dirname, "../integrations/vitest/index.mjs");
  const globalSetupFile = path.join(
    __dirname,
    "../integrations/vitest/global-setup.mjs",
  );

  const { startVitest } = await import("vitest/node");
  const vitest = await startVitest("test", opts.paths, {
    watch: false,
    setupFiles: [setupFile],
    globalSetup: [globalSetupFile],
    env,
    testTimeout: 120_000,
    hookTimeout: 120_000,
  });

  if (!vitest) return 1;
  const failed = vitest.state.getCountOfFailedTests() > 0;
  await vitest.close();
  return failed ? 1 : 0;
}
