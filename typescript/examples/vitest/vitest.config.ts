import { defineConfig } from "vitest/config";

// Example Vitest config for using deepeval in your own project: register the
// matchers + per-test trace capture via `setupFiles`, and post one TestRun at
// the end via `globalSetup`. `testTimeout` is raised because LLM-based metrics
// make network calls (Vitest's 5s default is too short).
//
// In practice you normally don't need this file at all — `deepeval test run`
// injects the same setup/globalSetup for you. It's shown here for reference and
// for running the examples with plain `vitest`.
export default defineConfig({
  test: {
    setupFiles: ["deepeval/vitest"],
    globalSetup: ["deepeval/vitest/global-setup"],
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
