// Port of `deepeval/telemetry/runtime.py`. Known blind spots, accepted:
// self-hosted runners, Vercel, and Lambda land in `script` or `container`.

import * as fs from "fs";

import { Runtime } from "@/telemetry/properties";

const CI_VENDOR_VARS = [
  "JENKINS_URL",
  "CIRCLECI",
  "BUILDKITE",
  "TF_BUILD",
  "TEAMCITY_VERSION",
  "TRAVIS",
  "APPVEYOR",
  "CI",
] as const;

function inContainer(): boolean {
  if (process.env.KUBERNETES_SERVICE_HOST) return true;
  try {
    return fs.existsSync("/.dockerenv");
  } catch {
    return false;
  }
}

/**
 * Not `isTTY`: a script launched from a shell has a terminal too. The REPL is
 * the only case with no script path in `argv`.
 */
function inRepl(): boolean {
  return process.argv.length <= 1 || require.main === undefined;
}

let cached: Runtime | null = null;

export function detectRuntime(): Runtime {
  if (cached !== null) return cached;
  cached = computeRuntime();
  return cached;
}

function computeRuntime(): Runtime {
  // `Runtime.NOTEBOOK` is Python-only: no Node equivalent to detect.
  if (process.env.GITHUB_ACTIONS) return Runtime.CI_GITHUB;
  if (process.env.GITLAB_CI) return Runtime.CI_GITLAB;
  if (CI_VENDOR_VARS.some((name) => process.env[name])) return Runtime.CI_OTHER;
  if (inContainer()) return Runtime.CONTAINER;
  if (inRepl()) return Runtime.INTERACTIVE;
  return Runtime.SCRIPT;
}

export function resetCacheForTesting(): void {
  cached = null;
}
