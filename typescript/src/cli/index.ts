#!/usr/bin/env node
import { assessGovernance } from "../governance";

function printHelp(): void {
  console.log(
    `Usage: deepeval <command> [options]

Commands:
  gate          Check your project against its governance policy and exit with a non-zero code if it doesn't pass.
  test run      Run test files with Jest/Vitest and post results to Confident AI.

Options:
  -q, --quiet   Suppress output. The exit code still reflects the verdict.
  -h, --help    Show this help message.`,
  );
}

async function runGate(quiet: boolean): Promise<void> {
  try {
    const { passed, governancePolicy } = await assessGovernance();
    const policyName = governancePolicy.name || "governance policy";

    if (passed) {
      if (!quiet) {
        console.log(`✅ Governance gate passed against ${policyName}.`);
      }
      process.exit(0);
    }

    if (!quiet) {
      console.error(
        `❌ Governance gate failed against ${policyName}. ` +
          "One or more controls did not pass.",
      );
    }
    process.exit(1);
  } catch (error) {
    if (!quiet) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(
        `❌ Could not assess governance for your project: ${message}\n` +
          "Make sure your project is associated with a governance policy. " +
          "If it isn't, please contact your organization administrator.",
      );
    }
    process.exit(1);
  }
}

async function runTestRun(args: string[]): Promise<void> {
  let jestArgs = args.filter((a) => a !== "run");
  const quiet = jestArgs.includes("-q") || jestArgs.includes("--quiet");

  if (!quiet) {
    console.log("🚀 Running DeepEval test suite...");
  }

  // Try Jest first, fall back to Vitest
  const jestPath = require.resolve("jest-cli/bin/jest", { paths: [process.cwd()] });
  const { run: jestRun } = await import(jestPath);

  try {
    const success = await jestRun(jestArgs);
    process.exit(success ? 0 : 1);
  } catch (err) {
    if (!quiet) {
      console.error("Failed to run tests:", (err as Error).message);
    }
    process.exit(1);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];
  const subcommand = args[1];
  const quiet = args.includes("-q") || args.includes("--quiet");

  if (command === "-h" || command === "--help") {
    printHelp();
    process.exit(0);
  }

  if (!command) {
    printHelp();
    process.exit(1);
  }

  switch (command) {
    case "gate":
      await runGate(quiet);
      return;
    case "test":
      if (subcommand === "run") {
        await runTestRun(args.slice(1));
        return;
      }
      console.error(`Unknown subcommand: test ${subcommand}\n`);
      printHelp();
      process.exit(1);
      return;
    default:
      console.error(`Unknown command: ${command}\n`);
      printHelp();
      process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
