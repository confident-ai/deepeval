#!/usr/bin/env node
import { spawn } from "node:child_process";
import { assessGovernance } from "../governance";

function printHelp(): void {
  console.log(
    `Usage: deepeval <command> [options]

Commands:
  gate          Check your project against its governance policy and exit with a non-zero code if it doesn't pass.
  test run      Run test files with Jest and post results to Confident AI.

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
        console.log(`[deepeval] Governance gate passed against ${policyName}.`);
      }
      process.exit(0);
    }

    if (!quiet) {
      console.error(
        `[deepeval] Governance gate failed against ${policyName}. ` +
          "One or more controls did not pass.",
      );
    }
    process.exit(1);
  } catch (error) {
    if (!quiet) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(
        `[deepeval] Could not assess governance for your project: ${message}\n` +
          "Make sure your project is associated with a governance policy. " +
          "If it isn't, please contact your organization administrator.",
      );
    }
    process.exit(1);
  }
}

function findJestBin(): string | null {
  const candidates = [
    "jest-cli/bin/jest.js",
    "jest-cli/bin/jest",
    "jest/build/cli.js",
    "@jest/core/build/cli/index.js",
    ".bin/jest",
  ];
  for (const rel of candidates) {
    try {
      const p = require.resolve(rel, { paths: [process.cwd()] });
      return p;
    } catch {
      // try next
    }
  }
  return null;
}

async function runTestRun(args: string[]): Promise<void> {
  const jestArgs = args.filter((a) => a !== "run");
  const quiet = jestArgs.includes("-q") || jestArgs.includes("--quiet");

  if (!quiet) {
    console.log("[deepeval] Running test suite...");
  }

  const jestPath = findJestBin();
  if (!jestPath) {
    console.error(
      "[deepeval] Could not find Jest. Ensure `jest` is installed as a devDependency.",
    );
    process.exit(1);
  }

  const child = spawn(process.execPath, [jestPath, ...jestArgs], {
    stdio: "inherit",
    cwd: process.cwd(),
    env: { ...process.env },
  });

  const exitCode = await new Promise<number>((resolve) => {
    child.on("exit", (code) => resolve(code ?? 1));
    child.on("error", () => resolve(1));
  });

  process.exit(exitCode);
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
