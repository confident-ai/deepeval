#!/usr/bin/env node
import { assessGovernance } from "../governance";
import { runTest, type TestRunOptions } from "./test-run";

function printHelp(): void {
  console.log(
    `Usage: deepeval <command> [options]

Commands:
  test run <path...>   Run Vitest test files as a deepeval test run: evaluate
                       assertTest / matcher results and post them to Confident AI.
  gate                 Check your project against its governance policy and exit
                       with a non-zero code if it doesn't pass.

Options for \`test run\`:
  -o, --official       Mark this run as the official baseline on Confident AI.
  -i, --identifier <s> Identify this test run.

Options:
  -q, --quiet   Suppress output. The exit code still reflects the verdict.
  -h, --help    Show this help message.`,
  );
}

function parseTestRunArgs(argv: string[]): TestRunOptions {
  const paths: string[] = [];
  let official = false;
  let identifier: string | undefined;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "-o" || arg === "--official") {
      official = true;
    } else if (arg === "-i" || arg === "--identifier") {
      identifier = argv[++i];
    } else if (arg.startsWith("-")) {
      console.error(`Unknown option for \`test run\`: ${arg}`);
      process.exit(1);
    } else {
      paths.push(arg);
    }
  }
  return { paths, official, identifier };
}

async function runTestCommand(argv: string[]): Promise<void> {
  if (argv[0] !== "run") {
    console.error(`Unknown test subcommand: ${argv[0] ?? "(none)"}\n`);
    printHelp();
    process.exit(1);
  }
  const options = parseTestRunArgs(argv.slice(1));
  if (options.paths.length === 0) {
    console.error("`deepeval test run` requires at least one path.\n");
    printHelp();
    process.exit(1);
  }
  const code = await runTest(options);
  process.exit(code);
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

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];
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
    case "test":
      await runTestCommand(args.slice(1));
      return;
    case "gate":
      await runGate(quiet);
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
