import { Command, InvalidArgumentError } from "commander";
import { type TestRunDisplay } from "@/env-flags";
import { runTest } from "@/cli/test-run";

const DISPLAY_CHOICES: TestRunDisplay[] = ["all", "passing", "failing"];

function parseDisplay(value: string): TestRunDisplay {
  const normalized = value.trim().toLowerCase();
  if (!DISPLAY_CHOICES.includes(normalized as TestRunDisplay)) {
    throw new InvalidArgumentError(
      `Expected one of ${DISPLAY_CHOICES.join(", ")}.`,
    );
  }
  return normalized as TestRunDisplay;
}

function parsePositiveInt(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new InvalidArgumentError("Expected a positive integer.");
  }
  return parsed;
}

export function registerTestCommands(program: Command): void {
  const test = program.command("test").description("Run deepeval test suites.");

  test
    .command("run")
    .description(
      "Run Vitest test files as a deepeval test run: evaluate toPass matcher " +
        "results and post them to Confident AI.",
    )
    .argument("<path...>", "Test file(s) or directory(ies) to run.")
    .option(
      "-o, --official",
      "Mark this run as the official baseline on Confident AI.",
    )
    .option("-i, --identifier <identifier>", "Identify this test run.")
    .option(
      "--ignore-errors",
      "Ignore metric errors instead of failing the test case.",
    )
    .option(
      "-s, --skip-on-missing-params",
      "Skip test cases that are missing metric parameters.",
    )
    .option("-v, --verbose", "Turn on verbose logs for every metric.")
    .option(
      "-d, --display <display>",
      `Which test cases to display at the end (${DISPLAY_CHOICES.join("|")}).`,
      parseDisplay,
    )
    .option(
      "--max-concurrent <n>",
      "Maximum number of metrics evaluated at once.",
      parsePositiveInt,
    )
    .option("-c, --use-cache", "Reuse cached metric results where possible.")
    .action(async (paths: string[], options) => {
      const code = await runTest({
        paths,
        official: options.official,
        identifier: options.identifier,
        ignoreErrors: options.ignoreErrors,
        skipOnMissingParams: options.skipOnMissingParams,
        verbose: options.verbose,
        display: options.display,
        maxConcurrent: options.maxConcurrent,
        useCache: options.useCache,
      });
      process.exit(code);
    });
}
