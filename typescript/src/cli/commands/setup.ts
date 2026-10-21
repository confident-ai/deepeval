import { Command } from "commander";
import {
  defaultLauncherDependencies,
  runSetup,
  type LauncherDependencies,
} from "@/cli/setup/launcher";

export const SETUP_DESCRIPTION =
  "Set up an evaluation for this project with a coding agent, optionally " +
  "saving results to Confident AI. Downloads the latest setup wizard " +
  "(macOS and Linux), verifies its checksum, caches it, and runs it. Wizard options such as --project-dir and --proj-id are passed " +
  "through; run `deepeval setup -h` to list them.";

export function registerSetupCommand(
  program: Command,
  dependencies: () => LauncherDependencies = defaultLauncherDependencies,
): void {
  program
    .command("setup")
    .description(SETUP_DESCRIPTION)
    .argument("[args...]", "Options passed to the setup wizard")
    // Only `--help` is ours, so `-h` reaches the wizard and lists its options.
    .helpOption("--help", "Show help for this command.")
    .allowUnknownOption()
    .allowExcessArguments()
    .passThroughOptions()
    .action(async (args: string[]) => {
      // An exit code rather than process.exit, so telemetry still flushes.
      process.exitCode = await runSetup(args, dependencies());
    });
}
