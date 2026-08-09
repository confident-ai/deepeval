import { Command } from "commander";
import {
  InspectLoadError,
  InspectUnavailableError,
  runInspect,
} from "@/inspect";

export function registerInspectCommand(program: Command): void {
  program
    .command("inspect")
    .description(
      "Explore the trace tree of a test run in an interactive terminal UI.",
    )
    .argument(
      "[path]",
      "A test_run_*.json file, or a folder to take the newest run from. " +
        "Defaults to the latest local run.",
    )
    .option(
      "-f, --folder <folder>",
      "Folder of exported runs to take the newest from.",
    )
    .action(
      async (target: string | undefined, options: { folder?: string }) => {
        try {
          await runInspect({ target, folder: options.folder });
        } catch (e) {
          if (
            e instanceof InspectLoadError ||
            e instanceof InspectUnavailableError
          ) {
            console.error(`❌ ${e.message}`);
            process.exitCode = 1;
            return;
          }
          throw e;
        }
      },
    );
}
