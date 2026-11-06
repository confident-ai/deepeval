import { Command, InvalidArgumentError } from "commander";
import {
  InspectLoadError,
  InspectUnavailableError,
  listStoredRuns,
  runInspect,
} from "@/inspect";
import { SqliteUnsupportedError } from "@/sqlite-store/mode";

function parseRunId(value: string): number {
  if (!/^\d+$/.test(value)) {
    throw new InvalidArgumentError("--run-id must be a positive integer.");
  }
  return Number.parseInt(value, 10);
}

export function registerInspectCommand(program: Command): void {
  program
    .command("inspect")
    .description(
      "Explore the trace tree of a test run in an interactive terminal UI.",
    )
    .argument(
      "[path]",
      "A test_run_*.json file, a deepeval.db SQLite store (optionally " +
        "`deepeval.db#<run_id>`), or a folder containing either. " +
        "Defaults to the latest local run.",
    )
    .option(
      "-f, --folder <folder>",
      "Folder of exported runs to take the newest from.",
    )
    .option(
      "--run-id <id>",
      "Open this run id from the SQLite store instead of the latest.",
      parseRunId,
    )
    .option("--list", "List the runs in the resolved SQLite store and exit.")
    .action(
      async (
        target: string | undefined,
        options: { folder?: string; runId?: number; list?: boolean },
      ) => {
        try {
          const shared = {
            target,
            folder: options.folder,
            runId: options.runId,
          };
          if (options.list) {
            listStoredRuns(shared);
            return;
          }
          await runInspect(shared);
        } catch (e) {
          if (
            e instanceof InspectLoadError ||
            e instanceof InspectUnavailableError ||
            e instanceof SqliteUnsupportedError
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
