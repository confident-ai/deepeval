#!/usr/bin/env node
import { Command } from "commander";
import { registerAuthCommands } from "@/cli/commands/auth";
import { registerDiagnoseCommand } from "@/cli/commands/diagnose";
import { registerGateCommand } from "@/cli/commands/gate";
import { registerInspectCommand } from "@/cli/commands/inspect";
import { registerProviderCommands } from "@/cli/commands/providers";
import { registerSettingsCommands } from "@/cli/commands/settings";
import { registerTestCommands } from "@/cli/commands/test";
import { registerViewCommand } from "@/cli/commands/view";
import { getVersion } from "@/cli/version";
import { captureCliCommand, flush } from "@/telemetry";

export function buildProgram(): Command {
  const program = new Command();
  program
    .name("deepeval")
    .description("The LLM evaluation framework.")
    .version(getVersion(), "-V, --version", "Show the DeepEval version.")
    .showHelpAfterError()
    .enablePositionalOptions();

  registerTestCommands(program);
  registerGateCommand(program);
  registerSettingsCommands(program);
  registerDiagnoseCommand(program);
  registerProviderCommands(program);
  registerAuthCommands(program);
  registerViewCommand(program);
  registerInspectCommand(program);

  // Covers every registered command, including sub-commands like
  // `deepeval test run`, which report as `test`.
  program.hook("preAction", (_thisCommand, actionCommand) => {
    captureCliCommand(
      topLevelNameOf(program, actionCommand),
      program.commands.map((command) => command.name()),
    );
  });

  return program;
}

function topLevelNameOf(
  program: Command,
  actionCommand: Command,
): string | undefined {
  let command: Command | null = actionCommand;
  while (command && command.parent && command.parent !== program) {
    command = command.parent;
  }
  return command?.name();
}

async function main(): Promise<void> {
  const program = buildProgram();

  if (process.argv.length <= 2) {
    program.outputHelp();
    process.exit(1);
  }

  await program.parseAsync(process.argv);
  // A short-lived CLI process can exit before a batched event is sent.
  flush();
}

// Guarded so a test can build the program without dispatching it.
if (require.main === module) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
