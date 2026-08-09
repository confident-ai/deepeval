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

function buildProgram(): Command {
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

  return program;
}

async function main(): Promise<void> {
  const program = buildProgram();

  if (process.argv.length <= 2) {
    program.outputHelp();
    process.exit(1);
  }

  await program.parseAsync(process.argv);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
