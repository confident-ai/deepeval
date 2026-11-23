#!/usr/bin/env node

import { helpText, parseArgs, requireInteractiveTty } from "./args.js";
import { isWizardCancellation, runWizard } from "./wizard.js";

const main = async (): Promise<void> => {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(helpText);
    return;
  }
  requireInteractiveTty();
  await runWizard(args);
};

main().catch((error: unknown) => {
  if (isWizardCancellation(error)) {
    process.exitCode = 130;
    return;
  }
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Setup failed: ${message}\n`);
  process.exitCode = 1;
});
