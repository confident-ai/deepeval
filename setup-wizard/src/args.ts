import { resolve } from "node:path";

import { z } from "zod";

export interface CliArgs {
  from: string;
  projectDir: string;
  appUrl: string;
  apiUrl: string;
  orgId?: string;
  projId?: string;
  help: boolean;
}

const urlSchema = z.string().url();
const sourceSchema = z
  .string()
  .trim()
  .min(1)
  .max(64)
  .regex(/^[a-zA-Z0-9_-]+$/);

const requireValue = (argv: string[], index: number, flag: string): string => {
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`${flag} requires a value.`);
  }
  return value;
};

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, "");

export const parseArgs = (
  argv: string[],
  cwd: string = process.cwd(),
): CliArgs => {
  const parsed: CliArgs = {
    from: "deepeval",
    projectDir: cwd,
    appUrl: "https://app.confident-ai.com",
    apiUrl: "https://api.confident-ai.com",
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    switch (flag) {
      case "--from":
        parsed.from = requireValue(argv, index, flag);
        index += 1;
        break;
      case "--project-dir":
        parsed.projectDir = requireValue(argv, index, flag);
        index += 1;
        break;
      case "--app-url":
        parsed.appUrl = requireValue(argv, index, flag);
        index += 1;
        break;
      case "--api-url":
        parsed.apiUrl = requireValue(argv, index, flag);
        index += 1;
        break;
      case "--org-id":
        parsed.orgId = requireValue(argv, index, flag);
        index += 1;
        break;
      case "--proj-id":
        parsed.projId = requireValue(argv, index, flag);
        index += 1;
        break;
      case "--help":
      case "-h":
        parsed.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${flag ?? ""}`);
    }
  }

  parsed.projectDir = resolve(cwd, parsed.projectDir);
  parsed.from = sourceSchema.parse(parsed.from);
  parsed.appUrl = trimTrailingSlash(urlSchema.parse(parsed.appUrl));
  parsed.apiUrl = trimTrailingSlash(urlSchema.parse(parsed.apiUrl));
  return parsed;
};

export const helpText = `DeepEval Setup Wizard

Usage: deepeval-setup [options]

Runs as \`deepeval setup\` (Python) or \`npx deepeval setup\` (TypeScript).

Options:
  --from <source>        Entry point label for sign-in and analytics (default: deepeval)
  --project-dir <path>   Project directory (default: current directory)
  --app-url <url>        Confident app URL (default: https://app.confident-ai.com)
  --api-url <url>        Confident API URL (default: https://api.confident-ai.com)
  --org-id <id>          Require this organization
  --proj-id <id>         Select this project
  -h, --help             Show help
`;

export const requireInteractiveTty = (
  stdinIsTty: boolean | undefined = process.stdin.isTTY,
  stdoutIsTty: boolean | undefined = process.stdout.isTTY,
): void => {
  if (!stdinIsTty || !stdoutIsTty) {
    throw new Error(
      "Setup requires an interactive TTY. Run it directly in a terminal.",
    );
  }
};
