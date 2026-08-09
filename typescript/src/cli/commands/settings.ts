import { Command } from "commander";
import {
  editSettings,
  formatSettingValue,
  getSettings,
  parseSettingValue,
  resolveSettingNames,
  SETTING_NAMES,
  type SettingName,
  type SettingsDraft,
} from "@/config/settings";
import { getFieldMeta } from "@/config/schema";
import {
  badParameter,
  handleSaveResult,
  printTable,
  QUIET_OPTION_HELP,
  SAVE_OPTION_HELP,
} from "@/cli/utils";

const REGIONS = ["US", "EU", "AU"] as const;
type Region = (typeof REGIONS)[number];

const REGION_FLAGS: Record<Region, string> = {
  US: "🇺🇸",
  EU: "🇪🇺",
  AU: "🇦🇺",
};

/** Keys `set-debug` / `unset-debug` own. */
const DEBUG_SETTINGS: SettingName[] = [
  "LOG_LEVEL",
  "DEEPEVAL_VERBOSE_MODE",
  "DEEPEVAL_LOG_STACK_TRACES",
  "DEEPEVAL_RETRY_BEFORE_LOG_LEVEL",
  "DEEPEVAL_RETRY_AFTER_LOG_LEVEL",
  "DEEPEVAL_GRPC_LOGGING",
  "GRPC_VERBOSITY",
  "GRPC_TRACE",
  "CONFIDENT_TRACE_VERBOSE",
  "CONFIDENT_TRACE_ENVIRONMENT",
  "CONFIDENT_TRACE_FLUSH",
  "CONFIDENT_TRACE_SAMPLE_RATE",
];

function printSettingsList(filters: string[]): void {
  const needles = filters
    .map((term) => term.trim().toLowerCase().replace(/-/g, "_"))
    .filter(Boolean);

  const settings = getSettings() as Record<string, unknown>;
  const rows: string[][] = [];
  for (const name of SETTING_NAMES) {
    const haystack = name.toLowerCase();
    if (needles.length > 0 && !needles.some((n) => haystack.includes(n))) {
      continue;
    }
    rows.push([
      name,
      formatSettingValue(name, settings[name]),
      getFieldMeta(name).description ?? "",
    ]);
  }

  if (rows.length === 0) {
    badParameter(`No settings matched: ${filters.join(", ")}`);
  }
  printTable(["Name", "Value", "Description"], rows, "Settings");
}

export function registerSettingsCommands(program: Command): void {
  program
    .command("settings")
    .description(
      "Set or unset any DeepEval setting, with type validation. Supports " +
        "partial, case-insensitive matching for --unset and --list.",
    )
    .argument(
      "[filters...]",
      "Optional filter(s) for --list (case-insensitive substring match).",
    )
    .option(
      "-u, --set <key=value>",
      "Set a setting (repeatable). Format: KEY=VALUE",
      (value: string, previous: string[] = []) => [...previous, value],
    )
    .option(
      "-U, --unset <key>",
      "Unset setting(s) by name or partial match (repeatable, case-insensitive).",
      (value: string, previous: string[] = []) => [...previous, value],
    )
    .option("-l, --list", "List available settings.")
    .option("-s, --save [target]", SAVE_OPTION_HELP)
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action((filters: string[], options) => {
      const save = normalizeSave(options.save);
      const toSet: string[] = options.set ?? [];
      const toUnset: string[] = options.unset ?? [];

      if (filters.length > 0 && !options.list) {
        badParameter("FILTER can only be used with --list / -l.");
      }
      if (options.list) {
        if (toSet.length > 0 || toUnset.length > 0) {
          badParameter("--list cannot be combined with --set/--unset.");
        }
        printSettingsList(filters);
        return;
      }

      const plan = new Map<SettingName, unknown>();

      for (const filter of toUnset) {
        const matches = resolveSettingNames(filter);
        if (matches.length === 0) {
          badParameter(`No settings matched: ${filter}`);
        }
        for (const name of matches) plan.set(name, null);
      }

      for (const item of toSet) {
        const index = item.indexOf("=");
        if (index < 0) {
          badParameter(`--set must be KEY=VALUE (got '${item}')`);
        }
        const key = item.slice(0, index);
        const raw = item.slice(index + 1);
        const matches = resolveSettingNames(key);
        if (matches.length === 0) badParameter(`Unknown setting: '${key}'`);
        if (matches.length > 1) {
          badParameter(
            `Ambiguous setting '${key}'; matches: ${matches.join(", ")}`,
          );
        }
        plan.set(matches[0], parseSettingValue(matches[0], raw));
      }

      if (plan.size === 0) return;

      const result = editSettings(
        (draft) => {
          for (const [name, value] of plan) {
            (draft as Record<string, unknown>)[name] = value;
          }
        },
        { save },
      );

      const changed =
        Object.keys(result.updated).length > 0 || result.removed.length > 0;
      handleSaveResult({
        result,
        save,
        quiet: options.quiet,
        successMessage: changed ? "🔧 Settings updated." : undefined,
      });
    });

  program
    .command("set-confident-region")
    .description("Set the Confident AI data region.")
    .argument("<region>", `The data region to use (${REGIONS.join(", ")}).`)
    .option("-s, --save [target]", SAVE_OPTION_HELP)
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action((region: string, options) => {
      const normalized = region.trim().toUpperCase() as Region;
      if (!REGIONS.includes(normalized)) {
        badParameter(`Region must be one of ${REGIONS.join(", ")}.`);
      }

      const save = normalizeSave(options.save);
      const result = editSettings(
        (draft) => {
          draft.CONFIDENT_REGION = normalized;
        },
        { save },
      );

      handleSaveResult({
        result,
        save,
        quiet: options.quiet,
        successMessage:
          `🙌 Congratulations! You're now using the ${REGION_FLAGS[normalized]} ` +
          `${normalized} data region for Confident AI.`,
      });
    });

  program
    .command("set-debug")
    .description(
      "Configure log level, verbose mode, retry log levels, gRPC logging and " +
        "Confident AI trace toggles.",
    )
    .option(
      "--log-level <level>",
      "Global LOG_LEVEL (DEBUG|INFO|WARNING|ERROR|CRITICAL|NOTSET).",
    )
    .option("--verbose", "Turn on DEEPEVAL_VERBOSE_MODE.")
    .option("--no-verbose", "Turn off DEEPEVAL_VERBOSE_MODE.")
    .option("--log-stack-traces", "Include stack traces in logged errors.")
    .option("--no-log-stack-traces", "Omit stack traces from logged errors.")
    .option("--retry-before-level <level>", "Log level before a retry attempt.")
    .option("--retry-after-level <level>", "Log level after a retry attempt.")
    .option("--grpc", "Turn on gRPC logging.")
    .option("--no-grpc", "Turn off gRPC logging.")
    .option(
      "--grpc-verbosity <level>",
      "Set GRPC_VERBOSITY (DEBUG|INFO|ERROR|NONE).",
    )
    .option(
      "--grpc-trace <tracers>",
      "Set GRPC_TRACE (comma-separated tracer names, or '*' for all).",
    )
    .option("--trace-verbose", "Enable CONFIDENT_TRACE_VERBOSE.")
    .option("--no-trace-verbose", "Disable CONFIDENT_TRACE_VERBOSE.")
    .option(
      "--trace-env <environment>",
      'Set CONFIDENT_TRACE_ENVIRONMENT ("development", "staging", "production", "testing").',
    )
    .option("--trace-flush", "Enable CONFIDENT_TRACE_FLUSH.")
    .option("--no-trace-flush", "Disable CONFIDENT_TRACE_FLUSH.")
    .option(
      "--trace-sample-rate <rate>",
      "Set CONFIDENT_TRACE_SAMPLE_RATE (0 to 1).",
    )
    .option("-s, --save [target]", SAVE_OPTION_HELP)
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action((options) => {
      const save = normalizeSave(options.save);
      const result = editSettings(
        (draft) => {
          assign(draft, "LOG_LEVEL", options.logLevel);
          assign(
            draft,
            "DEEPEVAL_VERBOSE_MODE",
            explicitBoolean(options, "verbose"),
          );
          assign(
            draft,
            "DEEPEVAL_LOG_STACK_TRACES",
            explicitBoolean(options, "logStackTraces"),
          );
          assign(
            draft,
            "DEEPEVAL_RETRY_BEFORE_LOG_LEVEL",
            options.retryBeforeLevel,
          );
          assign(
            draft,
            "DEEPEVAL_RETRY_AFTER_LOG_LEVEL",
            options.retryAfterLevel,
          );
          assign(
            draft,
            "DEEPEVAL_GRPC_LOGGING",
            explicitBoolean(options, "grpc"),
          );
          assign(draft, "GRPC_VERBOSITY", options.grpcVerbosity);
          assign(draft, "GRPC_TRACE", options.grpcTrace);
          assign(
            draft,
            "CONFIDENT_TRACE_VERBOSE",
            explicitBoolean(options, "traceVerbose"),
          );
          assign(draft, "CONFIDENT_TRACE_ENVIRONMENT", options.traceEnv);
          assign(
            draft,
            "CONFIDENT_TRACE_FLUSH",
            explicitBoolean(options, "traceFlush"),
          );
          if (options.traceSampleRate !== undefined) {
            draft.CONFIDENT_TRACE_SAMPLE_RATE = parseSettingValue(
              "CONFIDENT_TRACE_SAMPLE_RATE",
              options.traceSampleRate,
            ) as number;
          }
        },
        { save },
      );

      const changed =
        Object.keys(result.updated).length > 0 || result.removed.length > 0;
      handleSaveResult({
        result,
        save,
        quiet: options.quiet,
        successMessage: changed ? "🔊 Debug options updated." : undefined,
      });
    });

  program
    .command("unset-debug")
    .description(
      "Restore default behavior by removing debug-related overrides.",
    )
    .option(
      "-s, --save [target]",
      "Remove only the debug-related environment variables from a dotenv file. " +
        "Usage: --save=dotenv[:path] (default: .env.local)",
    )
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action((options) => {
      const save = normalizeSave(options.save);
      const result = editSettings(
        (draft) => {
          for (const name of DEBUG_SETTINGS) {
            (draft as Record<string, unknown>)[name] = null;
          }
        },
        { save },
      );

      const changed =
        Object.keys(result.updated).length > 0 || result.removed.length > 0;
      handleSaveResult({
        result,
        save,
        quiet: options.quiet,
        successMessage: changed ? "🔇 Debug options unset." : undefined,
      });
    });
}

/** Commander gives `true` for a bare `--save`, meaning the default dotenv path. */
export function normalizeSave(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (value === true) return "dotenv";
  return String(value);
}

export function assign<K extends SettingName>(
  draft: SettingsDraft,
  name: K,
  value: unknown,
): void {
  if (value === undefined) return;
  (draft as Record<string, unknown>)[name] = value;
}

/** Commander defaults a `--x` / `--no-x` pair to `true`, so consult argv. */
export function explicitBoolean(
  options: Record<string, unknown>,
  key: string,
): boolean | undefined {
  const flag = key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);
  const passed = process.argv.some(
    (arg) => arg === `--${flag}` || arg === `--no-${flag}`,
  );
  return passed ? (options[key] as boolean) : undefined;
}
