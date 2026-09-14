import * as os from "os";
import { Command } from "commander";
import {
  getSettings,
  getSettingSource,
  isSecretSetting,
  keystoreLocation,
  SETTING_NAMES,
  type SettingName,
} from "@/config/settings";
import { loadedDotenvPaths } from "@/config/dotenv-load";
import { getBaseApiUrl } from "@/confident/api";
import { getProvider, type ProviderSpec } from "@/cli/providers";
import { selectProvider } from "@/models/provider-selection";
import { printTable } from "@/cli/utils";
import { getVersion } from "@/cli/version";
import { HIDDEN_DIR } from "@/constants";
import {
  LOCAL_STORE_JSON,
  LOCAL_STORE_SQLITE,
  normalizeLocalStoreMode,
  isSqliteSupported,
  sqliteUnsupportedMessage,
  type LocalStoreMode,
} from "@/sqlite-store/mode";
import { resolveDbPath, resolveIncludeRowJson } from "@/sqlite-store/store";

// Settings worth showing when set, matching Python's `_RELEVANT_MARKERS`.
const RELEVANT_MARKERS = [
  "CONFIDENT_",
  "USE_",
  "_API_KEY",
  "_MODEL_NAME",
  "_DEPLOYMENT_NAME",
  "_BASE_URL",
  "_ENDPOINT",
  "TEMPERATURE",
  "DEEPEVAL_DEFAULT_SAVE",
  "DEEPEVAL_RESULTS_FOLDER",
  "DEEPEVAL_LOCAL_STORE",
  "DEEPEVAL_SQLITE_INCLUDE_ROW_JSON",
];

function maskSecret(value: string): string {
  if (!value) return "";
  return value.length <= 12 ? "*".repeat(8) : "*".repeat(8) + value.slice(-6);
}

function displayValue(name: SettingName, value: unknown): string {
  if (value === undefined || value === null) return "";
  const text = typeof value === "boolean" ? String(value) : String(value);
  if (isSecretSetting(name)) return maskSecret(text);
  return text.length <= 80 ? text : `${text.slice(0, 77)}…`;
}

/** Asks the same resolver metrics use, so the report cannot drift from it. */
function resolveActiveProvider(): {
  provider: ProviderSpec | null;
  reason: string;
} {
  const selected = selectProvider();
  if (selected === null) {
    return {
      provider: getProvider("openai") ?? null,
      reason: "no provider selected, defaulting to OpenAI",
    };
  }
  const provider = getProvider(selected) ?? null;
  const reason =
    selected === "ollama"
      ? "LOCAL_MODEL_API_KEY is the Ollama placeholder"
      : `${provider?.useFlag} is set`;
  return { provider, reason };
}

interface DiagnoseReport {
  deepeval: { version: string; node: string; platform: string };
  confident: {
    loggedIn: boolean;
    apiKey: string;
    apiKeySource: string;
    region: string;
    apiUrl: string;
  };
  model: { provider: string | null; model: string | null; reason: string };
  localStorage: {
    backend: LocalStoreMode;
    backendSource: string;
    location: string;
    resultsFolder: string | null;
    resultsFolderSource: string | null;
    includeRowJson?: boolean;
    error?: string;
  };
  sources: { dotenvFiles: string[]; keystore: string };
  settings: Array<{
    name: string;
    value: string;
    source: string;
    secret: boolean;
  }>;
}

/** Where finished test runs go, always reported (even on defaults). */
function localStorageSection(
  settings: Record<string, unknown>,
): DiagnoseReport["localStorage"] {
  const configured = settings.DEEPEVAL_LOCAL_STORE as string | undefined;
  const backend = normalizeLocalStoreMode(configured ?? "") ?? LOCAL_STORE_JSON;
  const backendSource =
    configured === undefined || configured === ""
      ? "built-in default"
      : getSettingSource("DEEPEVAL_LOCAL_STORE");
  const resultsFolder =
    (settings.DEEPEVAL_RESULTS_FOLDER as string | undefined)?.trim() || null;
  const base = {
    backend,
    backendSource,
    resultsFolder,
    resultsFolderSource: resultsFolder
      ? getSettingSource("DEEPEVAL_RESULTS_FOLDER")
      : null,
  };
  if (backend === LOCAL_STORE_SQLITE) {
    return {
      ...base,
      location: resolveDbPath(resultsFolder ?? undefined),
      includeRowJson: resolveIncludeRowJson(),
      ...(isSqliteSupported() ? {} : { error: sqliteUnsupportedMessage() }),
    };
  }
  return {
    ...base,
    location: resultsFolder
      ? `${resultsFolder}/test_run_*.json`
      : `${HIDDEN_DIR}/.latest_test_run.json (latest run only)`,
  };
}

function buildReport(): DiagnoseReport {
  const settings = getSettings() as Record<string, unknown>;
  const apiKey = (settings.CONFIDENT_API_KEY as string | undefined) ?? "";
  const { provider, reason } = resolveActiveProvider();

  const relevant = SETTING_NAMES.filter((name) => {
    const value = settings[name];
    if (value === undefined || value === null || value === "") return false;
    return RELEVANT_MARKERS.some((marker) => name.includes(marker));
  });

  return {
    deepeval: {
      version: getVersion(),
      node: process.version,
      platform: `${os.type()} ${os.release()} (${os.arch()})`,
    },
    confident: {
      loggedIn: apiKey.trim() !== "",
      apiKey: maskSecret(apiKey),
      apiKeySource:
        apiKey.trim() === ""
          ? "not set"
          : getSettingSource("CONFIDENT_API_KEY"),
      region:
        (settings.CONFIDENT_REGION as string | undefined) ?? "US (default)",
      apiUrl: getBaseApiUrl(apiKey || null),
    },
    model: {
      provider: provider?.label ?? null,
      model: provider
        ? ((settings[provider.modelSetting] as string | undefined) ??
          "(provider default)")
        : null,
      reason,
    },
    localStorage: localStorageSection(settings),
    sources: {
      dotenvFiles: loadedDotenvPaths(),
      keystore: keystoreLocation(),
    },
    settings: relevant.map((name) => ({
      name,
      value: displayValue(name, settings[name]),
      source: getSettingSource(name),
      secret: isSecretSetting(name),
    })),
  };
}

function printReport(report: DiagnoseReport): void {
  const BOLD = "\x1b[1m";
  const RESET = "\x1b[0m";

  console.log(`\n${BOLD}DeepEval${RESET}`);
  console.log(`  version    ${report.deepeval.version}`);
  console.log(`  node       ${report.deepeval.node}`);
  console.log(`  platform   ${report.deepeval.platform}`);

  console.log(`\n${BOLD}Confident AI${RESET}`);
  console.log(`  logged in  ${report.confident.loggedIn ? "yes" : "no"}`);
  console.log(
    `  api key    ${
      report.confident.apiKey
        ? `${report.confident.apiKey} (${report.confident.apiKeySource})`
        : "(not set)"
    }`,
  );
  console.log(`  region     ${report.confident.region}`);
  console.log(`  api url    ${report.confident.apiUrl}`);

  console.log(`\n${BOLD}Evaluation model${RESET}`);
  console.log(`  provider   ${report.model.provider ?? "(none configured)"}`);
  if (report.model.model) console.log(`  model      ${report.model.model}`);
  console.log(`  reason     ${report.model.reason}`);

  const storage = report.localStorage;
  console.log(`\n${BOLD}Local storage${RESET}`);
  console.log(`  backend    ${storage.backend} (${storage.backendSource})`);
  console.log(`  location   ${storage.location}`);
  if (storage.resultsFolder) {
    console.log(
      `  folder     ${storage.resultsFolder} (${storage.resultsFolderSource})`,
    );
  }
  if (storage.includeRowJson !== undefined) {
    console.log(
      `  row json   ${storage.includeRowJson ? "on" : "off (default)"}`,
    );
  }
  if (storage.error) console.log(`  ⚠ ${storage.error}`);
  console.log(
    "  change with `npx deepeval set-local-store <json|sqlite> --save=dotenv`",
  );

  console.log(`\n${BOLD}Configuration sources${RESET}`);
  console.log(
    `  dotenv     ${
      report.sources.dotenvFiles.length > 0
        ? report.sources.dotenvFiles.join(", ")
        : "(none loaded)"
    }`,
  );
  console.log(`  keystore   ${report.sources.keystore}`);

  if (report.settings.length > 0) {
    printTable(
      ["Name", "Value", "Source"],
      report.settings.map((setting) => [
        setting.name,
        setting.value,
        setting.source,
      ]),
      "Configured settings",
    );
  } else {
    console.log("\nNo DeepEval settings are configured.\n");
  }
}

export function registerDiagnoseCommand(program: Command): void {
  program
    .command("diagnose")
    .description(
      "Report the effective DeepEval configuration: evaluation model, where " +
        "each setting came from, and Confident AI login status.",
    )
    .option("--json", "Output the report as JSON (secrets are masked).")
    .action((options: { json?: boolean }) => {
      const report = buildReport();
      if (options.json) {
        console.log(JSON.stringify(report, null, 2));
        return;
      }
      printReport(report);
    });
}

export { buildReport as buildDiagnoseReport };
