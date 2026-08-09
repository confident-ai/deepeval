// The `set-<provider>` / `unset-<provider>` command pairs, generated from the
// registry in ../providers.ts so every provider behaves identically.

import * as fs from "fs";
import { Command, InvalidArgumentError } from "commander";
import { editSettings, type SettingsDraft } from "@/config/settings";
import {
  ownedSettings,
  PROVIDERS,
  USE_LLM_FLAGS,
  COST_OPTION_HELP,
  type ProviderSpec,
} from "@/cli/providers";
import {
  badParameter,
  coerceBlankToNull,
  handleSaveResult,
  prompt,
  QUIET_OPTION_HELP,
  SAVE_OPTION_HELP,
} from "@/cli/utils";
import { normalizeSave } from "@/cli/commands/settings";

function parseFloatOption(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new InvalidArgumentError("Expected a number.");
  }
  return parsed;
}

function loadJsonFile(path: string, flag: string): string {
  let raw: string;
  try {
    raw = fs.readFileSync(path, "utf-8").trim();
  } catch {
    badParameter(`Could not read file for ${flag}: ${path}`);
  }
  if (!raw) badParameter(`File for ${flag} is empty: ${path}`);
  try {
    return JSON.stringify(JSON.parse(raw));
  } catch {
    badParameter(`File for ${flag} does not contain valid JSON: ${path}`);
  }
}

function switchModelProvider(
  draft: SettingsDraft,
  provider: ProviderSpec,
): void {
  for (const flag of USE_LLM_FLAGS) {
    if (flag !== provider.useFlag) {
      (draft as Record<string, unknown>)[flag] = null;
    }
  }
  (draft as Record<string, unknown>)[provider.useFlag] = true;
}

function registerSetCommand(program: Command, provider: ProviderSpec): void {
  const command = program
    .command(`set-${provider.id}`)
    .description(`Configure ${provider.label} as the active LLM provider.`)
    .option(
      "-m, --model <model>",
      provider.modelHelp ?? "Model identifier to use for this provider.",
    );

  if (provider.apiKeyPrompt) {
    command.option(
      "-k, --prompt-api-key",
      `Prompt for ${provider.apiKeyPrompt.setting} (input hidden). Not suitable for CI.`,
    );
  }
  if (provider.credentialPrompts) {
    command.option(
      "-a, --prompt-credentials",
      `Prompt for ${provider.credentialPrompts
        .map((credential) => credential.setting)
        .join(" and ")} (input hidden). Not suitable for CI.`,
    );
  }
  for (const option of provider.extraOptions ?? []) {
    command.option(option.flags, option.description);
  }
  if (provider.costSettings) {
    command
      .option(
        "-i, --cost-per-input-token <cost>",
        COST_OPTION_HELP.input,
        parseFloatOption,
      )
      .option(
        "-o, --cost-per-output-token <cost>",
        COST_OPTION_HELP.output,
        parseFloatOption,
      );
  }

  command
    .option("-s, --save [target]", SAVE_OPTION_HELP)
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action(async (options: Record<string, unknown>) => {
      const save = normalizeSave(options.save);
      const model = coerceBlankToNull(options.model as string | undefined);

      const secrets: Array<{ setting: string; value: string }> = [];
      if (options.promptApiKey && provider.apiKeyPrompt) {
        const value = await prompt(`${provider.apiKeyPrompt.label}: `, true);
        const coerced = coerceBlankToNull(value);
        if (coerced) {
          secrets.push({
            setting: provider.apiKeyPrompt.setting,
            value: coerced,
          });
        }
      }
      if (options.promptCredentials && provider.credentialPrompts) {
        for (const credential of provider.credentialPrompts) {
          const value = await prompt(`${credential.label}: `, true);
          const coerced = coerceBlankToNull(value);
          if (coerced)
            secrets.push({ setting: credential.setting, value: coerced });
        }
      }

      const result = editSettings(
        (draft) => {
          switchModelProvider(draft, provider);
          const write = (setting: string, value: unknown) => {
            (draft as Record<string, unknown>)[setting] = value;
          };

          for (const [setting, value] of Object.entries(
            provider.fixedValues ?? {},
          )) {
            write(setting, value);
          }
          if (model !== null) write(provider.modelSetting, model);
          for (const secret of secrets) write(secret.setting, secret.value);

          for (const option of provider.extraOptions ?? []) {
            const raw = options[option.key];
            const value =
              raw === undefined
                ? option.defaultValue
                : (raw as string | number);
            if (value === undefined) continue;
            if (option.fromJsonFile) {
              write(option.setting, loadJsonFile(String(value), option.flags));
            } else if (option.numeric) {
              const parsed = Number(value);
              if (!Number.isFinite(parsed)) {
                badParameter(
                  `${option.flags} expects a number (got '${value}').`,
                );
              }
              write(option.setting, parsed);
            } else {
              const coerced = coerceBlankToNull(String(value));
              if (coerced !== null) write(option.setting, coerced);
            }
          }

          if (provider.costSettings) {
            if (options.costPerInputToken !== undefined) {
              write(provider.costSettings.input, options.costPerInputToken);
            }
            if (options.costPerOutputToken !== undefined) {
              write(provider.costSettings.output, options.costPerOutputToken);
            }
          }

          // An API key means the Developer API; project/location or a service
          // account means Vertex AI.
          if (provider.id === "gemini") {
            const hasApiKey = secrets.some(
              (secret) => secret.setting === "GOOGLE_API_KEY",
            );
            const hasVertexInputs =
              options.project !== undefined ||
              options.location !== undefined ||
              options.serviceAccountFile !== undefined;
            if (hasApiKey) {
              draft.GOOGLE_GENAI_USE_VERTEXAI = false;
            } else if (hasVertexInputs) {
              draft.GOOGLE_GENAI_USE_VERTEXAI = true;
            }
          }
        },
        { save },
      );

      const effectiveModel =
        model ??
        (result.updated[provider.modelSetting] as string | undefined) ??
        (process.env[provider.modelSetting] || null);
      if (!effectiveModel) {
        badParameter(
          `${provider.label} model name is not set. Pass --model (or set ` +
            `${provider.modelSetting}).`,
        );
      }

      handleSaveResult({
        result,
        save,
        quiet: options.quiet as boolean | undefined,
        successMessage:
          `🙌 Congratulations! You're now using ${provider.successNoun ?? provider.label} ` +
          `\`${effectiveModel}\` for all evals that require an LLM.`,
      });
    });
}

function registerUnsetCommand(program: Command, provider: ProviderSpec): void {
  const command = program
    .command(`unset-${provider.id}`)
    .description(`Unset ${provider.label} as the active LLM provider.`)
    .option(
      "-s, --save [target]",
      `Remove only the ${provider.label} related environment variables from a ` +
        "dotenv file. Usage: --save=dotenv[:path] (default: .env.local)",
    );

  if (provider.secretSettings && provider.secretSettings.length > 0) {
    command.option(
      "-x, --clear-secrets",
      `Also remove ${provider.secretSettings.join(", ")} from the dotenv store.`,
    );
  }

  command.option("-q, --quiet", QUIET_OPTION_HELP).action((options) => {
    const save = normalizeSave(options.save);
    const result = editSettings(
      (draft) => {
        for (const setting of ownedSettings(provider)) {
          (draft as Record<string, unknown>)[setting] = null;
        }
        if (options.clearSecrets) {
          for (const setting of provider.secretSettings ?? []) {
            (draft as Record<string, unknown>)[setting] = null;
          }
        }
      },
      { save },
    );

    if (
      handleSaveResult({
        result,
        save,
        quiet: options.quiet,
        updatedMessage: `Removed ${provider.label} environment variables from {path}.`,
      })
    ) {
      const openaiKey = process.env.OPENAI_API_KEY;
      if (openaiKey && openaiKey.trim() !== "") {
        console.log(
          "🙌 OpenAI will still be used by default because OPENAI_API_KEY is set.",
        );
      } else {
        console.log(
          `${provider.label} has been unset. No active provider is configured. ` +
            "Set one with the CLI, or add credentials to .env[.local].",
        );
      }
    }
  });
}

export function registerProviderCommands(program: Command): void {
  for (const provider of PROVIDERS) {
    registerSetCommand(program, provider);
    registerUnsetCommand(program, provider);
  }
}
