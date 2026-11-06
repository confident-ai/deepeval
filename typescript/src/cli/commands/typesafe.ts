// `set-typesafe` / `unset-typesafe`. Not part of the provider registry: that
// table switches the active LLM, while a System One model sits beside it and
// is picked up only through the eval mode.

import { Command, InvalidArgumentError } from "commander";
import { editSettings, getSettings } from "@/config/settings";
import { EVAL_MODE_ENV_VAR, EvalMode } from "@/config/eval-mode";
import { DEFAULT_TYPESAFE_MODEL } from "@/models/system-one/constants";
import {
  coerceBlankToNull,
  handleSaveResult,
  prompt,
  QUIET_OPTION_HELP,
  SAVE_OPTION_HELP,
} from "@/cli/utils";
import { normalizeSave } from "@/cli/commands/settings";

function parseCost(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new InvalidArgumentError("Expected a number >= 0.");
  }
  return parsed;
}

export function registerTypeSafeCommands(program: Command): void {
  program
    .command("set-typesafe")
    .description(
      "Configure TypeSafe AI as the System One model (Jev) used by the hybrid and system_one eval modes.",
    )
    .option(
      "-m, --model <model>",
      `TypeSafe AI System One model (default: ${DEFAULT_TYPESAFE_MODEL}).`,
    )
    .option(
      "-k, --prompt-api-key",
      "Prompt for TYPESAFE_API_KEY (input hidden). Not suitable for CI. If " +
        "--save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to " +
        "dotenv in plaintext.",
    )
    .option(
      "-i, --cost-per-input-token <cost>",
      "USD per input token override used for cost tracking. Preconfigured for Jev models.",
      parseCost,
    )
    .option("-s, --save [target]", SAVE_OPTION_HELP)
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action(async (options: Record<string, unknown>) => {
      const save = normalizeSave(options.save);
      const model = coerceBlankToNull(options.model as string | undefined);
      let apiKey: string | null = null;
      if (options.promptApiKey) {
        apiKey = coerceBlankToNull(await prompt("TypeSafe AI API key: ", true));
      }

      const result = editSettings(
        (draft) => {
          if (apiKey !== null) draft.TYPESAFE_API_KEY = apiKey;
          if (model !== null) draft.TYPESAFE_MODEL_NAME = model;
          if (options.costPerInputToken !== undefined) {
            draft.TYPESAFE_COST_PER_INPUT_TOKEN =
              options.costPerInputToken as number;
          }
        },
        { save },
      );

      const effectiveModel =
        getSettings().TYPESAFE_MODEL_NAME || DEFAULT_TYPESAFE_MODEL;
      handleSaveResult({
        result,
        save,
        quiet: options.quiet as boolean | undefined,
        successMessage:
          `🙌 TypeSafe AI \`${effectiveModel}\` is configured. Pick how it ` +
          `judges with \`npx deepeval set-eval-mode ${EvalMode.HYBRID}\` or ` +
          `\`npx deepeval set-eval-mode ${EvalMode.SYSTEM_ONE}\`.`,
      });
    });

  program
    .command("unset-typesafe")
    .description("Remove the TypeSafe AI System One configuration.")
    .option(
      "-s, --save [target]",
      "Remove only the TypeSafe AI related environment variables from a " +
        "dotenv file. Usage: --save=dotenv[:path] (default: .env.local)",
    )
    .option(
      "-x, --clear-secrets",
      "Also remove TYPESAFE_API_KEY from the dotenv store.",
    )
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action((options) => {
      const save = normalizeSave(options.save);
      const result = editSettings(
        (draft) => {
          draft.TYPESAFE_MODEL_NAME = null;
          draft.TYPESAFE_COST_PER_INPUT_TOKEN = null;
          if (options.clearSecrets) draft.TYPESAFE_API_KEY = null;
        },
        { save },
      );

      if (
        handleSaveResult({
          result,
          save,
          quiet: options.quiet,
          updatedMessage:
            "Removed TypeSafe AI environment variables from {path}.",
        })
      ) {
        console.log(
          `The TypeSafe AI configuration has been removed. Metrics running ` +
            `with ${EVAL_MODE_ENV_VAR}=${EvalMode.HYBRID} or ` +
            `${EVAL_MODE_ENV_VAR}=${EvalMode.SYSTEM_ONE} will now fail until ` +
            `TYPESAFE_API_KEY is set again or you switch back with ` +
            `\`npx deepeval set-eval-mode ${EvalMode.LLM}\`.`,
        );
      }
    });
}
