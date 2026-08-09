import * as fs from "fs";
import * as path from "path";
import { randomUUID } from "crypto";
import { Command } from "commander";
import { HIDDEN_DIR, LATEST_TEST_RUN_FILE } from "@/constants";
import { DotenvHandler } from "@/config/dotenv-handler";
import { editSettings, getSettings, getSettingSource } from "@/config/settings";
import {
  AuthFlowError,
  browserPairingLogin,
  completeCliOnboarding,
  getCliOnboardingContext,
  promptCheckbox,
  promptSelect,
  promptText,
  type CliQuestionnaire,
  type CliQuestionnaireQuestion,
  type OnboardingRequest,
  type QuestionnaireAnswer,
} from "@/cli/auth/flow";
import {
  coerceBlankToNull,
  handleSaveResult,
  prompt,
  QUIET_OPTION_HELP,
} from "@/cli/utils";
import { openBrowser, PROD, withUtm, WWW } from "@/cli/utm";
import { normalizeSave } from "@/cli/commands/settings";

const DEFAULT_SAVE = "dotenv:.env.local";

const REGION_CHOICES: Array<[string, string]> = [
  ["🇺🇸 United States (US)", "US"],
  ["🇪🇺 European Union (EU)", "EU"],
  ["🇦🇺 Australia (AU)", "AU"],
];

const API_KEY_PATTERN =
  /^confident_(us|eu|au)_(org|proj|global)_[A-Za-z0-9+/]+={0,2}$/i;

function warnForPastedApiKey(apiKey: string): void {
  const match = API_KEY_PATTERN.exec(apiKey.trim());
  if (!match) return;
  const keyRegion = match[1].toUpperCase();
  const keyScope = match[2].toLowerCase();
  const configuredRegion = getSettings().CONFIDENT_REGION;

  if (configuredRegion && keyRegion !== configuredRegion.toUpperCase()) {
    console.log(
      `⚠️  This API key is for the ${keyRegion} region, but DeepEval is ` +
        `configured to use ${configuredRegion.toUpperCase()}. The region ` +
        "configured in DeepEval must match the key's region in Confident AI.",
    );
  }
  if (keyScope === "org") {
    console.log(
      "⚠️  This is an organization API key, which cannot be used to log in to " +
        "DeepEval. Use a project API key from Project Settings > API Keys instead.",
    );
  }
}

function printApiKeyLocation(): void {
  console.log(
    `Find your project API key at ${PROD} under Project Settings > API Keys.`,
  );
}

async function promptPasteApiKey(): Promise<string> {
  for (;;) {
    const apiKey = coerceBlankToNull(
      await prompt("🔐 Enter your project API key: ", true),
    );
    if (apiKey) {
      warnForPastedApiKey(apiKey);
      return apiKey;
    }
    console.log("❌ Project API key cannot be empty. Please try again.\n");
  }
}

/** Runs before the pairing session exists, so it all hits one regional backend. */
async function promptAndPersistRegion(save: string | undefined): Promise<void> {
  const current = (getSettings().CONFIDENT_REGION || "US").toUpperCase();
  const choices = [...REGION_CHOICES].sort(
    (a, b) => Number(a[1] !== current) - Number(b[1] !== current),
  );
  const region = await promptSelect(
    "Select your Confident AI data region:",
    choices,
  );
  if (region === getSettings().CONFIDENT_REGION) return;
  editSettings(
    (draft) => {
      draft.CONFIDENT_REGION = region;
    },
    { save },
  );
}

async function promptQuestionnaireText(
  question: CliQuestionnaireQuestion,
): Promise<string> {
  for (;;) {
    const value = (
      await promptText(
        question.prompt,
        question.defaultValue as string | undefined,
      )
    ).trim();
    if (question.required && !value) {
      console.log(`❌ ${question.prompt} cannot be empty. Please try again.\n`);
      continue;
    }
    if (question.maxLength != null && value.length > question.maxLength) {
      console.log(
        `❌ ${question.prompt} must be at most ${question.maxLength} characters.`,
      );
      continue;
    }
    return value;
  }
}

async function promptDynamicQuestionnaire(
  questionnaire: CliQuestionnaire,
): Promise<Record<string, QuestionnaireAnswer>> {
  const answers: Record<string, QuestionnaireAnswer> = {};

  for (const question of questionnaire.questions) {
    if (question.type === "text") {
      answers[question.id] = await promptQuestionnaireText(question);
      continue;
    }

    const choices: Array<[string, string | boolean]> = (
      question.options ?? []
    ).map((option) => [option.label, option.value]);

    if (question.type === "single_select") {
      if (question.defaultValue !== undefined) {
        choices.sort(
          (a, b) =>
            Number(a[1] !== question.defaultValue) -
            Number(b[1] !== question.defaultValue),
        );
      }
      answers[question.id] = (await promptSelect(
        question.prompt,
        choices,
      )) as QuestionnaireAnswer;
      continue;
    }

    const minimum = question.minSelections ?? (question.required ? 1 : 0);
    let selected: Array<string | boolean>;
    for (;;) {
      selected = await promptCheckbox(question.prompt, choices, minimum);
      const exclusiveValues = new Set(
        (question.options ?? [])
          .filter((option) => option.exclusive)
          .map((option) => option.value),
      );
      if (
        selected.length > 1 &&
        selected.some((value) => exclusiveValues.has(value))
      ) {
        console.log(
          "❌ An exclusive option cannot be combined with another selection.",
        );
        continue;
      }
      break;
    }

    for (const option of question.options ?? []) {
      if (option.acceptsCustomValue && selected.includes(option.value)) {
        const custom = await promptText(
          option.customPrompt || "Please specify",
        );
        selected = selected.map((value) =>
          value === option.value ? custom : value,
        );
      }
    }
    answers[question.id] = selected.map(String);
  }

  return answers;
}

async function completeBrowserLogin(): Promise<string | null> {
  const authorization = await browserPairingLogin();
  if (!authorization) return null;

  try {
    const context = await getCliOnboardingContext(authorization.setupToken);
    let request: OnboardingRequest;

    if (context.state === "new_user") {
      console.log("\nLet's set up your workspace.");
      if (!context.questionnaire) {
        throw new AuthFlowError(
          "The server did not provide a CLI onboarding questionnaire.",
        );
      }
      const answers = await promptDynamicQuestionnaire(context.questionnaire);
      const organizationName = answers.organizationName;
      const projectName = answers.projectName;
      if (
        typeof organizationName !== "string" ||
        typeof projectName !== "string"
      ) {
        throw new AuthFlowError(
          "The server questionnaire did not collect the required organization " +
            "and project names.",
        );
      }
      request = {
        questionnaireVersion: context.questionnaire.version,
        questionnaireAnswers: answers,
      };
      console.log(
        `\nYour organization and project will be created as ` +
          `${organizationName} / ${projectName}.`,
      );
      const confirmed = (await prompt("Continue? [Y/n] ")).toLowerCase();
      if (confirmed && !["y", "yes"].includes(confirmed)) {
        throw new AuthFlowError("Setup cancelled.");
      }
    } else {
      const projects = context.projects.filter(
        (project) => project.canCreateApiKey,
      );
      if (projects.length === 0) {
        throw new AuthFlowError(
          "You do not have permission to create an API key for any project in " +
            "this organization.",
        );
      }
      const projectId = await promptSelect(
        "Select a project:",
        projects.map(
          (project) => [project.name, project.id] as [string, string],
        ),
      );
      request = { projectId };
    }

    return await completeCliOnboarding(
      authorization.setupToken,
      request,
      randomUUID(),
    );
  } catch (error) {
    if (error instanceof AuthFlowError) {
      console.log(
        `\n⚠️  CLI onboarding could not be completed: ${error.message}`,
      );
      return null;
    }
    throw error;
  }
}

async function resolveLoginKey(save: string | undefined): Promise<string> {
  const method = await promptSelect(
    "How would you like to log in to Confident AI?",
    [
      ["Log in via your browser", "browser"],
      ["Paste a project API key", "paste"],
    ],
  );

  if (method === "paste") {
    printApiKeyLocation();
    return promptPasteApiKey();
  }

  await promptAndPersistRegion(save);
  const key = await completeBrowserLogin();
  if (key) return key;

  console.log(
    "\nNo problem — paste a project API key from the platform instead.",
  );
  printApiKeyLocation();
  return promptPasteApiKey();
}

export function registerAuthCommands(program: Command): void {
  program
    .command("login")
    .description(
      "Log in to Confident AI. Opens the platform for authentication, then " +
        "completes organization and project setup in the terminal. The key is " +
        "saved to your environment variables, typically .env.local, unless a " +
        "different path is provided with --save.",
    )
    .option(
      "-s, --save [target]",
      "Where to persist settings. Format: dotenv[:path]. Defaults to .env.local.",
    )
    .option(
      "--api-key <key>",
      "Log in non-interactively with a project API key instead of the prompts.",
    )
    .action(async (options) => {
      const settings = getSettings();
      const save =
        normalizeSave(options.save) ??
        settings.DEEPEVAL_DEFAULT_SAVE ??
        DEFAULT_SAVE;

      try {
        const explicitKey = coerceBlankToNull(options.apiKey);
        let key: string;
        if (explicitKey) {
          key = explicitKey;
          warnForPastedApiKey(key);
        } else {
          key = await resolveLoginKey(save);
        }

        const result = editSettings(
          (draft) => {
            draft.CONFIDENT_API_KEY = key;
          },
          { save },
        );

        if (Object.keys(result.updated).length > 0) {
          if (!result.handled) {
            console.log("Unsupported --save option. Use --save=dotenv[:path].");
          } else if (result.path) {
            console.log(
              `Saved environment variables to ${result.path} (ensure it's git-ignored).`,
            );
          }
        }

        const quickstartUrl = withUtm(`${WWW}/docs/llm-evaluation/quickstart`, {
          content: "login_success_quickstart",
        });
        console.log(
          "\n🎉🥳 Congratulations! You've successfully logged in! 🙌",
        );
        console.log(
          "You're now using DeepEval with Confident AI. Follow our quickstart " +
            `tutorial here: ${quickstartUrl}`,
        );
      } catch (error) {
        console.log(`Login failed: ${(error as Error).message}`);
        process.exitCode = 1;
      }
    });

  program
    .command("logout")
    .description("Log out of Confident AI.")
    .option(
      "-s, --save [target]",
      "Where to remove the saved key from. Format: dotenv[:path]. The JSON " +
        "keystore is always cleared.",
    )
    .option("-q, --quiet", QUIET_OPTION_HELP)
    .action((options) => {
      const settings = getSettings();
      // Once the files are wiped we can no longer tell a shell export apart
      // from a file-loaded value.
      const keySource = getSettingSource("CONFIDENT_API_KEY");
      const save =
        normalizeSave(options.save) ??
        settings.DEEPEVAL_DEFAULT_SAVE ??
        DEFAULT_SAVE;

      const result = editSettings(
        (draft) => {
          draft.CONFIDENT_API_KEY = null;
        },
        { save },
      );

      // Sweep the whole search path, or a lower-precedence file logs the user
      // straight back in.
      for (const file of [".env", ".env.local"]) {
        const filePath = path.join(process.cwd(), file);
        if (!fs.existsSync(filePath)) continue;
        const contents = fs.readFileSync(filePath, "utf-8");
        if (!/^\s*CONFIDENT_API_KEY\s*=/m.test(contents)) continue;
        new DotenvHandler(filePath).unset(["CONFIDENT_API_KEY"]);
        if (!options.quiet) {
          console.log(`Removed Confident AI key(s) from ${file}.`);
        }
      }

      const shellExport = keySource === "env";
      if (
        handleSaveResult({
          result,
          save,
          quiet: options.quiet,
          updatedMessage: "Removed Confident AI key(s) from {path}.",
        }) &&
        !shellExport
      ) {
        console.log("\n🎉🥳 You've successfully logged out! 🙌");
      }

      if (shellExport && !options.quiet) {
        console.log(
          "\n⚠  CONFIDENT_API_KEY is exported by your shell, which deepeval " +
            "cannot unset — this terminal will still be logged in.",
        );
        console.log(
          "   Finish logging out with: unset CONFIDENT_API_KEY (and remove it " +
            "from your shell profile if it's set there)",
        );
      }

      const latestRun = path.join(
        process.cwd(),
        HIDDEN_DIR,
        LATEST_TEST_RUN_FILE,
      );
      try {
        fs.rmSync(latestRun, { force: true });
      } catch {}
    });
}

export { openBrowser };
