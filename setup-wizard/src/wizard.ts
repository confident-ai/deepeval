import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  cancel,
  intro,
  isCancel,
  multiselect,
  note,
  outro,
  password,
  select,
  taskLog,
  text,
} from "@clack/prompts";
import clipboard from "clipboardy";
import open from "open";
import pc from "picocolors";

import {
  checkAgent,
  executeAgent,
  supportedAgents,
  type AgentDefinition,
} from "./agents.js";
import { parseAgentProgressLine } from "./agent-progress.js";
import type {
  Onboarding,
  QuestionnaireAnswer,
  QuestionnaireAnswers,
} from "./api.js";
import { ConfidentApi } from "./api.js";
import type { CliArgs } from "./args.js";
import {
  checkExistingKey,
  connectConfidentAi,
  findExistingKey,
  type CloudCredentials,
  type ExistingKey,
  type KeyCheck,
} from "./credentials.js";
import {
  DEEPEVAL_PACKAGE,
  describeCommands,
  evaluationTarget,
  inspectDeepEval,
  installDeepEval,
  type Ecosystem,
  type EvaluationTarget,
} from "./deepeval.js";
import {
  CONFIDENT_API_KEY,
  ensureEnvLocalIgnored,
  readEnvValues,
  writeEnvValues,
} from "./env.js";
import { restoreTerminalFocus } from "./focus.js";
import { describeGitStatus, inspectGit, runCommand } from "./git.js";
import {
  detectJudge,
  judgeEnvValues,
  judgeMissing,
  judgeProviders,
  providersWithKey,
  type JudgeProvider,
} from "./judge.js";
import {
  applyCustomSelection,
  enforceExclusiveSelection,
  selectProject,
  setQuestionnaireAnswer,
  validateOrganization,
} from "./onboarding.js";
import {
  deferredSetupMessage,
  DEEPEVAL_DOCS_URL,
  EVALUATION_DOCS_URL,
  fullPermissionWarning,
  GITHUB_ISSUE_URL,
  judgeSkipWarning,
  listLabels,
  MANUAL_QUICKSTART_URL,
  promptDeliveryOptions,
  setupModeOptions,
  SUPPORT_URL,
  type PromptDelivery,
  type SetupMode,
} from "./modes.js";
import { buildAgentPrompt } from "./prompt.js";
import {
  readSetupResult,
  type RejectedSetupResult,
  type SetupResult,
} from "./result.js";
import { classifyErrorCode, SetupTelemetry } from "./telemetry.js";
import {
  accent,
  alert,
  banner,
  brand,
  stepHeading,
  wizardStepsFor,
} from "./theme.js";
import { log, spinner } from "./ui.js";
import { verifyTestRun, type VerificationResult } from "./verification.js";

class WizardCancelledError extends Error {}

export interface JudgeSelection {
  provider?: JudgeProvider;
}

/**
 * A prompt can be cancelled from anywhere, so the footer's docs link lives here
 * rather than being threaded through every prompt. It cites DeepEval's own docs
 * until the user opts into Confident AI.
 */
let evaluationDocsUrl = DEEPEVAL_DOCS_URL;

const showCancellation = (message = "Setup cancelled."): void => {
  cancel(
    [
      message,
      "",
      `If you ran into an issue, please open a GitHub issue: ${GITHUB_ISSUE_URL}`,
      "",
      pc.dim(`- Contact support: ${SUPPORT_URL}`),
      pc.dim(`- Evaluation documentation: ${evaluationDocsUrl}`),
    ].join("\n"),
  );
};

const requiredPrompt = <T>(value: T | symbol | undefined): T => {
  if (isCancel(value)) {
    showCancellation();
    throw new WizardCancelledError("Setup cancelled.");
  }
  if (value === undefined) {
    throw new Error("The interactive prompt returned no value.");
  }
  return value as T;
};

export const pairingUrlForApp = (
  appUrl: string,
  verificationUriComplete: string,
): string => {
  const provided = new URL(verificationUriComplete);
  const configured = new URL(appUrl);
  configured.pathname = provided.pathname;
  configured.search = provided.search;
  configured.hash = provided.hash;
  return configured.toString();
};

const ignoreEnvLocal = async (projectDirectory: string): Promise<void> => {
  const changed = await ensureEnvLocalIgnored(projectDirectory, runCommand);
  if (changed) log.info("Added .env.local to .gitignore.");
};

const confirmUnsafeGitState = async (
  status: Awaited<ReturnType<typeof inspectGit>>,
): Promise<void> => {
  const proceed = requiredPrompt<"continue" | "exit">(
    await select({
      message: [
        status.isRepository
          ? `${alert(pc.bold("Git changes detected."))} This repository already has local changes:`
          : `${alert(pc.bold("Warning:"))} This folder is not a Git repository.`,
        "",
        describeGitStatus(status),
        "",
        status.isRepository
          ? `${pc.bold("Setup can continue, but its edits will be mixed with these changes. Continue?")}`
          : pc.bold("Continue without Git safety checks?"),
      ].join("\n"),
      options: [
        {
          label: "Yes",
          value: "continue",
          hint: status.isRepository
            ? "Continue with local changes"
            : "Continue without Git",
        },
        {
          label: status.isRepository
            ? "Exit and protect current changes"
            : "Exit and change directories (recommended)",
          value: "exit",
          hint: status.isRepository
            ? "Commit or stash changes, then rerun setup"
            : "Rerun setup from your project directory",
        },
      ],
    }),
  );
  if (proceed === "exit") {
    showCancellation("Setup cancelled. No changes were made by the wizard.");
    throw new WizardCancelledError("Git preflight declined.");
  }
};

const ECOSYSTEM_LABELS: Record<Ecosystem, string> = {
  python: "Python",
  node: "TypeScript",
};

/**
 * A missing SDK is cheap to fix here and expensive later: without it the agent
 * improvises an install mid-run, or the evaluation fails after a key already
 * exists on disk. The answer also tells the agent which SDK to instrument
 * with, which one prompt cannot infer as reliably as the install can.
 */
const ensureDeepEval = async (
  projectDirectory: string,
): Promise<EvaluationTarget | undefined> => {
  const checking = spinner();
  checking.start("Looking for DeepEval…");
  const { installed, targets, languages } =
    await inspectDeepEval(projectDirectory);
  /**
   * An application in neither language cannot be instrumented, so the wizard
   * says so once here rather than letting the agent discover it mid-run.
   */
  if (languages.length === 0) {
    log.warn(
      [
        "This project is not written in Python or TypeScript, the two languages DeepEval has an SDK for.",
        "The evaluation will call your application from the outside and score its answers, so it grades the whole application rather than its components.",
      ].join("\n"),
    );
  }
  if (installed) {
    checking.stop(`DeepEval found in ${accent(installed.label)}.`);
    return evaluationTarget(languages, installed.ecosystem);
  }
  checking.stop("DeepEval is not installed yet.");

  const [only] = targets;
  const answer = requiredPrompt<Ecosystem | "skip">(
    await select({
      message: [
        `${alert(pc.bold("DeepEval is not installed."))} The evaluation cannot run without it.`,
        "",
        ...targets.flatMap((target) => [
          `Install it into ${accent(target.label)} with:`,
          "",
          describeCommands(target.install),
          "",
        ]),
        pc.bold(
          targets.length > 1
            ? "Which one should the evaluation use?"
            : "Install DeepEval now?",
        ),
      ].join("\n"),
      options: [
        ...targets.map((target) => ({
          label:
            targets.length > 1
              ? `Yes, the ${ECOSYSTEM_LABELS[target.ecosystem]} SDK`
              : "Yes, install DeepEval",
          value: target.ecosystem,
          hint:
            targets.length > 1
              ? `Installs into ${target.label}`
              : "Runs the command above in this project",
        })),
        {
          label: "No, I will install it myself",
          value: "skip" as const,
          hint: "Setup continues, but the evaluation may not run",
        },
      ],
    }),
  );
  if (answer === "skip") {
    log.warn(
      `Install ${DEEPEVAL_PACKAGE} before the evaluation runs, or the run will fail.`,
    );
    /** One target leaves no ambiguity about the language, a declined choice does. */
    return targets.length === 1
      ? evaluationTarget(languages, only?.ecosystem)
      : undefined;
  }

  const target = targets.find(({ ecosystem }) => ecosystem === answer) ?? only;
  if (!target) return undefined;

  const installing = spinner();
  installing.start(`Installing DeepEval into ${target.label}…`);
  try {
    await installDeepEval(target, projectDirectory);
    installing.stop(`DeepEval installed into ${accent(target.label)}.`);
  } catch (error) {
    installing.error("Could not install DeepEval.");
    log.warn(
      [
        error instanceof Error ? error.message : String(error),
        `Install ${DEEPEVAL_PACKAGE} yourself before the evaluation runs.`,
      ].join("\n"),
    );
  }
  return evaluationTarget(languages, target.ecosystem);
};

const confirmConfidentAi = async (): Promise<boolean> => {
  const proceed = requiredPrompt<"yes" | "no">(
    await select({
      message:
        "DeepEval can run evals locally without the platform. Use Confident AI to save results in the cloud?",
      options: [
        {
          label: "Yes, set up Confident AI",
          value: "yes",
          hint: "Sign in, save an API key, then add and verify the evaluation",
        },
        {
          label: "No, keep DeepEval local",
          value: "no",
          hint: "Skip sign-in; still add a local evaluation",
        },
      ],
    }),
  );
  return proceed === "yes";
};

const describeKeySource = (key: ExistingKey): string =>
  key.source === "environment" ? "your environment" : ".env.local";

/** Names where the key was found, never its value. */
const confirmKeyReuse = async (key: ExistingKey): Promise<boolean> => {
  const choice = requiredPrompt<"reuse" | "sign-in">(
    await select({
      message: `Found ${CONFIDENT_API_KEY} in ${describeKeySource(key)}. Which key should the evaluation use?`,
      options: [
        {
          label: "Use the Confident AI key already in this project",
          value: "reuse",
          hint: "Skips sign-in; nothing new is saved",
        },
        {
          label: "Sign in and create a new key",
          value: "sign-in",
          hint: "Creates another project API key",
        },
      ],
    }),
  );
  return choice === "reuse";
};

const checkKeyWithProgress = async (
  apiUrl: string,
  apiKey: string,
  projectId?: string,
): Promise<KeyCheck> => {
  const checking = spinner();
  checking.start("Checking the existing key with Confident AI…");
  const check = await checkExistingKey(apiUrl, apiKey, projectId);
  switch (check.status) {
    case "valid":
      checking.stop(
        "Confident AI accepted the existing key. Skipping sign-in, project, and credentials.",
      );
      break;
    case "rejected":
      checking.error("Confident AI rejected the existing key.");
      log.warn("Sign in to create a new key.");
      break;
    case "wrong-project":
      checking.error(
        `The existing key does not belong to project ${accent(projectId ?? "")}.`,
      );
      log.warn("Sign in to create a key for that project.");
      break;
    case "unverified":
      checking.error(`Could not check the existing key (${check.detail}).`);
      log.warn("Sign in to create a new key instead.");
      break;
  }
  return check;
};

const promptQuestionnaire = async (
  onboarding: Extract<Onboarding, { state: "new_user" }>,
): Promise<QuestionnaireAnswers> => {
  let answers: QuestionnaireAnswers = {};
  for (const question of onboarding.questionnaire.questions) {
    if (question.type === "text") {
      const defaultValue =
        typeof question.defaultValue === "string"
          ? question.defaultValue
          : undefined;
      const answer = requiredPrompt(
        await text({
          message: question.prompt,
          ...(defaultValue !== undefined
            ? { placeholder: defaultValue, defaultValue }
            : {}),
          validate: (value) => {
            const candidate = value ?? "";
            if (question.required && !candidate.trim()) return "Required";
            if (question.maxLength && candidate.length > question.maxLength) {
              return `Maximum ${question.maxLength} characters`;
            }
            return undefined;
          },
        }),
      );
      answers = setQuestionnaireAnswer(answers, question.id, answer);
      continue;
    }

    const options = question.options ?? [];
    if (question.type === "single_select") {
      const answer = requiredPrompt<QuestionnaireAnswer>(
        await select({
          message: question.prompt,
          options: options.map((option) => ({
            label: option.label,
            value: option.value,
          })),
          ...(question.defaultValue !== undefined
            ? { initialValue: question.defaultValue }
            : {}),
        }),
      );
      answers = setQuestionnaireAnswer(answers, question.id, answer);
      continue;
    }

    let selections = requiredPrompt<string[]>(
      await multiselect({
        message: question.prompt,
        options: options.map((option) => ({
          label: option.label,
          value: String(option.value),
        })),
        required: question.required,
      }),
    );
    selections = enforceExclusiveSelection(
      selections,
      options
        .filter((option) => option.exclusive)
        .map((option) => String(option.value)),
    );
    const custom = options.find(
      (option) =>
        option.acceptsCustomValue && selections.includes(String(option.value)),
    );
    if (custom) {
      const customValue = requiredPrompt(
        await text({
          message: custom.customPrompt ?? "Custom value",
          validate: (value) => (value?.trim() ? undefined : "Required"),
        }),
      );
      selections = applyCustomSelection(
        selections,
        String(custom.value),
        customValue,
      );
    }
    answers = setQuestionnaireAnswer(answers, question.id, selections);
  }
  return answers;
};

const chooseAndCompleteProject = async (
  api: ConfidentApi,
  setupToken: string,
  onboarding: Onboarding,
  args: CliArgs,
): Promise<{ apiKey: string; projectId: string }> => {
  validateOrganization(onboarding, args.orgId);

  if (onboarding.state === "new_user") {
    if (args.projId) {
      throw new Error("--proj-id cannot select a project for a new account.");
    }
    note(
      "Your browser login is new to Confident AI. Complete the short project questionnaire.",
      "Create your first evaluation project",
    );
    const questionnaireAnswers = await promptQuestionnaire(onboarding);
    return api.completeOnboarding(setupToken, {
      questionnaireVersion: onboarding.questionnaire.version,
      questionnaireAnswers,
    });
  }

  const preselected = selectProject(onboarding, args.projId);
  const project =
    preselected ??
    requiredPrompt<{ id: string; name: string }>(
      await select({
        message: "Choose the Confident AI project for this evaluation",
        options: onboarding.projects
          .filter((candidate) => candidate.canCreateApiKey)
          .map((candidate) => ({
            label: candidate.name,
            value: { id: candidate.id, name: candidate.name },
          })),
      }),
    );
  return api.completeOnboarding(setupToken, { projectId: project.id });
};

const prepareResultFile = async (): Promise<{
  directory: string;
  path: string;
}> => {
  const directory = await mkdtemp(join(tmpdir(), "deepeval-setup-"));
  return { directory, path: join(directory, "result.json") };
};

const showOwnAgentPrompt = async (
  delivery: PromptDelivery,
  prompt: string,
  resultFile: string,
  useConfidentAi: boolean,
): Promise<void> => {
  if (delivery === "clipboard") {
    try {
      await clipboard.write(prompt);
      log.success("Copied evaluation setup prompt to the clipboard.");
    } catch (error) {
      log.warn(
        `Could not copy the prompt: ${error instanceof Error ? error.message : String(error)}`,
      );
      process.stdout.write(`\n${prompt}\n\n`);
    }
  } else {
    process.stdout.write(`\n${prompt}\n\n`);
  }
  note(
    [
      "Paste the prompt into your coding agent as the next message.",
      "Let it finish building and running the evaluation in this project.",
      "Come back here and confirm when it is done.",
      "",
      ...(useConfidentAi
        ? [
            "DeepEval reads CONFIDENT_API_KEY from your environment or .env.local. Do not ask the agent to read .env.local.",
          ]
        : ["Do not ask the agent to read .env.local."]),
      `The prompt already includes the result path: ${resultFile}`,
    ].join("\n"),
    "What to do next",
  );
};

const summarizeResult = (result: SetupResult, testRunUrl?: string): string => {
  const lines = [
    `Status: ${
      result.status === "completed"
        ? accent(result.status)
        : alert(result.status)
    }`,
    `SDKs: ${result.sdks.join(", ")}`,
    `Shape: ${result.shape}`,
    `Levels: ${result.levels.join(", ")}`,
    `Dataset: ${result.datasetSource}`,
    `Metrics: ${result.metrics.join(", ") || "none"}`,
    `Rerun: ${pc.bold(result.rerunCommand)}`,
  ];
  if (testRunUrl) lines.push(`Test run: ${accent(testRunUrl)}`);
  if (result.errors?.length) lines.push(`Issues: ${result.errors.join("; ")}`);
  return lines.join("\n");
};

const completionOutro = (verified: boolean, useConfidentAi: boolean): string =>
  [
    `${useConfidentAi ? brand("Confident AI") : accent("DeepEval")} ${pc.dim("evaluation setup complete.")}`,
    "",
    verified
      ? `${accent("✔")} Your evaluation is ready to rerun.${
          useConfidentAi ? " Review it in Confident AI." : ""
        }`
      : `Next: finish the evaluation.${
          useConfidentAi ? " Then verify its Confident AI test run." : ""
        }`,
    "",
    `If you encountered an issue, please open a GitHub issue: ${GITHUB_ISSUE_URL}`,
    "",
    pc.dim(`- Contact support: ${SUPPORT_URL}`),
    pc.dim(`- Evaluation documentation: ${evaluationDocsUrl}`),
  ].join("\n");

const verifyTestRunWithProgress = async (
  args: CliArgs,
  cloud: CloudCredentials,
  testRunId: string,
): Promise<VerificationResult | undefined> => {
  const verifying = spinner();
  verifying.start("Verifying Confident AI test run…");
  try {
    const result = await verifyTestRun(
      cloud.apiUrl,
      args.appUrl,
      cloud.apiKey,
      cloud.projectId,
      testRunId,
    );
    verifying.stop(
      result.status && result.status !== "COMPLETED"
        ? `Test run found on Confident AI (status: ${result.status}).`
        : "Confident AI test run verified.",
    );
    return result;
  } catch (error) {
    verifying.error("Test-run verification failed.");
    log.warn(
      [
        error instanceof Error ? error.message : String(error),
        "Your evaluation files and project credentials were left in place.",
      ].join(" "),
    );
    return undefined;
  }
};

/**
 * The suite has already run by the time its result file is read, so a file the
 * contract rejects is a reporting problem, not a failed evaluation. Name what
 * to fix and hand back the test run and rerun command anyway.
 */
const reportRejectedResult = async (
  args: CliArgs,
  rejected: RejectedSetupResult,
  cloud?: CloudCredentials,
): Promise<void> => {
  const { rerunCommand, testRunId, testRunUrl } = rejected.salvaged;
  log.warn(
    [
      "The agent reported a result the setup contract rejects:",
      ...rejected.issues.map((issue) => `  • ${issue}`),
      rerunCommand
        ? `Rerun the evaluation with: ${pc.bold(rerunCommand)}`
        : "Review the evaluation files the agent wrote before rerunning.",
    ].join("\n"),
  );
  if (!cloud || !testRunId) {
    if (testRunUrl) note(accent(testRunUrl), "Reported Confident AI test run");
    return;
  }
  const verification = await verifyTestRunWithProgress(args, cloud, testRunId);
  const verifiedUrl = verification?.testRunUrl ?? testRunUrl;
  if (verification && verifiedUrl) {
    note(accent(verifiedUrl), "Verified Confident AI test run");
  }
};

const executeAgentWithProgress = async (
  agent: AgentDefinition,
  projectDirectory: string,
  prompt: string,
  apiKey: string | undefined,
  resultFile: string,
): Promise<void> => {
  const output = taskLog({
    title: `Running ${agent.label} to build your evaluation`,
    limit: 9,
    spacing: 0,
    retainLog: false,
  });
  let lastLine: string | undefined;
  output.message("Starting agent…");
  try {
    await executeAgent(
      agent,
      projectDirectory,
      prompt,
      apiKey,
      resultFile,
      undefined,
      (line) => {
        for (const message of parseAgentProgressLine(
          agent.kind,
          line,
          projectDirectory,
        )) {
          if (message === lastLine) continue;
          lastLine = message;
          output.message(message);
        }
      },
    );
    output.success(`${agent.label} finished.`);
  } catch (error) {
    output.error(`${agent.label} failed.`);
    throw error;
  }
};

const discoverReadyAgents = async (
  projectDirectory: string,
): Promise<AgentDefinition[]> => {
  const checking = spinner();
  checking.start("Searching for available coding agents…");
  const checks = await Promise.all(
    supportedAgents.map((agent) => checkAgent(agent, projectDirectory)),
  );
  const readyAgents = checks
    .filter((check) => check.readOnlyReady)
    .map((check) => check.agent);
  if (readyAgents.length) {
    checking.stop(
      `Detected ${listLabels(
        readyAgents.map((agent) => agent.label),
        "and",
      )}.`,
    );
    return readyAgents;
  }

  checking.stop("No usable coding agents found.");
  const details = checks.map((check) => {
    const status = !check.discovered
      ? "not installed"
      : !check.authenticated
        ? "not authenticated"
        : "read-only smoke check failed";
    return `- ${check.agent.label}: ${status}`;
  });
  note(
    `${details.join("\n")}\n\nPaste the prompt into your own coding agent, or follow the DeepEval docs instead.`,
    "No agent to launch for you",
  );
  return [];
};

/**
 * DeepEval cannot run an LLM judge without provider credentials, so this either
 * confirms what the environment already offers or collects one key and writes it
 * beside `CONFIDENT_API_KEY`. The secret is only ever passed to the dotenv
 * writer: it is never logged, echoed, or handed to an agent as an argument.
 */
const configureJudgeModel = async (
  projectDirectory: string,
): Promise<JudgeSelection> => {
  const detecting = spinner();
  detecting.start("Looking for judge-model credentials…");
  // DeepEval autoloads `.env.local` but lets the real environment win.
  const env = { ...(await readEnvValues(projectDirectory)), ...process.env };
  const detected = detectJudge(env);
  if (detected && detected.missing.length === 0) {
    const found = detected.provider.secrets
      .filter((secret) => env[secret.envVar]?.trim())
      .map((secret) => secret.envVar);
    detecting.stop(
      `Judge model ready: ${accent(detected.provider.label)}${
        found.length ? ` (${listLabels(found, "and")} found)` : ""
      }.`,
    );
    return { provider: detected.provider };
  }
  // Keys DeepEval would ignore today because their provider flag is unset.
  const reusable = providersWithKey(env).filter(
    (provider) => provider !== detected?.provider,
  );
  const reusableKeys = reusable.flatMap((provider) =>
    provider.secrets
      .filter((secret) => (env[secret.envVar] ?? "").trim())
      .map((secret) => secret.envVar),
  );
  detecting.stop(
    detected
      ? `${detected.provider.label} is selected but incomplete (missing ${detected.missing.join(", ")}).`
      : reusableKeys.length
        ? `Found ${listLabels(reusableKeys, "and")} in this environment, but DeepEval is not set to use ${reusableKeys.length > 1 ? "any of them" : "it"} yet.`
        : "No judge-model credentials found in this environment.",
  );

  const choice = requiredPrompt<JudgeProvider | "skip">(
    await select<JudgeProvider | "skip">({
      message: "Which model should judge your evaluation?",
      maxItems: 8,
      ...(detected || reusable.length
        ? { initialValue: detected?.provider ?? reusable[0]! }
        : {}),
      options: [
        // Providers whose key is already here come first, so the common case
        // is one keystroke rather than a paste.
        ...reusable.map((provider) => ({
          label: provider.label,
          value: provider,
          hint: `Reuse the ${provider.secrets
            .filter((secret) => (env[secret.envVar] ?? "").trim())
            .map((secret) => secret.envVar)
            .join(" and ")} already in your environment`,
        })),
        ...judgeProviders
          .filter((provider) => !reusable.includes(provider))
          .map((provider) => ({
            label: provider.label,
            value: provider,
            hint: provider.hint,
          })),
        {
          label: "Skip for now",
          value: "skip" as const,
          hint: "Deterministic metrics only until you add a key",
        },
      ],
    }),
  );
  if (choice === "skip") {
    log.warn(judgeSkipWarning);
    return {};
  }

  // Ask only for what this environment cannot already supply.
  const needed = judgeMissing(choice, env);
  const secrets: Record<string, string> = {};
  for (const secret of choice.secrets) {
    if ((env[secret.envVar] ?? "").trim()) continue;
    const required = needed.includes(secret.envVar);
    const answer = requiredPrompt(
      await password({
        message: required
          ? `Paste your ${secret.label} (saved to .env.local, never displayed)`
          : `Paste your ${secret.label}, or leave it empty to skip`,
        ...(required
          ? {
              validate: (value: string | undefined) =>
                value?.trim() ? undefined : "A value is required",
            }
          : {}),
      }),
    ).trim();
    if (answer) secrets[secret.envVar] = answer;
  }
  const settings: Record<string, string> = {};
  for (const setting of choice.settings.filter((candidate) =>
    needed.includes(candidate.envVar),
  )) {
    settings[setting.envVar] = requiredPrompt(
      await text({
        message: setting.label,
        ...(setting.placeholder ? { placeholder: setting.placeholder } : {}),
        validate: (value) => (value?.trim() ? undefined : "Required"),
      }),
    ).trim();
  }

  const values = judgeEnvValues(choice, { secrets, settings });
  if (Object.keys(values).length)
    await writeEnvValues(projectDirectory, values);
  const reused = choice.secrets
    .filter((secret) => !secrets[secret.envVar] && env[secret.envVar]?.trim())
    .map((secret) => secret.envVar);
  log.success(
    reused.length
      ? `Selected ${choice.label} in .env.local. DeepEval reads ${listLabels(reused, "and")} from your environment.`
      : `Saved ${choice.label} judge-model settings to .env.local. DeepEval loads them from there.`,
  );
  return { provider: choice };
};

const runBuiltInMode = async (
  args: CliArgs,
  readyAgents: AgentDefinition[],
  judge: JudgeSelection,
  cloud?: CloudCredentials,
  evaluation?: EvaluationTarget,
): Promise<boolean> => {
  if (!readyAgents.length) {
    throw new Error("No built-in coding agent is available.");
  }
  const agent =
    readyAgents.length === 1
      ? readyAgents[0]!
      : requiredPrompt<AgentDefinition>(
          await select({
            message: "Which detected agent should we launch?",
            options: readyAgents.map((candidate) => ({
              label: candidate.label,
              value: candidate,
              hint: `We run ${candidate.label} for you here`,
            })),
          }),
        );

  const permissionChoice = requiredPrompt<"proceed" | "cancel">(
    await select({
      message: `${fullPermissionWarning(agent.label)}\n\n${pc.bold("Proceed?")}`,
      options: [
        {
          label: "Confirm",
          value: "proceed",
          hint: `Run ${agent.label}`,
        },
        {
          label: "Cancel setup",
          value: "cancel",
          hint: "Exit without running the coding agent",
        },
      ],
    }),
  );
  if (permissionChoice === "cancel") {
    showCancellation("Setup cancelled. The coding agent was not started.");
    throw new WizardCancelledError("Full-permission execution declined.");
  }

  const resultFile = await prepareResultFile();
  const prompt = buildAgentPrompt(
    args.projectDir,
    resultFile.path,
    judge,
    Boolean(cloud),
    evaluation,
  );
  try {
    await executeAgentWithProgress(
      agent,
      args.projectDir,
      prompt,
      cloud?.apiKey,
      resultFile.path,
    );
    const outcome = await readSetupResult(resultFile.path, Boolean(cloud));
    if (!outcome.ok) {
      await reportRejectedResult(args, outcome.rejected, cloud);
      return false;
    }
    const { result } = outcome;
    const verification =
      cloud && result.status === "completed" && result.testRunId
        ? await verifyTestRunWithProgress(args, cloud, result.testRunId)
        : undefined;
    // A reused key has no known project, so the agent's own link stands.
    if (verification?.testRunUrl) result.testRunUrl = verification.testRunUrl;
    note(summarizeResult(result, result.testRunUrl), "Evaluation setup result");
    return cloud ? Boolean(verification) : result.status === "completed";
  } finally {
    await rm(resultFile.directory, { recursive: true, force: true });
  }
};

const finishOwnAgentMode = async (
  args: CliArgs,
  resultFile: { directory: string; path: string },
  cloud?: CloudCredentials,
): Promise<boolean> => {
  const decision = requiredPrompt<"finished" | "later">(
    await select({
      message:
        "After you paste the prompt into your coding agent and it finishes, continue here.",
      options: [
        {
          label: "The agent finished",
          value: "finished",
          hint: "Read the structured result file",
        },
        {
          label: "I'll finish later",
          value: "later",
          hint: "Keep the prompt and continue later",
        },
      ],
    }),
  );
  if (decision === "later") {
    log.info(`Finish later using the prompt's result path: ${resultFile.path}`);
    return false;
  }

  try {
    const outcome = await readSetupResult(resultFile.path, Boolean(cloud));
    if (!outcome.ok) {
      await reportRejectedResult(args, outcome.rejected, cloud);
      return false;
    }
    const { result } = outcome;
    const verification =
      cloud && result.testRunId
        ? await verifyTestRunWithProgress(args, cloud, result.testRunId)
        : undefined;
    if (verification?.testRunUrl) result.testRunUrl = verification.testRunUrl;
    note(summarizeResult(result, result.testRunUrl), "Evaluation setup result");
    return cloud
      ? result.status === "completed" && Boolean(verification)
      : result.status === "completed";
  } finally {
    await rm(resultFile.directory, { recursive: true, force: true });
  }
};

const finishManualMode = async (
  args: CliArgs,
  cloud?: CloudCredentials,
): Promise<boolean> => {
  const docsUrl = cloud ? MANUAL_QUICKSTART_URL : DEEPEVAL_DOCS_URL;
  const decision = requiredPrompt<"verify" | "later">(
    await select({
      message: [
        "Follow the evaluation quickstart for your project:",
        accent(docsUrl),
        "",
        pc.bold("Did you complete and run the evaluation?"),
      ].join("\n"),
      options: [
        {
          label: "Yes, the evaluation ran",
          value: "verify",
          hint: "Continue",
        },
        {
          label: "Finish evaluation later",
          value: "later",
          hint: "Exit and continue later",
        },
      ],
    }),
  );
  if (decision === "later") return false;
  if (!cloud) return true;

  const testRunId = requiredPrompt(
    await text({
      message: "Test-run ID",
      validate: (value) => (value?.trim() ? undefined : "Required"),
    }),
  ).trim();
  const verification = await verifyTestRunWithProgress(args, cloud, testRunId);
  if (!verification) return false;
  if (verification.testRunUrl) {
    note(accent(verification.testRunUrl), "Verified Confident AI test run");
  }
  return true;
};

/**
 * Steps 1-3: browser pairing, project choice, and saving a new project key.
 * Only this path opens an auth session, so only it carries telemetry.
 */
const signInAndSaveKey = async (
  args: CliArgs,
  telemetry: SetupTelemetry,
  heading: (step: number, detail?: string) => string,
  existing: ExistingKey | undefined,
): Promise<{ apiKey: string; projectId: string }> => {
  log.message(heading(1));
  const api = new ConfidentApi(args.apiUrl);
  const session = await api.createAuthSession({
    source: args.from,
    ...(args.orgId ? { organizationId: args.orgId } : {}),
    ...(args.projId ? { projectId: args.projId } : {}),
  });
  telemetry.setTelemetryToken(session.telemetryToken);
  await telemetry.send({
    event: "wizard_started",
    step: "bootstrap",
    result: "succeeded",
  });
  await telemetry.send({
    event: "authentication_started",
    step: "authentication",
    result: "started",
  });
  const pairingUrl = pairingUrlForApp(
    args.appUrl,
    session.verificationUriComplete,
  );
  log.info(
    [
      pc.bold(
        "Sign in to continue setup. Your browser should have opened automatically.",
      ),
      "",
      `Verification code: ${pc.bold(session.userCode)}`,
      "",
      pc.dim(
        "If your browser did not open automatically, open the link below:",
      ),
      accent(pairingUrl),
    ].join("\n"),
  );
  await open(pairingUrl).catch(() => {
    log.warn("Could not open the browser automatically. Use the URL above.");
  });

  const waiting = spinner();
  waiting.start("Waiting for browser setup (the link remains valid briefly)…");
  const { authorization, onboarding } = await (async () => {
    try {
      const authorized = await api.pollAuthSession(session);
      const state = await api.getOnboarding(authorized.setupToken);
      return { authorization: authorized, onboarding: state };
    } catch (error) {
      waiting.error("Browser setup stopped.");
      throw error;
    }
  })();
  await restoreTerminalFocus();
  const browserProject =
    onboarding.state === "existing_user"
      ? onboarding.projects.find(
          (candidate) =>
            candidate.id === args.projId ||
            (onboarding.projects.length === 1 && candidate.canCreateApiKey),
        )
      : undefined;
  waiting.stop(
    onboarding.state === "existing_user" &&
      onboarding.organization &&
      browserProject
      ? `Browser setup complete. (org: ${accent(onboarding.organization.name)}, project: ${accent(browserProject.name)})`
      : authorization.email
        ? `Browser sign-in complete (${authorization.email}).`
        : "Browser sign-in complete.",
  );
  await telemetry.send({
    event: "authentication_completed",
    step: "authentication",
    result: "succeeded",
  });

  log.message(heading(2));
  const completion = await chooseAndCompleteProject(
    api,
    authorization.setupToken,
    onboarding,
    args,
  );
  if (onboarding.state === "existing_user") {
    const project = onboarding.projects.find(
      (candidate) => candidate.id === completion.projectId,
    );
    const labels = [onboarding.organization?.name, project?.name].filter(
      Boolean,
    );
    if (labels.length && project?.id !== browserProject?.id) {
      log.success(`Project setup complete (${labels.join(" / ")}).`);
    }
  }

  log.message(heading(3));
  await writeEnvValues(args.projectDir, {
    [CONFIDENT_API_KEY]: completion.apiKey,
  });
  log.success("Saved project credentials securely to .env.local.");
  if (existing?.source === "environment") {
    log.warn(
      `${CONFIDENT_API_KEY} is also set in your environment, and DeepEval prefers it over .env.local. Unset it so DeepEval uses the new key.`,
    );
  }
  await ignoreEnvLocal(args.projectDir);
  return { apiKey: completion.apiKey, projectId: completion.projectId };
};

export const runWizard = async (args: CliArgs): Promise<void> => {
  const directory = await stat(args.projectDir).catch(() => undefined);
  if (!directory?.isDirectory()) {
    throw new Error(`Project directory does not exist: ${args.projectDir}`);
  }

  intro(pc.bold("Evaluation setup"));
  const useConfidentAi = await confirmConfidentAi();
  evaluationDocsUrl = useConfidentAi ? EVALUATION_DOCS_URL : DEEPEVAL_DOCS_URL;

  const telemetry = new SetupTelemetry(args.apiUrl);
  process.stdout.write(
    `\n${banner(process.stdout.columns ?? 80, useConfidentAi)}\n\n`,
  );
  const steps = wizardStepsFor(useConfidentAi);
  const heading = (step: number, detail?: string): string =>
    stepHeading(step, detail, steps);

  try {
    const gitStatus = await inspectGit(args.projectDir);
    if (!gitStatus.isRepository || gitStatus.dirty) {
      await confirmUnsafeGitState(gitStatus);
    }
    const evaluation = await ensureDeepEval(args.projectDir);

    let cloud: CloudCredentials | undefined;
    if (useConfidentAi) {
      cloud = await connectConfidentAi(args, {
        findKey: (projectDirectory) => findExistingKey(projectDirectory),
        chooseReuse: confirmKeyReuse,
        checkKey: checkKeyWithProgress,
        signIn: (existing) =>
          signInAndSaveKey(args, telemetry, heading, existing),
      });
    }

    log.message(heading(useConfidentAi ? 4 : 1));
    const judge = await configureJudgeModel(args.projectDir);
    // Saving a new project key already ignored the file.
    if (!cloud || cloud.reused) await ignoreEnvLocal(args.projectDir);

    log.message(heading(useConfidentAi ? 5 : 2));
    const readyAgents = await discoverReadyAgents(args.projectDir);
    const mode = requiredPrompt<SetupMode>(
      await select({
        message: "How should we add the evaluation?",
        options: setupModeOptions(readyAgents).filter(
          (option) => option.value !== "built-in" || readyAgents.length > 0,
        ),
      }),
    );
    await telemetry.send({
      event: "setup_started",
      step: "configuration",
      result: "started",
    });

    log.message(
      heading(
        useConfidentAi ? 6 : 3,
        mode === "built-in"
          ? "Launch the detected agent"
          : mode === "own-agent"
            ? "Paste the prompt into your agent"
            : "Follow the evaluation quickstart",
      ),
    );
    if (mode === "built-in") {
      const verified = await runBuiltInMode(
        args,
        readyAgents,
        judge,
        cloud,
        evaluation,
      );
      await telemetry.send({
        event: "setup_completed",
        step: "evaluation",
        result: verified ? "succeeded" : "failed",
      });
      outro(completionOutro(verified, useConfidentAi));
      return;
    }

    let setupVerified = false;
    if (mode === "own-agent") {
      const resultFile = await prepareResultFile();
      const delivery = requiredPrompt<PromptDelivery>(
        await select({
          message: "How should we give you the prompt to paste?",
          options: promptDeliveryOptions,
        }),
      );
      await showOwnAgentPrompt(
        delivery,
        buildAgentPrompt(
          args.projectDir,
          resultFile.path,
          judge,
          useConfidentAi,
          evaluation,
        ),
        resultFile.path,
        useConfidentAi,
      );
      setupVerified = await finishOwnAgentMode(args, resultFile, cloud);
      if (!setupVerified) log.info(deferredSetupMessage(mode, useConfidentAi));
    } else {
      await open(cloud ? MANUAL_QUICKSTART_URL : DEEPEVAL_DOCS_URL).catch(
        () => undefined,
      );
      setupVerified = await finishManualMode(args, cloud);
      if (!setupVerified) log.info(deferredSetupMessage(mode, useConfidentAi));
    }
    await telemetry.send({
      event: "setup_completed",
      step: setupVerified ? "evaluation" : "configuration",
      result: setupVerified ? "succeeded" : "cancelled",
    });
    outro(completionOutro(setupVerified, useConfidentAi));
  } catch (error) {
    if (error instanceof WizardCancelledError) throw error;
    await telemetry.send({
      event: "setup_failed",
      step: "configuration",
      result: "failed",
      errorCode: classifyErrorCode(error),
    });
    throw error;
  }
};

export const isWizardCancellation = (
  error: unknown,
): error is WizardCancelledError => error instanceof WizardCancelledError;
