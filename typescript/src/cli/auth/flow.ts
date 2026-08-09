// Browser authentication and CLI onboarding for `deepeval login`, ported from
// deepeval/cli/auth/flow.py. Device-code pairing (RFC 8628 shaped) with no
// localhost callback server, so the flow works over SSH and in containers.

import axios from "axios";
import { getBaseApiUrl } from "@/confident/api";
import { getVersion } from "@/cli/version";
import { openBrowser, withUtm } from "@/cli/utm";
import { prompt } from "@/cli/utils";

const CREATE_PAIRING_ENDPOINT = "/cli/auth/sessions";
const PAIRING_TOKEN_ENDPOINT = "/cli/auth/sessions/token";
const ONBOARDING_ENDPOINT = "/cli/onboarding";
const ONBOARDING_COMPLETE_ENDPOINT = "/cli/onboarding/complete";

const REQUEST_TIMEOUT_MS = 10_000;
// Completion creates the org and project in one backend transaction, which can
// outlast a normal request.
const COMPLETE_TIMEOUT_MS = 30_000;
const COMPLETE_MAX_ATTEMPTS = 3;

const DEFAULT_EXPIRES_IN_SECONDS = 600;
const DEFAULT_POLL_INTERVAL_SECONDS = 3;

export class AuthFlowError extends Error {}

export interface DevicePairing {
  userCode: string;
  deviceCode: string;
  verificationUrl: string;
  expiresIn: number;
  interval: number;
}

export interface CliAuthorization {
  setupToken: string;
  email?: string;
}

export interface CliOnboardingProject {
  id: string;
  name: string;
  canCreateApiKey: boolean;
}

export type QuestionnaireAnswer = string | boolean | string[];

export interface CliQuestionnaireOption {
  label: string;
  value: string | boolean;
  exclusive?: boolean;
  acceptsCustomValue?: boolean;
  customPrompt?: string;
}

export interface CliQuestionnaireQuestion {
  id: string;
  prompt: string;
  required: boolean;
  type: "text" | "single_select" | "multi_select";
  options?: CliQuestionnaireOption[];
  defaultValue?: string | boolean;
  maxLength?: number;
  minSelections?: number;
}

export interface CliQuestionnaire {
  version: number;
  questions: CliQuestionnaireQuestion[];
}

export interface CliOnboardingContext {
  state: "new_user" | "existing_user";
  projects: CliOnboardingProject[];
  questionnaire?: CliQuestionnaire;
}

function requestHeaders(): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "X-DeepEval-Version": getVersion(),
  };
}

/** Unwraps the `{success, data, error}` envelope, tolerating a bare payload. */
function unwrap(data: unknown, what: string): Record<string, unknown> {
  if (!data || typeof data !== "object") {
    throw new AuthFlowError(`${what} returned an unexpected payload.`);
  }
  const payload = data as Record<string, unknown>;
  if ("success" in payload) {
    if (!payload.success) {
      throw new AuthFlowError(
        (payload.error as string) || "The pairing request failed.",
      );
    }
    const inner = payload.data;
    return inner && typeof inner === "object"
      ? (inner as Record<string, unknown>)
      : {};
  }
  return payload;
}

async function createPairing(): Promise<DevicePairing> {
  const url = `${getBaseApiUrl()}${CREATE_PAIRING_ENDPOINT}`;
  let response;
  try {
    response = await axios.post(
      url,
      {},
      { headers: requestHeaders(), timeout: REQUEST_TIMEOUT_MS },
    );
  } catch (error) {
    const status = (error as { response?: { status?: number } }).response
      ?.status;
    if (status === 404) {
      throw new AuthFlowError(
        `The backend does not support browser login yet (404 from POST ${url}).`,
      );
    }
    throw new AuthFlowError(`Could not reach POST ${url}.`);
  }

  const data = unwrap(response.data, `POST ${url}`);
  const userCode = data.userCode as string | undefined;
  const deviceCode = data.deviceCode as string | undefined;
  const verificationUrl = data.verificationUriComplete as string | undefined;
  if (!userCode || !deviceCode || !verificationUrl) {
    throw new AuthFlowError(
      `POST ${url} did not return the required login session fields.`,
    );
  }
  return {
    userCode,
    deviceCode,
    verificationUrl,
    expiresIn: (data.expiresIn as number) ?? DEFAULT_EXPIRES_IN_SECONDS,
    interval: (data.interval as number) ?? DEFAULT_POLL_INTERVAL_SECONDS,
  };
}

/** `null` while pending or on a transient error; throws on terminal failures. */
async function pollOnce(deviceCode: string): Promise<CliAuthorization | null> {
  const url = `${getBaseApiUrl()}${PAIRING_TOKEN_ENDPOINT}`;
  let response;
  try {
    response = await axios.post(
      url,
      { deviceCode },
      { headers: requestHeaders(), timeout: REQUEST_TIMEOUT_MS },
    );
  } catch (error) {
    const status = (error as { response?: { status?: number } }).response
      ?.status;
    if (status === 404) {
      throw new AuthFlowError(
        "This pairing is no longer valid, or the backend does not support " +
          "browser login yet.",
      );
    }
    return null;
  }

  const data = unwrap(response.data, `POST ${url}`);
  const status = data.status as string | undefined;
  if (status === "pending") return null;
  if (status === "authenticated") {
    const setupToken = data.setupToken as string | undefined;
    if (!setupToken) {
      throw new AuthFlowError(
        "Browser authentication completed but the server did not return a " +
          "setup token.",
      );
    }
    return { setupToken, email: data.email as string | undefined };
  }
  throw new AuthFlowError(
    `The pairing is no longer valid (status: ${status ?? "unknown"}). ` +
      "Run `deepeval login` again.",
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Returns `null` when the flow was aborted, timed out, or is unavailable. */
export async function browserPairingLogin(): Promise<CliAuthorization | null> {
  let pairing: DevicePairing;
  try {
    pairing = await createPairing();
  } catch {
    console.log(
      "\n⚠️  Unexpected error — seems like browser login isn't available right now.",
    );
    return null;
  }

  const loginUrl = withUtm(pairing.verificationUrl, {
    content: "login_pair_browser_open",
  });
  const fallbackUrl = withUtm(pairing.verificationUrl, {
    content: "login_pair_fallback_link",
  });

  console.log("\n🌐 Opening your browser — confirm this pairing code there:");
  console.log(`\n    ${pairing.userCode}\n`);
  await openBrowser(loginUrl);
  console.log(`Browser didn't open? Use ${fallbackUrl}`);
  console.log("Waiting for the browser... press Ctrl+C to stop.");

  const deadline = Date.now() + pairing.expiresIn * 1000;
  while (Date.now() < deadline) {
    await sleep(pairing.interval * 1000);
    let completed: CliAuthorization | null;
    try {
      completed = await pollOnce(pairing.deviceCode);
    } catch (error) {
      console.log(`\n⚠️  ${(error as Error).message}`);
      return null;
    }
    if (completed) return completed;
  }
  console.log("\n⌛ Timed out waiting for the browser.");
  return null;
}

export async function getCliOnboardingContext(
  setupToken: string,
): Promise<CliOnboardingContext> {
  const url = `${getBaseApiUrl()}${ONBOARDING_ENDPOINT}`;
  let response;
  try {
    response = await axios.get(url, {
      headers: { ...requestHeaders(), Authorization: `Bearer ${setupToken}` },
      timeout: REQUEST_TIMEOUT_MS,
    });
  } catch (error) {
    throw new AuthFlowError(
      `Could not reach GET ${url} (${(error as Error).message}).`,
    );
  }
  const data = response.data as Partial<CliOnboardingContext> | undefined;
  if (!data || (data.state !== "new_user" && data.state !== "existing_user")) {
    throw new AuthFlowError(`GET ${url} returned an unexpected payload.`);
  }
  return {
    state: data.state,
    projects: data.projects ?? [],
    questionnaire: data.questionnaire,
  };
}

export type OnboardingRequest =
  | {
      questionnaireVersion: number;
      questionnaireAnswers: Record<string, QuestionnaireAnswer>;
    }
  | { projectId: string };

export async function completeCliOnboarding(
  setupToken: string,
  request: OnboardingRequest,
  idempotencyKey: string,
): Promise<string> {
  const url = `${getBaseApiUrl()}${ONBOARDING_COMPLETE_ENDPOINT}`;
  let lastError: unknown = null;

  for (let attempt = 0; attempt < COMPLETE_MAX_ATTEMPTS; attempt++) {
    const retryable = attempt < COMPLETE_MAX_ATTEMPTS - 1;
    try {
      const response = await axios.post(url, request, {
        headers: {
          ...requestHeaders(),
          Authorization: `Bearer ${setupToken}`,
          "Idempotency-Key": idempotencyKey,
        },
        timeout: COMPLETE_TIMEOUT_MS,
      });
      const apiKey = (response.data as { apiKey?: string } | undefined)?.apiKey;
      if (!apiKey) {
        throw new AuthFlowError(
          "CLI onboarding completed without returning an API key.",
        );
      }
      return apiKey;
    } catch (error) {
      if (error instanceof AuthFlowError) throw error;
      lastError = error;
      const response = (
        error as { response?: { status?: number; data?: unknown } }
      ).response;
      // A timed-out attempt may still hold the completion lock server-side;
      // the idempotency key makes retrying safe.
      const inProgress =
        response?.status === 409 &&
        JSON.stringify(response.data ?? "").includes("in progress");
      if (retryable) {
        if (!inProgress) console.log("Still working — retrying...");
        await sleep(inProgress ? 3000 : 2000);
        continue;
      }
    }
  }

  throw new AuthFlowError(
    `Could not complete POST ${url} (${
      lastError instanceof Error ? lastError.message : "unknown error"
    }).`,
  );
}

// Numbered prompts, the fallback questionary degrades to in Python when the
// terminal isn't interactive.

export async function promptText(
  message: string,
  defaultValue?: string,
): Promise<string> {
  const suffix = defaultValue ? ` [${defaultValue}]` : "";
  const answer = await prompt(`? ${message}${suffix} `);
  return answer || defaultValue || "";
}

export async function promptSelect<T>(
  message: string,
  choices: Array<[string, T]>,
): Promise<T> {
  console.log(message);
  choices.forEach(([label], index) => console.log(`  ${index + 1}. ${label}`));
  for (;;) {
    const raw = await prompt("Enter a number: ");
    const selected = Number(raw);
    if (
      Number.isInteger(selected) &&
      selected >= 1 &&
      selected <= choices.length
    ) {
      return choices[selected - 1][1];
    }
    console.log(`❌ Please enter a number between 1 and ${choices.length}.`);
  }
}

export async function promptCheckbox<T>(
  message: string,
  choices: Array<[string, T]>,
  minSelections = 1,
): Promise<T[]> {
  console.log(message);
  choices.forEach(([label], index) => console.log(`  ${index + 1}. ${label}`));
  for (;;) {
    const raw = await prompt(
      minSelections === 0
        ? "Enter numbers (comma-separated), or leave blank: "
        : "Enter one or more numbers (comma-separated): ",
    );
    if (!raw.trim() && minSelections === 0) return [];
    const parsed = raw
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isInteger(value));
    const unique = new Set(parsed);
    const valid =
      parsed.length >= minSelections &&
      unique.size === parsed.length &&
      parsed.every((value) => value >= 1 && value <= choices.length);
    if (valid) return parsed.map((value) => choices[value - 1][1]);
    console.log(
      `❌ Select at least ${minSelections} unique number(s) between 1 and ` +
        `${choices.length}, separated by commas.`,
    );
  }
}
