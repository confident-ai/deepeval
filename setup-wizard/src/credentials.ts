import { CONFIDENT_API_KEY, readEnvValues } from "./env.js";

/**
 * Every `/cli/onboarding/complete` mints another project key, so a project that
 * already holds one is offered it first. Only the key's presence and where it
 * was found ever leave this module; its value goes to the check request alone.
 */
export interface ExistingKey {
  apiKey: string;
  source: "environment" | ".env.local";
}

/**
 * DeepEval lets the real environment win over `.env.local`, so this looks in
 * the same order and reports the key DeepEval would actually use.
 */
export const findExistingKey = async (
  projectDirectory: string,
  env: NodeJS.ProcessEnv = process.env,
  readValues: (
    directory: string,
  ) => Promise<Record<string, string>> = readEnvValues,
): Promise<ExistingKey | undefined> => {
  const fromEnvironment = env[CONFIDENT_API_KEY]?.trim();
  if (fromEnvironment) {
    return { apiKey: fromEnvironment, source: "environment" };
  }
  const fromFile = (await readValues(projectDirectory))[
    CONFIDENT_API_KEY
  ]?.trim();
  return fromFile ? { apiKey: fromFile, source: ".env.local" } : undefined;
};

const DEFAULT_API_URL = "https://api.confident-ai.com";
const EU_API_URL = "https://eu.api.confident-ai.com";

/**
 * DeepEval sends a `confident_eu_` key to the EU API when nothing else names a
 * region, so a check against the default US API would wrongly reject it. An
 * explicit `--api-url` is left alone.
 */
export const apiUrlForKey = (apiUrl: string, apiKey: string): string =>
  apiUrl === DEFAULT_API_URL &&
  apiKey.trim().toLowerCase().startsWith("confident_eu_")
    ? EU_API_URL
    : apiUrl;

/**
 * A project-key GET with no plan gate of its own. The test-run routes answer a
 * free plan with 401, which would read as a rejected key.
 */
export const KEY_CHECK_PATH = "/v1/metric-collections";

export type KeyCheck =
  | { status: "valid" }
  | { status: "rejected" }
  | { status: "wrong-project" }
  | { status: "unverified"; detail: string };

export const checkExistingKey = async (
  apiUrl: string,
  apiKey: string,
  projectId?: string,
  fetchImplementation: typeof globalThis.fetch = globalThis.fetch,
  timeoutMs = 15_000,
): Promise<KeyCheck> => {
  let response: Response;
  try {
    response = await fetchImplementation(
      `${apiUrl.replace(/\/+$/, "")}${KEY_CHECK_PATH}`,
      {
        headers: {
          CONFIDENT_API_KEY: apiKey,
          ...(projectId ? { "confident-project-id": projectId } : {}),
        },
        signal: AbortSignal.timeout(timeoutMs),
      },
    );
  } catch {
    // The error is never echoed: it is about the request, not the key.
    return {
      status: "unverified",
      detail: "Confident AI could not be reached",
    };
  }
  await response.body?.cancel().catch(() => undefined);

  // A plan or feature gate (403) runs only after the key itself was accepted.
  if (response.ok || response.status === 403) return { status: "valid" };
  if (response.status === 401) return { status: "rejected" };
  // The backend answers a project id that is not the key's own with 400.
  if (projectId && (response.status === 400 || response.status === 404)) {
    return { status: "wrong-project" };
  }
  return {
    status: "unverified",
    detail: `Confident AI answered ${response.status}`,
  };
};

export interface CloudCredentials {
  apiKey: string;
  /** The API the key belongs to, which differs from `--api-url` for EU keys. */
  apiUrl: string;
  /**
   * Unknown for a reused key unless `--proj-id` named it: no public endpoint
   * maps a project key to its project, so links come from the agent instead.
   */
  projectId?: string;
  reused: boolean;
}

export interface ConnectRequest {
  projectDir: string;
  apiUrl: string;
  orgId?: string;
  projId?: string;
}

export interface ConnectDependencies {
  findKey: (projectDirectory: string) => Promise<ExistingKey | undefined>;
  chooseReuse: (key: ExistingKey) => Promise<boolean>;
  checkKey: (
    apiUrl: string,
    apiKey: string,
    projectId?: string,
  ) => Promise<KeyCheck>;
  /** Pairing, project choice and saving: steps 1-3. */
  signIn: (
    existing: ExistingKey | undefined,
  ) => Promise<{ apiKey: string; projectId: string }>;
}

/**
 * Reuse a key the backend accepts, otherwise sign in. `--org-id` alone cannot
 * be checked against a project key, so it always signs in; `--proj-id` is
 * checked by the backend itself.
 */
export const connectConfidentAi = async (
  request: ConnectRequest,
  dependencies: ConnectDependencies,
): Promise<CloudCredentials> => {
  const existing =
    request.orgId && !request.projId
      ? undefined
      : await dependencies.findKey(request.projectDir);

  if (existing && (await dependencies.chooseReuse(existing))) {
    const apiUrl = apiUrlForKey(request.apiUrl, existing.apiKey);
    const check = await dependencies.checkKey(
      apiUrl,
      existing.apiKey,
      request.projId,
    );
    if (check.status === "valid") {
      return {
        apiKey: existing.apiKey,
        apiUrl,
        reused: true,
        ...(request.projId ? { projectId: request.projId } : {}),
      };
    }
  }

  const created = await dependencies.signIn(existing);
  return { ...created, apiUrl: request.apiUrl, reused: false };
};
