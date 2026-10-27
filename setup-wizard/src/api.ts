import { randomUUID } from "node:crypto";

import { z } from "zod";

const envelope = <T extends z.ZodType>(schema: T) =>
  z.object({ success: z.literal(true), data: schema });

const authSessionSchema = z.object({
  deviceCode: z.string(),
  userCode: z.string(),
  verificationUri: z.string().url(),
  verificationUriComplete: z.string().url(),
  expiresIn: z.number().positive(),
  interval: z.number().positive(),
  protocolVersion: z.number(),
  telemetryToken: z.string().optional(),
});

const authTokenSchema = z.discriminatedUnion("status", [
  z.object({ status: z.literal("pending") }),
  z.object({
    status: z.literal("authenticated"),
    setupToken: z.string(),
    email: z.string().nullable().optional(),
  }),
]);

const questionnaireOptionSchema = z.object({
  label: z.string(),
  value: z.union([z.string(), z.boolean()]),
  exclusive: z.boolean().optional(),
  acceptsCustomValue: z.boolean().optional(),
  customPrompt: z.string().optional(),
});

const questionnaireQuestionSchema = z.object({
  id: z.string(),
  type: z.enum(["text", "single_select", "multi_select"]),
  prompt: z.string(),
  required: z.boolean(),
  defaultValue: z.union([z.string(), z.boolean()]).optional(),
  maxLength: z.number().optional(),
  minSelections: z.number().optional(),
  options: z.array(questionnaireOptionSchema).optional(),
});

const projectSchema = z.object({
  id: z.string(),
  name: z.string(),
  canCreateApiKey: z.boolean(),
});

export const onboardingSchema = z.discriminatedUnion("state", [
  z.object({
    state: z.literal("new_user"),
    user: z.object({
      email: z.string().nullable(),
      name: z.string().nullable(),
    }),
    organization: z.null(),
    projects: z.array(projectSchema).length(0),
    questionnaire: z.object({
      version: z.number().int().positive(),
      questions: z.array(questionnaireQuestionSchema),
    }),
  }),
  z.object({
    state: z.literal("existing_user"),
    user: z.object({
      email: z.string().nullable(),
      name: z.string().nullable(),
    }),
    organization: z.object({ id: z.string(), name: z.string() }).nullable(),
    projects: z.array(projectSchema),
  }),
]);

const completionSchema = z.object({
  status: z.literal("completed"),
  apiKey: z.string().min(1),
  projectId: z.string().min(1),
});

export type AuthSession = z.infer<typeof authSessionSchema>;
export interface AuthSessionContext {
  source: string;
  organizationId?: string;
  projectId?: string;
}
export type Onboarding = z.infer<typeof onboardingSchema>;
export type QuestionnaireAnswer = string | boolean | string[];
export type QuestionnaireAnswers = Record<string, QuestionnaireAnswer>;
export type OnboardingCompletion = z.infer<typeof completionSchema>;

export interface ApiDependencies {
  fetch: typeof globalThis.fetch;
  sleep: (milliseconds: number) => Promise<void>;
  now: () => number;
}

const defaultDependencies: ApiDependencies = {
  fetch: globalThis.fetch,
  sleep: (milliseconds) =>
    new Promise((resolve) => setTimeout(resolve, milliseconds)),
  now: Date.now,
};

const responseError = async (response: Response): Promise<Error> => {
  let message = `Confident API request failed (${response.status}).`;
  try {
    const body: unknown = await response.json();
    const parsed = z
      .object({
        error: z.union([z.string(), z.record(z.string(), z.unknown())]),
      })
      .safeParse(body);
    if (parsed.success && typeof parsed.data.error === "string") {
      message = parsed.data.error;
    }
  } catch {
    // Keep the status-only message when the body is not JSON.
  }
  return new Error(message);
};

export class ConfidentApi {
  readonly #baseUrl: string;
  readonly #dependencies: ApiDependencies;

  constructor(baseUrl: string, dependencies: Partial<ApiDependencies> = {}) {
    this.#baseUrl = baseUrl.replace(/\/+$/, "");
    this.#dependencies = { ...defaultDependencies, ...dependencies };
  }

  async createAuthSession(context: AuthSessionContext): Promise<AuthSession> {
    const response = await this.#dependencies.fetch(
      `${this.#baseUrl}/cli/auth/sessions`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ context }),
      },
    );
    if (!response.ok) throw await responseError(response);
    const body: unknown = await response.json();
    return envelope(authSessionSchema).parse(body).data;
  }

  async pollAuthSession(
    session: AuthSession,
    signal?: AbortSignal,
  ): Promise<{ setupToken: string; email?: string | null }> {
    const deadline = this.#dependencies.now() + session.expiresIn * 1_000;

    while (this.#dependencies.now() < deadline) {
      if (signal?.aborted)
        throw new Error("Browser authorization was cancelled.");
      const response = await this.#dependencies.fetch(
        `${this.#baseUrl}/cli/auth/sessions/token`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ deviceCode: session.deviceCode }),
          ...(signal ? { signal } : {}),
        },
      );
      if (!response.ok) throw await responseError(response);
      const body: unknown = await response.json();
      const token = envelope(authTokenSchema).parse(body).data;
      if (token.status === "authenticated") {
        return {
          setupToken: token.setupToken,
          ...(token.email !== undefined ? { email: token.email } : {}),
        };
      }
      await this.#dependencies.sleep(session.interval * 1_000);
    }
    throw new Error("Browser authorization expired. Run the wizard again.");
  }

  async getOnboarding(setupToken: string): Promise<Onboarding> {
    const response = await this.#dependencies.fetch(
      `${this.#baseUrl}/cli/onboarding`,
      { headers: { authorization: `Bearer ${setupToken}` } },
    );
    if (!response.ok) throw await responseError(response);
    const body: unknown = await response.json();
    return onboardingSchema.parse(body);
  }

  async completeOnboarding(
    setupToken: string,
    body:
      | { projectId: string }
      | {
          questionnaireVersion: number;
          questionnaireAnswers: QuestionnaireAnswers;
        },
    idempotencyKey: string = randomUUID(),
  ): Promise<OnboardingCompletion> {
    const response = await this.#dependencies.fetch(
      `${this.#baseUrl}/cli/onboarding/complete`,
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${setupToken}`,
          "content-type": "application/json",
          "idempotency-key": idempotencyKey,
        },
        body: JSON.stringify(body),
      },
    );
    if (!response.ok) throw await responseError(response);
    const value: unknown = await response.json();
    return completionSchema.parse(value);
  }
}
