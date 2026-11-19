/**
 * Judge-model credentials for DeepEval's LLM metrics.
 *
 * DeepEval resolves a judge provider by first-match on its `USE_*` flags and
 * otherwise falls back to OpenAI, reading every secret from the process
 * environment or an autoloaded dotenv (`.env.local` wins among files). This
 * mirrors that resolution so the wizard can tell whether metrics can actually
 * run before an agent spends a turn finding out.
 *
 * The catalog covers every provider DeepEval's `initialize_model` can select.
 * `settings` lists only what a provider has no default for, so a provider with
 * a built-in default model asks the user nothing beyond its key.
 */

export type JudgeProviderId =
  | "openai"
  | "anthropic"
  | "gemini"
  | "azure-openai"
  | "bedrock"
  | "openrouter"
  | "deepseek"
  | "grok"
  | "moonshot"
  | "litellm"
  | "portkey"
  | "ollama"
  | "local";

export interface JudgeSecret {
  envVar: string;
  label: string;
  /** DeepEval can authenticate without it, e.g. an ambient AWS role. */
  optional?: boolean;
  /** Optional only while this variable is set, e.g. Gemini through Vertex AI. */
  optionalWhen?: string;
  /** Values that mark a different provider rather than a key for this one. */
  reservedValues?: readonly string[];
}

export interface JudgeSetting {
  envVar: string;
  label: string;
  placeholder?: string;
}

export interface JudgeProvider {
  id: JudgeProviderId;
  label: string;
  hint: string;
  secrets: JudgeSecret[];
  /** Non-secret values DeepEval requires because it has no default for them. */
  settings: JudgeSetting[];
  /**
   * Written verbatim so DeepEval selects this provider, and matched to detect
   * it. `"true"` matches DeepEval's boolean parsing; anything else is exact.
   * OpenAI needs no flag because it is DeepEval's fallback.
   */
  flags: Record<string, string>;
  /** Position in DeepEval's `initialize_model` chain, which is first-match. */
  precedence: number;
}

const apiKey = (envVar: string, label: string): JudgeSecret => ({
  envVar,
  label: `${label} API key`,
});

/** Menu order: broadly used providers first, gateways and local servers last. */
export const judgeProviders: readonly JudgeProvider[] = [
  {
    id: "openai",
    label: "OpenAI",
    hint: "DeepEval's default judge; needs nothing but the key",
    secrets: [apiKey("OPENAI_API_KEY", "OpenAI")],
    settings: [],
    flags: {},
    precedence: 1,
  },
  {
    id: "anthropic",
    label: "Anthropic",
    hint: "Claude as the judge, on DeepEval's default model",
    secrets: [apiKey("ANTHROPIC_API_KEY", "Anthropic")],
    settings: [],
    flags: { USE_ANTHROPIC_MODEL: "true" },
    precedence: 12,
  },
  {
    id: "gemini",
    label: "Google Gemini",
    hint: "Gemini as the judge, on DeepEval's default model",
    secrets: [
      {
        ...apiKey("GOOGLE_API_KEY", "Google"),
        optionalWhen: "GOOGLE_GENAI_USE_VERTEXAI",
      },
    ],
    settings: [],
    flags: { USE_GEMINI_MODEL: "true" },
    precedence: 2,
  },
  {
    id: "azure-openai",
    label: "Azure OpenAI",
    hint: "Your Azure deployment; needs endpoint and deployment details",
    secrets: [apiKey("AZURE_OPENAI_API_KEY", "Azure OpenAI")],
    settings: [
      {
        envVar: "AZURE_OPENAI_ENDPOINT",
        label: "Azure OpenAI endpoint",
        placeholder: "https://my-resource.openai.azure.com",
      },
      { envVar: "AZURE_DEPLOYMENT_NAME", label: "Azure deployment name" },
      { envVar: "AZURE_MODEL_NAME", label: "Model behind that deployment" },
      {
        envVar: "OPENAI_API_VERSION",
        label: "Azure API version",
        placeholder: "2024-10-21",
      },
    ],
    flags: { USE_AZURE_OPENAI: "true" },
    precedence: 7,
  },
  {
    id: "bedrock",
    label: "AWS Bedrock",
    hint: "Bedrock models; can use your existing AWS credential chain",
    secrets: [
      {
        envVar: "AWS_ACCESS_KEY_ID",
        label: "AWS access key ID",
        optional: true,
      },
      {
        envVar: "AWS_SECRET_ACCESS_KEY",
        label: "AWS secret access key",
        optional: true,
      },
    ],
    settings: [
      {
        envVar: "AWS_BEDROCK_MODEL_NAME",
        label: "Bedrock model ID",
        placeholder: "anthropic.claude-3-5-sonnet-20241022-v2:0",
      },
      {
        envVar: "AWS_BEDROCK_REGION",
        label: "Bedrock region",
        placeholder: "us-east-1",
      },
    ],
    flags: { USE_AWS_BEDROCK_MODEL: "true" },
    precedence: 13,
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    hint: "Any OpenRouter model, on DeepEval's default model",
    secrets: [apiKey("OPENROUTER_API_KEY", "OpenRouter")],
    settings: [],
    flags: { USE_OPENROUTER_MODEL: "true" },
    precedence: 11,
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    hint: "DeepSeek's API; needs an explicit model name",
    secrets: [apiKey("DEEPSEEK_API_KEY", "DeepSeek")],
    settings: [
      {
        envVar: "DEEPSEEK_MODEL_NAME",
        label: "DeepSeek model name",
        placeholder: "deepseek-chat",
      },
    ],
    flags: { USE_DEEPSEEK_MODEL: "true" },
    precedence: 10,
  },
  {
    id: "grok",
    label: "xAI Grok",
    hint: "Grok as the judge; needs an explicit model name",
    secrets: [apiKey("GROK_API_KEY", "Grok")],
    settings: [
      {
        envVar: "GROK_MODEL_NAME",
        label: "Grok model name",
        placeholder: "grok-4",
      },
    ],
    flags: { USE_GROK_MODEL: "true" },
    precedence: 9,
  },
  {
    id: "moonshot",
    label: "Moonshot (Kimi)",
    hint: "Moonshot's API; needs an explicit model name",
    secrets: [apiKey("MOONSHOT_API_KEY", "Moonshot")],
    settings: [
      {
        envVar: "MOONSHOT_MODEL_NAME",
        label: "Moonshot model name",
        placeholder: "kimi-k2",
      },
    ],
    flags: { USE_MOONSHOT_MODEL: "true" },
    precedence: 8,
  },
  {
    id: "litellm",
    label: "LiteLLM",
    hint: "Route through LiteLLM; reuses a provider key when it has none",
    secrets: [
      {
        ...apiKey("LITELLM_API_KEY", "LiteLLM"),
        optional: true,
      },
    ],
    settings: [
      {
        envVar: "LITELLM_MODEL_NAME",
        label: "LiteLLM model name",
        placeholder: "anthropic/claude-sonnet-4-5",
      },
    ],
    flags: { USE_LITELLM: "true" },
    precedence: 3,
  },
  {
    id: "portkey",
    label: "Portkey",
    hint: "Portkey gateway; needs model, base URL, and upstream provider",
    secrets: [apiKey("PORTKEY_API_KEY", "Portkey")],
    settings: [
      { envVar: "PORTKEY_MODEL_NAME", label: "Portkey model name" },
      {
        envVar: "PORTKEY_BASE_URL",
        label: "Portkey base URL",
        placeholder: "https://api.portkey.ai/v1",
      },
      {
        envVar: "PORTKEY_PROVIDER_NAME",
        label: "Upstream provider name",
        placeholder: "openai",
      },
    ],
    flags: { USE_PORTKEY_MODEL: "true" },
    precedence: 4,
  },
  {
    id: "ollama",
    label: "Ollama (local)",
    hint: "No API key and no provider spend; needs Ollama running locally",
    secrets: [],
    settings: [
      {
        envVar: "OLLAMA_MODEL_NAME",
        label: "Ollama model name",
        placeholder: "llama3.1",
      },
    ],
    flags: { USE_LOCAL_MODEL: "true", LOCAL_MODEL_API_KEY: "ollama" },
    precedence: 5,
  },
  {
    id: "local",
    label: "Local OpenAI-compatible server",
    hint: "Any OpenAI-compatible endpoint, such as vLLM or LM Studio",
    secrets: [
      {
        envVar: "LOCAL_MODEL_API_KEY",
        label: "API key your server expects (any value if it needs none)",
        // DeepEval reads this exact value as its Ollama marker.
        reservedValues: ["ollama"],
      },
    ],
    settings: [
      { envVar: "LOCAL_MODEL_NAME", label: "Model name your server serves" },
      {
        envVar: "LOCAL_MODEL_BASE_URL",
        label: "Server base URL",
        placeholder: "http://localhost:8000/v1",
      },
    ],
    flags: { USE_LOCAL_MODEL: "true" },
    precedence: 6,
  },
];

export type EnvValues = Record<string, string | undefined>;

export interface JudgeStatus {
  provider: JudgeProvider;
  /** Environment variables the provider still needs. */
  missing: string[];
}

const isTruthy = (value: string | undefined): boolean =>
  ["true", "1", "yes"].includes((value ?? "").trim().toLowerCase());

const isFilled = (value: string | undefined): boolean =>
  (value ?? "").trim().length > 0;

const isRequired = (secret: JudgeSecret, env: EnvValues): boolean =>
  !secret.optional &&
  !(secret.optionalWhen && isTruthy(env[secret.optionalWhen]));

/** Every variable a provider reads, secret or not, for display in prompts. */
export const judgeVariables = (provider: JudgeProvider): string[] => [
  ...provider.secrets.map((secret) => secret.envVar),
  ...provider.settings.map((setting) => setting.envVar),
];

/** The provider's requirements this environment cannot satisfy yet. */
export const judgeMissing = (
  provider: JudgeProvider,
  env: EnvValues,
): string[] => [
  ...provider.secrets
    .filter(
      (secret) => isRequired(secret, env) && !isFilled(env[secret.envVar]),
    )
    .map((secret) => secret.envVar),
  ...provider.settings
    .filter((setting) => !isFilled(env[setting.envVar]))
    .map((setting) => setting.envVar),
];

/**
 * Providers holding a key in this environment. DeepEval ignores a key whose
 * `USE_*` flag is unset, so these are offered first: setting the flag costs
 * the user nothing, while pasting a key they already have is busywork.
 */
export const providersWithKey = (env: EnvValues): JudgeProvider[] =>
  judgeProviders.filter((provider) =>
    provider.secrets.some((secret) => {
      const value = (env[secret.envVar] ?? "").trim();
      return Boolean(value) && !secret.reservedValues?.includes(value);
    }),
  );

/** Which judge DeepEval would pick, and what it is still missing. */
export const detectJudge = (env: EnvValues): JudgeStatus | undefined => {
  const flagged = judgeProviders
    .filter((provider) => Object.keys(provider.flags).length > 0)
    .sort((left, right) => left.precedence - right.precedence)
    .find((provider) =>
      Object.entries(provider.flags).every(([envVar, value]) =>
        value === "true"
          ? isTruthy(env[envVar])
          : (env[envVar] ?? "").trim() === value,
      ),
    );
  const provider =
    flagged ??
    (isFilled(env.OPENAI_API_KEY)
      ? judgeProviders.find((candidate) => candidate.id === "openai")
      : undefined);
  if (!provider) return undefined;
  return { provider, missing: judgeMissing(provider, env) };
};

/** The values to write for a provider, given the answers collected. */
export const judgeEnvValues = (
  provider: JudgeProvider,
  answers: {
    secrets?: Record<string, string>;
    settings?: Record<string, string>;
  },
): Record<string, string> => {
  const values: Record<string, string> = { ...provider.flags };
  for (const { envVar } of [...provider.secrets, ...provider.settings]) {
    const value = (
      answers.secrets?.[envVar] ?? answers.settings?.[envVar]
    )?.trim();
    if (value) values[envVar] = value;
  }
  return values;
};
