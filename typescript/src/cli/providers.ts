// One declarative entry per LLM provider, driving both the `set-*` / `unset-*`
// commands and `deepeval diagnose`. Mirrors the provider groups in
// deepeval/key_handler.py so the two SDKs write the same env vars.
//
// LiteLLM is intentionally absent: it is a Python library with no TS SDK. Use
// `AISDKModel` to route through any AI SDK provider instead.

import type { SettingName } from "@/config/schema";

export interface ProviderSecretPrompt {
  setting: SettingName;
  label: string;
}

export interface ProviderExtraOption {
  /** Commander flag spec, e.g. "-u, --base-url <url>". */
  flags: string;
  description: string;
  setting: SettingName;
  /** Commander's camelCase name for the option. */
  key: string;
  defaultValue?: string;
  /** Read the value from a file and normalize it (service account JSON). */
  fromJsonFile?: boolean;
  numeric?: boolean;
}

export interface ProviderSpec {
  /** Command suffix: `set-<id>` / `unset-<id>`. */
  id: string;
  label: string;
  useFlag: SettingName;
  modelSetting: SettingName;
  modelHelp?: string;
  /** Prompted for with `-k/--prompt-api-key`. */
  apiKeyPrompt?: ProviderSecretPrompt;
  /** Prompted for with `-a/--prompt-credentials` (Bedrock). */
  credentialPrompts?: ProviderSecretPrompt[];
  /** Enables `-i/--cost-per-input-token` and `-o/--cost-per-output-token`. */
  costSettings?: { input: SettingName; output: SettingName };
  extraOptions?: ProviderExtraOption[];
  /** Set unconditionally by `set-*` (Ollama's placeholder API key). */
  fixedValues?: Partial<Record<SettingName, string>>;
  /** Extra keys `unset-*` clears beyond the ones above. */
  extraUnsetSettings?: SettingName[];
  /** Secrets `unset-* --clear-secrets` removes. */
  secretSettings?: SettingName[];
  /** Appended to the success message in place of the default sentence. */
  successNoun?: string;
}

const COST_HELP_INPUT =
  "USD per input token override used for cost tracking. Preconfigured for " +
  "known models; REQUIRED if you use a custom/unknown model.";
const COST_HELP_OUTPUT =
  "USD per output token override used for cost tracking. Preconfigured for " +
  "known models; REQUIRED if you use a custom/unknown model.";

export const COST_OPTION_HELP = {
  input: COST_HELP_INPUT,
  output: COST_HELP_OUTPUT,
};

export const PROVIDERS: ProviderSpec[] = [
  {
    id: "openai",
    label: "OpenAI",
    useFlag: "USE_OPENAI_MODEL",
    modelSetting: "OPENAI_MODEL_NAME",
    modelHelp: "Model identifier to use for this provider (e.g. `gpt-4.1`).",
    apiKeyPrompt: { setting: "OPENAI_API_KEY", label: "OpenAI API key" },
    costSettings: {
      input: "OPENAI_COST_PER_INPUT_TOKEN",
      output: "OPENAI_COST_PER_OUTPUT_TOKEN",
    },
    secretSettings: ["OPENAI_API_KEY"],
  },
  {
    id: "azure-openai",
    label: "Azure OpenAI",
    useFlag: "USE_AZURE_OPENAI",
    modelSetting: "AZURE_MODEL_NAME",
    apiKeyPrompt: {
      setting: "AZURE_OPENAI_API_KEY",
      label: "Azure OpenAI API key",
    },
    extraOptions: [
      {
        flags: "-u, --base-url <url>",
        description:
          "Override the API endpoint/base URL used by this provider.",
        setting: "AZURE_OPENAI_ENDPOINT",
        key: "baseUrl",
      },
      {
        flags: "-v, --api-version <version>",
        description: "Azure OpenAI API version.",
        setting: "OPENAI_API_VERSION",
        key: "apiVersion",
      },
      {
        flags: "-V, --model-version <version>",
        description: "Azure model version.",
        setting: "AZURE_MODEL_VERSION",
        key: "modelVersion",
      },
      {
        flags: "-d, --deployment-name <name>",
        description: "Azure deployment name.",
        setting: "AZURE_DEPLOYMENT_NAME",
        key: "deploymentName",
      },
    ],
    secretSettings: ["AZURE_OPENAI_API_KEY"],
  },
  {
    id: "anthropic",
    label: "Anthropic",
    useFlag: "USE_ANTHROPIC_MODEL",
    modelSetting: "ANTHROPIC_MODEL_NAME",
    apiKeyPrompt: { setting: "ANTHROPIC_API_KEY", label: "Anthropic API key" },
    costSettings: {
      input: "ANTHROPIC_COST_PER_INPUT_TOKEN",
      output: "ANTHROPIC_COST_PER_OUTPUT_TOKEN",
    },
    secretSettings: ["ANTHROPIC_API_KEY"],
  },
  {
    id: "bedrock",
    label: "AWS Bedrock",
    useFlag: "USE_AWS_BEDROCK_MODEL",
    modelSetting: "AWS_BEDROCK_MODEL_NAME",
    credentialPrompts: [
      { setting: "AWS_ACCESS_KEY_ID", label: "AWS access key ID" },
      { setting: "AWS_SECRET_ACCESS_KEY", label: "AWS secret access key" },
    ],
    costSettings: {
      input: "AWS_BEDROCK_COST_PER_INPUT_TOKEN",
      output: "AWS_BEDROCK_COST_PER_OUTPUT_TOKEN",
    },
    extraOptions: [
      {
        flags: "-r, --region <region>",
        description: "AWS region to use (e.g. us-east-1).",
        setting: "AWS_BEDROCK_REGION",
        key: "region",
      },
    ],
    secretSettings: [
      "AWS_ACCESS_KEY_ID",
      "AWS_SECRET_ACCESS_KEY",
      "AWS_SESSION_TOKEN",
    ],
  },
  {
    id: "ollama",
    label: "Ollama",
    useFlag: "USE_LOCAL_MODEL",
    modelSetting: "OLLAMA_MODEL_NAME",
    extraOptions: [
      {
        flags: "-u, --base-url <url>",
        description:
          "Override the API endpoint/base URL used by this provider.",
        setting: "LOCAL_MODEL_BASE_URL",
        key: "baseUrl",
        defaultValue: "http://localhost:11434",
      },
    ],
    // Ollama ignores the key, but the OpenAI-compatible client requires one.
    fixedValues: { LOCAL_MODEL_API_KEY: "ollama" },
    secretSettings: ["LOCAL_MODEL_API_KEY"],
    successNoun: "a local Ollama model",
  },
  {
    id: "local-model",
    label: "local model",
    useFlag: "USE_LOCAL_MODEL",
    modelSetting: "LOCAL_MODEL_NAME",
    apiKeyPrompt: {
      setting: "LOCAL_MODEL_API_KEY",
      label: "Local model API key",
    },
    extraOptions: [
      {
        flags: "-u, --base-url <url>",
        description:
          "Override the API endpoint/base URL used by this provider.",
        setting: "LOCAL_MODEL_BASE_URL",
        key: "baseUrl",
      },
      {
        flags: "-f, --format <format>",
        description:
          "Response format requested from the model (default: json).",
        setting: "LOCAL_MODEL_FORMAT",
        key: "format",
      },
    ],
    secretSettings: ["LOCAL_MODEL_API_KEY"],
    successNoun: "a local model",
  },
  {
    id: "grok",
    label: "Grok",
    useFlag: "USE_GROK_MODEL",
    modelSetting: "GROK_MODEL_NAME",
    apiKeyPrompt: { setting: "GROK_API_KEY", label: "Grok API key" },
    costSettings: {
      input: "GROK_COST_PER_INPUT_TOKEN",
      output: "GROK_COST_PER_OUTPUT_TOKEN",
    },
    secretSettings: ["GROK_API_KEY"],
  },
  {
    id: "moonshot",
    label: "Moonshot",
    useFlag: "USE_MOONSHOT_MODEL",
    modelSetting: "MOONSHOT_MODEL_NAME",
    apiKeyPrompt: { setting: "MOONSHOT_API_KEY", label: "Moonshot API key" },
    costSettings: {
      input: "MOONSHOT_COST_PER_INPUT_TOKEN",
      output: "MOONSHOT_COST_PER_OUTPUT_TOKEN",
    },
    extraOptions: [
      {
        flags: "-u, --base-url <url>",
        description:
          "Override the API endpoint/base URL used by this provider.",
        setting: "MOONSHOT_BASE_URL",
        key: "baseUrl",
      },
    ],
    secretSettings: ["MOONSHOT_API_KEY"],
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    useFlag: "USE_DEEPSEEK_MODEL",
    modelSetting: "DEEPSEEK_MODEL_NAME",
    apiKeyPrompt: { setting: "DEEPSEEK_API_KEY", label: "DeepSeek API key" },
    costSettings: {
      input: "DEEPSEEK_COST_PER_INPUT_TOKEN",
      output: "DEEPSEEK_COST_PER_OUTPUT_TOKEN",
    },
    secretSettings: ["DEEPSEEK_API_KEY"],
  },
  {
    id: "gemini",
    label: "Gemini",
    useFlag: "USE_GEMINI_MODEL",
    modelSetting: "GEMINI_MODEL_NAME",
    apiKeyPrompt: { setting: "GOOGLE_API_KEY", label: "Google API key" },
    extraOptions: [
      {
        flags: "-p, --project <project>",
        description: "Google Cloud project ID (Vertex AI).",
        setting: "GOOGLE_CLOUD_PROJECT",
        key: "project",
      },
      {
        flags: "-l, --location <location>",
        description: "Google Cloud location (Vertex AI).",
        setting: "GOOGLE_CLOUD_LOCATION",
        key: "location",
      },
      {
        flags: "-S, --service-account-file <path>",
        description: "Path to a Google service account JSON file.",
        setting: "GOOGLE_SERVICE_ACCOUNT_KEY",
        key: "serviceAccountFile",
        fromJsonFile: true,
      },
    ],
    extraUnsetSettings: ["GOOGLE_GENAI_USE_VERTEXAI"],
    secretSettings: ["GOOGLE_API_KEY", "GOOGLE_SERVICE_ACCOUNT_KEY"],
  },
  {
    id: "portkey",
    label: "Portkey",
    useFlag: "USE_PORTKEY_MODEL",
    modelSetting: "PORTKEY_MODEL_NAME",
    apiKeyPrompt: { setting: "PORTKEY_API_KEY", label: "Portkey API key" },
    extraOptions: [
      {
        flags: "-u, --base-url <url>",
        description:
          "Override the API endpoint/base URL used by this provider.",
        setting: "PORTKEY_BASE_URL",
        key: "baseUrl",
      },
      {
        flags: "-P, --provider <provider>",
        description: "Provider Portkey should route to.",
        setting: "PORTKEY_PROVIDER_NAME",
        key: "provider",
      },
    ],
    secretSettings: ["PORTKEY_API_KEY"],
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    useFlag: "USE_OPENROUTER_MODEL",
    modelSetting: "OPENROUTER_MODEL_NAME",
    modelHelp:
      "Model identifier to use for this provider (e.g. `openai/gpt-4.1`).",
    apiKeyPrompt: {
      setting: "OPENROUTER_API_KEY",
      label: "OpenRouter API key",
    },
    costSettings: {
      input: "OPENROUTER_COST_PER_INPUT_TOKEN",
      output: "OPENROUTER_COST_PER_OUTPUT_TOKEN",
    },
    extraOptions: [
      {
        flags: "-u, --base-url <url>",
        description:
          "Override the API endpoint/base URL used by this provider.",
        setting: "OPENROUTER_BASE_URL",
        key: "baseUrl",
      },
      {
        flags: "-t, --temperature <temperature>",
        description: "Override the global TEMPERATURE setting.",
        setting: "TEMPERATURE",
        key: "temperature",
        numeric: true,
      },
    ],
    secretSettings: ["OPENROUTER_API_KEY"],
  },
];

export const USE_LLM_FLAGS: SettingName[] = [
  ...new Set(PROVIDERS.map((provider) => provider.useFlag)),
];

export function getProvider(id: string): ProviderSpec | undefined {
  return PROVIDERS.find((provider) => provider.id === id);
}

export function ownedSettings(provider: ProviderSpec): SettingName[] {
  const settings: SettingName[] = [provider.modelSetting, provider.useFlag];
  if (provider.costSettings) {
    settings.push(provider.costSettings.input, provider.costSettings.output);
  }
  for (const option of provider.extraOptions ?? []) {
    settings.push(option.setting);
  }
  for (const setting of provider.extraUnsetSettings ?? []) {
    settings.push(setting);
  }
  return [...new Set(settings)];
}
