// Every environment variable the TS SDK reads, mirroring Python's
// `Settings(BaseSettings)` in deepeval/config/settings.py.

import { z } from "zod";
import {
  BOOL_TOKENS_MESSAGE,
  normalizeFileSystemMode,
  parseBool,
} from "@/config/utils";
import { isValidLogLevel } from "@/logger";
import { Environment } from "@/tracing/utils";

export interface SettingFieldMeta {
  /** Never written to the `.deepeval` JSON keystore; masked when printed. */
  secret?: boolean;
  description?: string;
}

const optionalString = () => z.string().optional();
// Not `z.coerce.boolean()` (JS truthiness parses "false" as true) nor
// `z.stringbool()` (its trim and quote handling are not configurable).
// Failing here lets `parseSettings` warn and drop the key.
const optionalBool = () =>
  z
    .string()
    .transform((value, ctx) => {
      const parsed = parseBool(value);
      if (parsed === undefined) {
        ctx.addIssue({ code: "custom", message: BOOL_TOKENS_MESSAGE });
        return z.NEVER;
      }
      return parsed;
    })
    .optional();
const optionalNumber = () => z.coerce.number().optional();

const logLevel = () =>
  z
    .string()
    .refine((value) => isValidLogLevel(value), {
      message:
        "Expected one of DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET, or a number.",
    })
    .optional();

const boolWithDefault = (fallback: boolean) => optionalBool().default(fallback);

const secretString = (description: string) =>
  optionalString().describe(description).meta({ secret: true });

export const settingsSchema = z.object({
  // Confident AI
  CONFIDENT_API_KEY: secretString(
    "API key used to authenticate with Confident AI.",
  ),
  CONFIDENT_REGION: z
    .string()
    .transform((value) => value.trim().toUpperCase())
    .refine((value) => ["US", "EU", "AU"].includes(value), {
      message: "Expected one of US, EU, AU.",
    })
    .optional()
    .describe("Confident AI data region (US, EU or AU)."),
  CONFIDENT_BASE_URL: optionalString().describe(
    "Override the Confident AI API URL. Takes precedence over CONFIDENT_REGION.",
  ),
  CONFIDENT_DISABLE_SSL: optionalBool().describe(
    "Skip TLS certificate verification for Confident AI requests. Intended for self-signed certificates; unsafe otherwise.",
  ),
  CONFIDENT_OPEN_BROWSER: boolWithDefault(true).describe(
    "Open a browser automatically for Confident AI links and flows.",
  ),

  // General
  APP_ENV: z
    .string()
    .default("dev")
    .describe(
      "Application environment name used for dotenv selection (loads .env.<APP_ENV> if present).",
    ),
  DEEPEVAL_DISABLE_DOTENV: optionalBool().describe(
    "Disable dotenv autoloading (.env → .env.<APP_ENV> → .env.local). Tip: set to 1 in CI to prevent loading env files on import.",
  ),
  ENV_DIR_PATH: optionalString().describe(
    "Directory containing .env files (default: current working directory).",
  ),

  // CLI
  DEEPEVAL_DEFAULT_SAVE: optionalString().describe(
    "Default persistence target for settings changes (e.g. 'dotenv' or 'dotenv:.env.local').",
  ),
  DEEPEVAL_IDENTIFIER: optionalString().describe(
    "Identifier to help identify your test run on Confident AI.",
  ),
  IGNORE_DEEPEVAL_ERRORS: optionalBool().describe(
    "Continue a run when a metric errors, instead of failing the test case.",
  ),
  SKIP_DEEPEVAL_MISSING_PARAMS: optionalBool().describe(
    "Skip a metric when the test case is missing a parameter it requires.",
  ),
  ENABLE_DEEPEVAL_CACHE: optionalBool().describe(
    "Reuse cached metric results for unchanged test cases and configurations.",
  ),
  DEEPEVAL_TELEMETRY_OPT_OUT: optionalBool().describe(
    "Disable anonymous telemetry.",
  ),
  DEEPEVAL_TELEMETRY_ENABLED: optionalBool().describe(
    "Deprecated inverse of DEEPEVAL_TELEMETRY_OPT_OUT. Any OFF signal wins if both are set.",
  ),

  // Storage & output
  DEEPEVAL_RESULTS_FOLDER: optionalString().describe(
    "If set, export a timestamped JSON of the latest test run into this folder.",
  ),
  DEEPEVAL_CACHE_FOLDER: optionalString().describe(
    "Directory DeepEval uses for its cache and key files (default: .deepeval).",
  ),
  DEEPEVAL_HOME: optionalString().describe(
    "Directory holding the per-machine anonymous telemetry id (default: ~/.deepeval).",
  ),
  DEEPEVAL_FILE_SYSTEM: z
    .string()
    .transform((value, ctx) => {
      const mode = normalizeFileSystemMode(value);
      if (mode === undefined) {
        ctx.addIssue({
          code: "custom",
          message: "Expected READ_ONLY (aliases: READ-ONLY, READONLY, RO).",
        });
        return z.NEVER;
      }
      return mode;
    })
    .optional()
    .describe(
      "Set to READ_ONLY to stop DeepEval writing its keystore, dotenv, cache, and test-run files.",
    ),

  // Debug & tracing
  LOG_LEVEL: logLevel().describe(
    "Global log level (DEBUG|INFO|WARNING|ERROR|CRITICAL|NOTSET).",
  ),
  DEEPEVAL_VERBOSE_MODE: optionalBool().describe(
    "Turn on verbose logs for every metric.",
  ),
  DEEPEVAL_LOG_STACK_TRACES: optionalBool().describe(
    "Include stack traces in logged errors.",
  ),
  DEEPEVAL_RETRY_BEFORE_LOG_LEVEL: logLevel().describe(
    "Log level used before a retry attempt (defaults to LOG_LEVEL, else INFO).",
  ),
  DEEPEVAL_RETRY_AFTER_LOG_LEVEL: logLevel().describe(
    "Log level used when retries are exhausted (defaults to ERROR).",
  ),
  DEEPEVAL_GRPC_LOGGING: optionalBool().describe(
    "Turn on gRPC logging for the OTLP trace exporter.",
  ),
  GRPC_VERBOSITY: optionalString().describe(
    "gRPC verbosity, honoured by grpc-js (DEBUG|INFO|ERROR|NONE).",
  ),
  GRPC_TRACE: optionalString().describe(
    "gRPC tracers, honoured by grpc-js (comma-separated names, '*' for all).",
  ),
  CONFIDENT_TRACE_ENVIRONMENT: z
    .enum(Environment)
    .default(Environment.DEVELOPMENT)
    .describe(
      'Environment traces are attributed to ("development", "staging", "production", "testing").',
    ),
  CONFIDENT_TRACE_VERBOSE: boolWithDefault(true).describe(
    "Print tracing diagnostics.",
  ),
  CONFIDENT_TRACE_FLUSH: optionalBool().describe(
    "Flush traces synchronously at the end of a run.",
  ),
  CONFIDENT_TRACE_INTERNAL: optionalBool().describe(
    "Trace DeepEval's own metric and model methods inside @observe spans.",
  ),
  CONFIDENT_TRACE_SAMPLE_RATE: z.coerce
    .number()
    .min(0)
    .max(1)
    .default(1.0)
    .describe("Fraction of traces to sample, between 0 and 1."),
  CONFIDENT_OTEL_URL: z
    .string()
    .default("https://otel.confident-ai.com")
    .describe("OTLP endpoint traces are exported to."),

  // Model providers — active provider toggles
  USE_OPENAI_MODEL: optionalBool().describe("Use OpenAI as the LLM provider."),
  USE_AZURE_OPENAI: optionalBool().describe(
    "Use Azure OpenAI as the LLM provider.",
  ),
  USE_ANTHROPIC_MODEL: optionalBool().describe(
    "Use Anthropic as the LLM provider.",
  ),
  USE_AWS_BEDROCK_MODEL: optionalBool().describe(
    "Use Amazon Bedrock as the LLM provider.",
  ),
  USE_LOCAL_MODEL: optionalBool().describe(
    "Use a local OpenAI-compatible model (including Ollama) as the LLM provider.",
  ),
  USE_GROK_MODEL: optionalBool().describe("Use Grok as the LLM provider."),
  USE_MOONSHOT_MODEL: optionalBool().describe(
    "Use Moonshot (Kimi) as the LLM provider.",
  ),
  USE_DEEPSEEK_MODEL: optionalBool().describe(
    "Use DeepSeek as the LLM provider.",
  ),
  USE_GEMINI_MODEL: optionalBool().describe("Use Gemini as the LLM provider."),
  USE_PORTKEY_MODEL: optionalBool().describe(
    "Use Portkey as the LLM provider.",
  ),
  USE_OPENROUTER_MODEL: optionalBool().describe(
    "Use OpenRouter as the LLM provider.",
  ),

  // Model providers — general
  TEMPERATURE: optionalNumber().describe(
    "Sampling temperature used by evaluation models.",
  ),

  // OpenAI
  OPENAI_API_KEY: secretString("OpenAI API key."),
  OPENAI_MODEL_NAME: optionalString().describe("OpenAI model name."),
  OPENAI_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the OpenAI model.",
  ),
  OPENAI_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the OpenAI model.",
  ),

  // Azure OpenAI
  AZURE_OPENAI_API_KEY: secretString("Azure OpenAI API key."),
  AZURE_OPENAI_ENDPOINT: optionalString().describe(
    "Azure OpenAI endpoint / base URL.",
  ),
  AZURE_OPENAI_AD_TOKEN: secretString(
    "Microsoft Entra ID token, used instead of AZURE_OPENAI_API_KEY.",
  ),
  OPENAI_API_VERSION: optionalString().describe("Azure OpenAI API version."),
  AZURE_DEPLOYMENT_NAME: optionalString().describe(
    "Azure OpenAI deployment name.",
  ),
  AZURE_MODEL_NAME: optionalString().describe("Azure OpenAI model name."),
  AZURE_MODEL_VERSION: optionalString().describe("Azure OpenAI model version."),

  // Anthropic
  ANTHROPIC_API_KEY: secretString("Anthropic API key."),
  ANTHROPIC_MODEL_NAME: optionalString().describe("Anthropic model name."),
  ANTHROPIC_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the Anthropic model.",
  ),
  ANTHROPIC_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the Anthropic model.",
  ),

  // Amazon Bedrock
  AWS_ACCESS_KEY_ID: secretString("AWS access key ID."),
  AWS_SECRET_ACCESS_KEY: secretString("AWS secret access key."),
  AWS_SESSION_TOKEN: secretString("AWS session token."),
  AWS_BEDROCK_MODEL_NAME: optionalString().describe("Bedrock model name."),
  AWS_BEDROCK_REGION: optionalString().describe(
    "AWS region used for Bedrock (e.g. us-east-1).",
  ),
  AWS_BEDROCK_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the Bedrock model.",
  ),
  AWS_BEDROCK_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the Bedrock model.",
  ),

  // Ollama / local models
  OLLAMA_MODEL_NAME: optionalString().describe("Ollama model name."),
  LOCAL_MODEL_NAME: optionalString().describe("Local model name."),
  LOCAL_MODEL_BASE_URL: optionalString().describe(
    "Base URL of the local OpenAI-compatible server.",
  ),
  LOCAL_MODEL_API_KEY: secretString("API key for the local model server."),
  LOCAL_MODEL_FORMAT: optionalString().describe(
    "Response format requested from the local model (e.g. json).",
  ),

  // Grok
  GROK_API_KEY: secretString("Grok API key."),
  GROK_MODEL_NAME: optionalString().describe("Grok model name."),
  GROK_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the Grok model.",
  ),
  GROK_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the Grok model.",
  ),

  // Moonshot (Kimi)
  MOONSHOT_API_KEY: secretString("Moonshot API key."),
  MOONSHOT_MODEL_NAME: optionalString().describe("Moonshot model name."),
  MOONSHOT_BASE_URL: optionalString().describe("Moonshot base URL."),
  MOONSHOT_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the Moonshot model.",
  ),
  MOONSHOT_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the Moonshot model.",
  ),

  // DeepSeek
  DEEPSEEK_API_KEY: secretString("DeepSeek API key."),
  DEEPSEEK_MODEL_NAME: optionalString().describe("DeepSeek model name."),
  DEEPSEEK_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the DeepSeek model.",
  ),
  DEEPSEEK_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the DeepSeek model.",
  ),

  // Gemini
  GOOGLE_API_KEY: secretString("Google AI Studio API key."),
  GEMINI_API_KEY: secretString("Gemini API key (alias of GOOGLE_API_KEY)."),
  GEMINI_MODEL_NAME: optionalString().describe("Gemini model name."),
  GEMINI_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the Gemini model.",
  ),
  GEMINI_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the Gemini model.",
  ),
  GOOGLE_GENAI_USE_VERTEXAI: optionalBool().describe(
    "Use Vertex AI instead of the Gemini Developer API.",
  ),
  VERTEX_AI_MODEL_NAME: optionalString().describe(
    "Vertex AI model name, preferred over GEMINI_MODEL_NAME when GOOGLE_GENAI_USE_VERTEXAI is on.",
  ),
  GOOGLE_CLOUD_PROJECT: optionalString().describe(
    "Google Cloud project ID used for Vertex AI.",
  ),
  GOOGLE_CLOUD_LOCATION: optionalString().describe(
    "Google Cloud location used for Vertex AI.",
  ),
  GOOGLE_SERVICE_ACCOUNT_KEY: secretString(
    "Google service account JSON, serialized to a single line.",
  ),

  // Portkey
  PORTKEY_API_KEY: secretString("Portkey API key."),
  PORTKEY_MODEL_NAME: optionalString().describe("Portkey model name."),
  PORTKEY_BASE_URL: optionalString().describe("Portkey base URL."),
  PORTKEY_PROVIDER_NAME: optionalString().describe(
    "Provider Portkey should route to.",
  ),

  // OpenRouter
  OPENROUTER_API_KEY: secretString("OpenRouter API key."),
  OPENROUTER_MODEL_NAME: optionalString().describe(
    "OpenRouter model name (e.g. openai/gpt-4.1).",
  ),
  OPENROUTER_BASE_URL: optionalString().describe("OpenRouter base URL."),
  OPENROUTER_COST_PER_INPUT_TOKEN: optionalNumber().describe(
    "USD per input token for the OpenRouter model.",
  ),
  OPENROUTER_COST_PER_OUTPUT_TOKEN: optionalNumber().describe(
    "USD per output token for the OpenRouter model.",
  ),
});

export type Settings = z.infer<typeof settingsSchema>;
export type SettingName = keyof Settings & string;

const shape = settingsSchema.shape as Record<string, z.ZodType>;

export const SETTING_NAMES: SettingName[] = Object.keys(
  shape,
).sort() as SettingName[];

export function getFieldSchema(name: SettingName): z.ZodType {
  return shape[name];
}

export function getFieldMeta(name: SettingName): SettingFieldMeta {
  const field = shape[name];
  const meta = (field?.meta() ?? {}) as SettingFieldMeta;
  return {
    secret: meta.secret === true,
    description: field?.description ?? meta.description ?? "",
  };
}

export function isSecretSetting(name: string): boolean {
  return getFieldMeta(name as SettingName).secret === true;
}

export const SECRET_SETTING_NAMES: SettingName[] = SETTING_NAMES.filter(
  (name) => isSecretSetting(name),
);

/** "log-level" / "log_level" / "LOG_LEVEL" all normalize to the same key. */
export function normalizeSettingKey(raw: string): string {
  return raw.trim().toLowerCase().replace(/-/g, "_");
}

export function resolveSettingNames(query: string): SettingName[] {
  const needle = normalizeSettingKey(query);
  const exact = SETTING_NAMES.filter(
    (name) => normalizeSettingKey(name) === needle,
  );
  if (exact.length > 0) return exact;
  return SETTING_NAMES.filter((name) =>
    normalizeSettingKey(name).includes(needle),
  );
}
