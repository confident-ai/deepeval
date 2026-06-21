# Models — TypeScript port overview

Status of the TS model layer (`typescript/src/models/`): which providers are
supported, the generation API, and how it differs from Python. Raw material for docs.

## The model contract

Every model extends `DeepEvalBaseLLM` (`base-model.ts`) and implements a **single,
always-async** generation method:

```ts
abstract generate<T = string>(
  prompt: string,
  schema?: ZodType<T>,
): Promise<GenerationResult<T>>;   // { output: T; cost: number | null }
```

### Key API facts (and how they differ from Python)

- **`generate()` is the only call, and it is always `async`.** There is **no
  `aGenerate` / sync split** like Python's `generate()` + `a_generate()`. Pass a prompt
  string; optionally pass a **zod schema** to get a parsed, validated object back as
  `output` (otherwise `output` is the raw string).
- **Every model returns `{ output, cost }`** — so in metric-land all TS models are
  "native" and cost is accrued whenever the model reports it.
- **Cost is opt-in and explicit.** `cost` is `null` unless you pass **both**
  `costPerInputToken` and `costPerOutputToken` to the constructor; then
  `cost = inTok·inRate + outTok·outRate` (`computeCost` in `utils.ts`). There is **no
  built-in price table** (Python ships per-model pricing maps; TS does not).
- **Capability flags** (override `DeepEvalBaseLLM` defaults, which all return `null`):
  `supportsStructuredOutputs()`, `supportsMultimodal()`, `supportsLogProbs()`.
  **No model reports log-prob support** (`supportsLogProbs()` is `null` everywhere) — this
  is why GEval can't do log-prob-weighted scoring in TS.
- **Structured output is "best-effort JSON + parse"**: each provider asks for JSON in its
  own way (OpenAI `response_format: json_schema` with `strict: false`; Gemini
  `responseMimeType`; Ollama `format`), then the response is run through
  `extractJson()` (slices the first `{`…last `}`, strips trailing commas) and
  `schema.parse()`. The AI SDK path uses native `generateObject`.
- **SDKs are optional/lazy.** Provider SDKs are dynamically `import()`-ed on first use;
  a missing package throws a friendly "install X" error (`importOptional`). `openai` is
  the one commonly-present dep (OpenAI-compatible base).
- **Temperature is only sent when explicitly set** (many reasoning models reject it).

## Two families of models

1. **OpenAI-compatible** — thin subclasses of `DeepEvalOpenAICompatibleModel`
   (`openai-compatible-model.ts`), which holds all the logic (client, generation,
   structured output, cost, multimodal). Each subclass only resolves its own defaults:
   model name, base URL, env-var-backed API key, headers. Uses the official `openai` SDK
   pointed at the right endpoint.
2. **Native-SDK providers** — their own `generate()` against a provider SDK (Gemini,
   Anthropic, Bedrock, Ollama) or the Vercel AI SDK (`AISDKModel`).

## Supported providers

### OpenAI-compatible (via the `openai` SDK)

| Class | Default model | API key env | Base URL | Notes |
|---|---|---|---|---|
| `OpenAIModel` | `gpt-4.1` (`OPENAI_MODEL_NAME`) | `OPENAI_API_KEY` | OpenAI default | canonical |
| `AzureOpenAIModel` | deployment name | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_ENDPOINT` (req.) | uses `AzureOpenAI` client; routes by `deployment`; `OPENAI_API_VERSION` |
| `DeepSeekModel` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` | |
| `GrokModel` (xAI) | `grok-3` | `GROK_API_KEY` / `XAI_API_KEY` | `https://api.x.ai/v1` | |
| `KimiModel` (Moonshot) | `moonshot-v1-8k` | `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` | `.ai/v1` for international |
| `LocalModel` (vLLM / LM Studio) | `LOCAL_MODEL_NAME` (req.) | `LOCAL_MODEL_API_KEY` (placeholder ok) | `LOCAL_MODEL_BASE_URL` (req.) | any OpenAI `/v1` server |
| `OpenRouterModel` (gateway) | `openai/gpt-4.1` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | ranking headers via `defaultHeaders` |
| `PortkeyModel` (gateway) | `PORTKEY_MODEL_NAME` | `PORTKEY_API_KEY` | `https://api.portkey.ai/v1` | auth via `x-portkey-*` headers; `provider` option |

All of the above report `supportsStructuredOutputs() = true` and
`supportsMultimodal() = true`.

### Native-SDK providers

| Class | Default model | API key env | SDK package | Multimodal |
|---|---|---|---|---|
| `GeminiModel` | `gemini-2.5-flash` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `@google/genai` | ✅ (fetches+base64s remote images) |
| `AnthropicModel` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `@anthropic-ai/sdk` | ✅ |
| `AmazonBedrockModel` | `AWS_BEDROCK_MODEL_NAME` (req.) | AWS creds / region | `@aws-sdk/client-bedrock-runtime` | ⚠️ see discrepancy |
| `OllamaModel` | `OLLAMA_MODEL_NAME` (req.) | — (local) | `ollama` | ❌ text-only |
| `AISDKModel` | from the AI SDK model | (per AI SDK provider) | `ai` (+ a provider, e.g. `@ai-sdk/openai`) | ✅ |

- `GeminiModel` also supports **Vertex AI** (`useVertexAI` / `GOOGLE_GENAI_USE_VERTEXAI`,
  with `project` / `location`).
- `AnthropicModel` sends `max_tokens` (default `4096`, configurable).
- `AISDKModel` wraps any Vercel AI SDK `LanguageModel` (e.g. `openai("gpt-4o")`); uses
  `generateObject` for schemas, `generateText` otherwise.

## Multimodal support

Image slugs in the prompt are split into provider-specific text+image parts by
`multimodal.ts` (`openAIContent`, `aiSdkContent`, `anthropicContent`, `geminiContents`).
Wired into: **OpenAI-compatible base** (so OpenAI/Azure/Grok/Kimi/Local/OpenRouter/Portkey),
**Anthropic**, **Gemini**, **AI SDK**. Plain-text prompts pass through unchanged.

## Gaps & discrepancies vs Python

- **`AmazonBedrockModel` reports `supportsMultimodal() = true` but its `generate()` only
  sends text** (`content: [{ text: prompt }]`, no slug→image conversion). The flag is
  ahead of the implementation — image slugs would be sent as literal text. Treat Bedrock
  as text-only until the Converse content builder is added.
- **`OllamaModel` is text-only** and (unlike the others) does **not** override
  `supportsMultimodal()`, so it returns the base `null`.
- **No log-prob support anywhere** (`supportsLogProbs()` is `null` for all models) — blocks
  GEval log-prob-weighted scoring.
- **No built-in pricing tables** — Python auto-computes cost from per-model price maps;
  TS returns `cost: null` unless you pass `costPerInputToken` + `costPerOutputToken`.
- **Structured output uses `strict: false`** + a tolerant `extractJson` rather than strict
  JSON-schema enforcement; a weak model can still return unparseable JSON (raises a
  "use a more capable model" error).

## Usage examples

```ts
import {
  OpenAIModel, AnthropicModel, GeminiModel, AISDKModel, LocalModel, AzureOpenAIModel,
} from "deepeval/models";

// Default (gpt-4.1, OPENAI_API_KEY)
const m1 = new OpenAIModel();

// Explicit model + cost tracking
const m2 = new OpenAIModel({
  model: "gpt-4.1-mini",
  costPerInputToken: 0.4 / 1e6,
  costPerOutputToken: 1.6 / 1e6,
});

// Other providers
const m3 = new AnthropicModel({ model: "claude-sonnet-4-6" });
const m4 = new GeminiModel({ model: "gemini-2.5-flash" });
const m5 = new LocalModel({ model: "llama3", baseURL: "http://localhost:8000/v1" });
const m6 = new AzureOpenAIModel({ deployment: "my-gpt4", endpoint: "https://x.openai.azure.com" });

// Vercel AI SDK
import { openai } from "@ai-sdk/openai";
const m7 = new AISDKModel({ model: openai("gpt-4o") });

// Plain text
const { output, cost } = await m1.generate("Summarize: ...");

// Structured (zod) — returns a parsed, typed object
import { z } from "zod";
const schema = z.object({ score: z.number(), reason: z.string() });
const { output: parsed } = await m1.generate("Rate this 0-1 with a reason ...", schema);
parsed.score; // number

// Use directly in a metric
const metric = new AnswerRelevancyMetric({ threshold: 0.7, model: m3 });
```
