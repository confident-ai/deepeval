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

- **`generate()` is the only required call, and it is always `async`.** There is **no
  `aGenerate` / sync split** like Python's `generate()` + `a_generate()`. Pass a prompt
  string; optionally pass a **zod schema** to get a parsed, validated object back as
  `output` (otherwise `output` is the raw string).
- **`generateRaw()` is an optional second method** for per-token log probabilities:

  ```ts
  generateRaw?(prompt: string, options?: { topLogprobs?: number }):
    Promise<{ output: string; cost: number | null; logProbs?: ContentTokenLogProbs[] }>;
  ```

  Only `DeepEvalOpenAICompatibleModel` implements it (so OpenAI, Azure, OpenRouter,
  Portkey), matching the providers Python gives a `generate_raw_response`. Its
  **absence is the capability signal** — the TS equivalent of Python's
  `AttributeError` gate — so every caller keeps a non-log-prob fallback. `logProbs`
  is normalized away from OpenAI's wire format, unlike Python which passes
  `ChatCompletion` straight through to the metric. The request deliberately omits
  `response_format`, as Python's does: structured outputs and `top_logprobs` don't
  compose, so the prompt asks for JSON and the caller recovers it with `extractJson`.
- **Every model returns `{ output, cost }`** — so in metric-land all TS models are
  "native" and cost is accrued whenever the model reports it.
- **Cost is automatic for known models.** Prices come from the generated registry (see
  below), so `cost` is populated without configuration. `costPerInputToken` /
  `costPerOutputToken` still work and take precedence; `cost` is `null` only when neither
  the caller nor the registry knows the rates. The math lives in `resolveCost` on
  `DeepEvalBaseLLM`.
- **Capability flags** are registry-backed: `supportsStructuredOutputs()`,
  `supportsMultimodal()`, `supportsLogProbs()`, `maxLogProbs()`, `supportsTemperature()`.
  Providers fall back to what their transport supports for models the registry omits.
- **Structured output is "best-effort JSON + parse"**: each provider asks for JSON in its
  own way (OpenAI `response_format: json_schema` with `strict: false`; Gemini
  `responseMimeType`; Ollama `format`), then the response is run through
  `extractJson()` (slices the first `{`…last `}`, strips trailing commas) and
  `schema.parse()`. The AI SDK path uses native `generateObject`.
- **SDKs are optional/lazy.** Provider SDKs are dynamically `import()`-ed on first use;
  a missing package throws a friendly "install X" error (`importOptional`). `openai` is
  the one commonly-present dep (OpenAI-compatible base).
- **Temperature defaults to `0`, matching Python**, and is dropped for models the
  registry flags `supportsTemperature: false` (reasoning models reject it). Passing
  `temperature: null` forces omission — useful for a new reasoning model the registry
  does not cover yet, where the `0` default would otherwise be rejected. Resolution is
  `resolveTemperature()` on `DeepEvalBaseLLM`.
- **Unrecognized options are the escape hatch** for anything not exposed as a
  first-class option. Every options type extends `ExtraGenerationParams` (an index
  signature), and every constructor destructures the options it knows and keeps the
  rest — so where Python needs a nested `generation_kwargs` dict, TypeScript takes the
  same keys inline. The rest bag is merged **last**, so a key given there overrides the
  equivalent option (e.g. `temperature`). Each provider forwards it to the natural place
  for its SDK: the request body for OpenAI-compatible providers and Anthropic, `config`
  for Gemini, `inferenceConfig` for Bedrock, the `options` bag for Ollama, and the
  `generateText`/`generateObject` call for the AI SDK.

  A subclass that adds its own options must destructure them out **before** calling
  `super(...)`, or they land in the rest bag and get sent to the provider — that is why
  `AzureOpenAIModel` peels off `endpoint`/`apiVersion`/`deployment`/`adToken` and
  `PortkeyModel` peels off `provider`.

## The model registry (Python is the source of truth)

Per-model pricing and capabilities live in `deepeval/models/llms/constants.py`. That file
is the **single source of truth for both SDKs** and is the only place to edit this data.
`scripts/compile_model_registry.py` projects it into a committed JSON artifact for TS:

```bash
python scripts/compile_model_registry.py   # writes src/models/registry/models.json
```

This mirrors how metric templates are shared, with one difference: templates dual-emit
into both packages, whereas this is a **one-way emit** — Python keeps importing
`constants.py` directly, so nothing is generated back into the Python package.

`models.json` is a committed build artifact — never hand-edit it.
`tests/test_models/test_model_registry.py` rebuilds it and byte-compares, so CI fails if
someone edits `constants.py` without recompiling. The script loads `constants.py` in
isolation (stubbing its one deepeval import), so the check needs only `pip install pytest`.

Lookups go through `getModelData(namespace, modelName)` in `registry/index.ts`. Each
provider declares a `registryNamespace` matching its Python registry (`openai`,
`anthropic`, `gemini`, `grok`, `kimi`, `deepseek`, `ollama`, `bedrock`). `LocalModel`,
`OpenRouterModel` and `PortkeyModel` declare none, because Python has no registry for
them either. Unknown models resolve to `DEFAULT_MODEL_DATA`, matching Python's
`ModelDataRegistry.get()` returning an empty `DeepEvalModelData`.

`AzureOpenAIModel` sets `registryModelName` as well: requests route by deployment, which
can be named anything, but pricing belongs to the underlying `model`.

### Default models are generated too

`DEFAULT_MODELS` in `constants.py` is projected into the same artifact, under `_defaults`,
and read back through `defaultModelName(namespace)`. **Never write a default model name
into a provider** — change `constants.py` and recompile, or the two SDKs will fall back to
different judges, which is precisely how they drifted before. The `_defaults` keys are
typed off the generated JSON, so removing one breaks the build at the provider that reads
it, and a test asserts no provider reintroduces a `DEFAULT_*_MODEL` literal.

Grok, DeepSeek and Kimi are the deliberate exceptions: they keep TypeScript-only defaults
because Python requires their `*_MODEL_NAME` instead. Both sides of that asymmetry are
documented next to `DEFAULT_MODELS`.

### Models that exist only in TypeScript

`AISDKModel` has no Python counterpart, yet most models reached through it *are* models
Python already prices. The AI SDK reports provider ids like `openai.chat`, so
`resolveAiSdkNamespace` (in `providers/ai-sdk-model.ts`) routes `openai("gpt-4o")` into
the `openai` namespace instead of duplicating the entry.

Providers the AI SDK supports and Python does not — Mistral, Cohere, and the like — are
simply unrouted: no pricing, and capabilities fall back to what the transport supports.
There is deliberately **no TypeScript-side data table**. If one becomes necessary, put it
next to the provider that needs it and consult it only for models the generated registry
lacks, so `constants.py` stays authoritative and nothing shadows it. Prefer adding the
model to `constants.py` — then both SDKs get it.

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
| `OpenAIModel` | `gpt-5.4` † (`OPENAI_MODEL_NAME`) | `OPENAI_API_KEY` | OpenAI default | canonical |
| `AzureOpenAIModel` | deployment name | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_ENDPOINT` (req.) | uses `AzureOpenAI` client; routes by `deployment`; `OPENAI_API_VERSION` |
| `DeepSeekModel` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` | |
| `GrokModel` (xAI) | `grok-3` | `GROK_API_KEY` / `XAI_API_KEY` | `https://api.x.ai/v1` | |
| `KimiModel` (Moonshot) | `moonshot-v1-8k` | `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` | `.ai/v1` for international |
| `LocalModel` (vLLM / LM Studio) | `LOCAL_MODEL_NAME` (req.) | `LOCAL_MODEL_API_KEY` (placeholder ok) | `LOCAL_MODEL_BASE_URL` (req.) | any OpenAI `/v1` server |
| `OpenRouterModel` (gateway) | `openai/gpt-5.4` † | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | ranking headers via `defaultHeaders` |
| `PortkeyModel` (gateway) | `PORTKEY_MODEL_NAME` | `PORTKEY_API_KEY` | `https://api.portkey.ai/v1` | auth via `x-portkey-*` headers; `provider` option |

All of the above report `supportsStructuredOutputs() = true` and
`supportsMultimodal() = true`.

† Generated from `DEFAULT_MODELS` in `constants.py` — this table records the current value,
it does not define it.

### Native-SDK providers

| Class | Default model | API key env | SDK package | Multimodal |
|---|---|---|---|---|
| `GeminiModel` | `gemini-3.6-flash` † | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `@google/genai` | ✅ (fetches+base64s remote images) |
| `AnthropicModel` | `claude-opus-5` † | `ANTHROPIC_API_KEY` | `@anthropic-ai/sdk` | ✅ |
| `AmazonBedrockModel` | `AWS_BEDROCK_MODEL_NAME` (req.) | AWS creds / region | `@aws-sdk/client-bedrock-runtime` | ✅ (Converse image blocks, inlined as bytes) |
| `OllamaModel` | `OLLAMA_MODEL_NAME` (req.) | — (local) | `ollama` | ✅ (base64 in per-message `images`) |
| `AISDKModel` | from the AI SDK model | (per AI SDK provider) | `ai` (+ a provider, e.g. `@ai-sdk/openai`) | ✅ |

- `GeminiModel` also supports **Vertex AI** (`useVertexAI` / `GOOGLE_GENAI_USE_VERTEXAI`,
  with `project` / `location`).
- `AnthropicModel` sends `max_tokens` (default `4096`, configurable).
- `AISDKModel` wraps any Vercel AI SDK `LanguageModel` (e.g. `openai("gpt-4o")`); uses
  `generateObject` for schemas, `generateText` otherwise.

## Multimodal support

Image slugs in the prompt are split into provider-specific text+image parts by
`multimodal.ts` (`openAIContent`, `aiSdkContent`, `anthropicContent`, `geminiContents`,
`ollamaMessages`, `bedrockContent`). Wired into **every** provider: the OpenAI-compatible
base (so OpenAI/Azure/Grok/Kimi/Local/OpenRouter/Portkey), Anthropic, Gemini, AI SDK,
Ollama, and Bedrock. Plain-text prompts pass through unchanged.

Providers differ in what they accept, so the builders differ in shape: OpenAI and the AI
SDK take a URL or data URI; Anthropic takes a remote URL directly or base64; Gemini,
Ollama, and Bedrock require the bytes, so remote images are fetched and inlined.

Metrics refuse a multimodal test case outright when the evaluation model isn't
vision-capable — see `checkMultimodalSupport` in `metrics/utils.ts`.

## Gaps & discrepancies vs Python

- **Bedrock image `format` is lowercase.** Python uppercases the mime subtype
  (`JPEG`), which the Converse API rejects; TS sends the documented lowercase value and
  throws for formats Bedrock doesn't accept at all.
- **No `TEMPERATURE` global** — Python's `settings.TEMPERATURE` is written by the CLI's
  `set-*` commands; the TS CLI has no config store, so there is nothing to read. TS uses
  the same `0` default, just without the env-var layer in between.
- **Cost precedence differs.** Python's `require_costs` returns the registry price even
  when the caller passed `cost_per_input_token`, so explicit costs are ignored for models
  it prices. TS lets the explicit value win.
- **Temperature omission is uniform.** For models that reject `temperature`, Python is
  inconsistent — `OpenAIModel` forces `1`, `AzureOpenAIModel` and `AnthropicModel` omit it.
  TS always omits, which every provider accepts. Python's `AnthropicModel` also defaults
  to unset rather than `0.0`; TS uses `0` there like every other provider.
- **Structured output uses `strict: false`** + a tolerant `extractJson` rather than strict
  JSON-schema enforcement; a weak model can still return unparseable JSON (raises a
  "use a more capable model" error).

## Usage examples

```ts
import {
  OpenAIModel, AnthropicModel, GeminiModel, AISDKModel, LocalModel, AzureOpenAIModel,
} from "deepeval/models";

// Default: whatever DEFAULT_MODELS.openai generates, plus OPENAI_API_KEY. Cost is
// reported automatically — every generated default is priced in the registry.
const m1 = new OpenAIModel();

// Override the registry's prices (e.g. negotiated rates)
const m2 = new OpenAIModel({
  model: "gpt-4.1-mini",
  costPerInputToken: 0.4 / 1e6,
  costPerOutputToken: 1.6 / 1e6,
});

// Reasoning models: temperature is dropped automatically for known ones. For a model
// too new for the registry, pass null to suppress the default of 0.
const m2c = new OpenAIModel({ model: "o3-mini" });
const m2d = new OpenAIModel({ model: "brand-new-reasoning-model", temperature: null });

// Provider params with no first-class option are passed inline and forwarded as-is
const m2b = new OpenAIModel({
  model: "gpt-4.1",
  top_p: 0.9,
  seed: 42,
  max_completion_tokens: 512,
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
