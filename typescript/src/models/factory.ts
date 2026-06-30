import { DeepEvalBaseLLM } from "./base-model";
import { OpenAIModel } from "./providers/openai-model";
import { AzureOpenAIModel } from "./providers/azure-model";
import { AnthropicModel } from "./providers/anthropic-model";
import { GeminiModel } from "./providers/gemini-model";
import { DeepSeekModel } from "./providers/deepseek-model";
import { GrokModel } from "./providers/grok-model";
import { KimiModel } from "./providers/kimi-model";
import { LocalModel } from "./providers/local-model";
import { OllamaModel } from "./providers/ollama-model";
import { AmazonBedrockModel } from "./providers/bedrock-model";
import { DeepEvalBaseEmbeddingModel } from "./embedding-models/base-embedding-model";
import { OpenAIEmbeddingModel } from "./embedding-models/openai-embedding-model";

export type ModelProvider =
  | "openai"
  | "azure"
  | "anthropic"
  | "gemini"
  | "deepseek"
  | "grok"
  | "kimi"
  | "local"
  | "ollama"
  | "bedrock";

export interface ModelFactoryOptions {
  model: string;
  provider?: ModelProvider;
  apiKey?: string;
  baseURL?: string;
  temperature?: number;
}

const PROVIDER_MODEL_PREFIXES: Record<string, ModelProvider> = {
  "gpt-": "openai",
  "o1-": "openai",
  "o3-": "openai",
  "claude-": "anthropic",
  "gemini-": "gemini",
  "deepseek-": "deepseek",
  "grok-": "grok",
  "kimi-": "kimi",
  "llama-": "ollama",
  "mistral-": "ollama",
  "qwen-": "ollama",
  "command-": "ollama",
};

function detectProvider(model: string): ModelProvider {
  for (const [prefix, provider] of Object.entries(PROVIDER_MODEL_PREFIXES)) {
    if (model.startsWith(prefix)) return provider;
  }
  return "openai";
}

export class ModelFactory {
  static createLLM(options: ModelFactoryOptions): DeepEvalBaseLLM {
    const provider = options.provider ?? detectProvider(options.model);
    switch (provider) {
      case "openai":
        return new OpenAIModel({
          model: options.model,
          apiKey: options.apiKey,
          temperature: options.temperature,
        });
      case "azure":
        return new AzureOpenAIModel({
          model: options.model,
          apiKey: options.apiKey,
          baseURL: options.baseURL,
          temperature: options.temperature,
        });
      case "anthropic":
        return new AnthropicModel({
          model: options.model,
          apiKey: options.apiKey,
          temperature: options.temperature,
        });
      case "gemini":
        return new GeminiModel({
          model: options.model,
          apiKey: options.apiKey,
          temperature: options.temperature,
        });
      case "deepseek":
        return new DeepSeekModel({
          model: options.model,
          apiKey: options.apiKey,
          temperature: options.temperature,
        });
      case "grok":
        return new GrokModel({
          model: options.model,
          apiKey: options.apiKey,
          temperature: options.temperature,
        });
      case "kimi":
        return new KimiModel({
          model: options.model,
          apiKey: options.apiKey,
          temperature: options.temperature,
        });
      case "local":
        return new LocalModel({
          model: options.model,
          apiKey: options.apiKey,
          baseURL: options.baseURL,
          temperature: options.temperature,
        });
      case "ollama":
        return new OllamaModel({
          model: options.model,
          temperature: options.temperature,
        });
      case "bedrock":
        return new AmazonBedrockModel({
          model: options.model,
          temperature: options.temperature,
        });
    }
  }

  static createEmbedding(
    model: string,
    options?: { apiKey?: string; dimensions?: number },
  ): DeepEvalBaseEmbeddingModel {
    return new OpenAIEmbeddingModel({
      model,
      apiKey: options?.apiKey,
      dimensions: options?.dimensions,
    });
  }
}
