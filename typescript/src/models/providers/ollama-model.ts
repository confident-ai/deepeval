import type { ZodType } from "zod";
import { DeepEvalBaseLLM, type GenerationResult } from "../base-model";
import {
  computeCost,
  extractJson,
  importOptional,
  toJsonSchema,
} from "../utils";

const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";

export interface OllamaModelOptions {
  model?: string;
  baseURL?: string;
  temperature?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class OllamaModel extends DeepEvalBaseLLM {
  private readonly baseURL: string;
  private readonly temperature?: number;
  private readonly costPerInputToken?: number;
  private readonly costPerOutputToken?: number;
  private client?: any;

  constructor(options: OllamaModelOptions = {}) {
    super(options.model ?? process.env.OLLAMA_MODEL_NAME);
    this.baseURL =
      options.baseURL ??
      process.env.LOCAL_MODEL_BASE_URL ??
      DEFAULT_OLLAMA_BASE_URL;
    // Only sent when explicitly set — some models (e.g. reasoning models) reject `temperature`.
    this.temperature = options.temperature;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
  }

  private async getClient(): Promise<any> {
    if (!this.client) {
      const { Ollama } = await importOptional("ollama", "Ollama");
      this.client = new Ollama({ host: this.baseURL });
    }
    return this.client;
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    if (!this.modelName) {
      throw new Error(
        "Ollama requires a model name. Pass `model` or set OLLAMA_MODEL_NAME.",
      );
    }
    const client = await this.getClient();

    const request: Record<string, unknown> = {
      model: this.modelName,
      messages: [{ role: "user", content: prompt }],
      ...(this.temperature !== undefined && {
        options: { temperature: this.temperature },
      }),
    };
    if (schema) {
      request.format = toJsonSchema(schema);
    }

    const response = await client.chat(request);
    const text: string = response.message?.content ?? "";
    const cost = computeCost(
      response.prompt_eval_count,
      response.eval_count,
      this.costPerInputToken,
      this.costPerOutputToken,
    );

    if (schema) {
      return { output: schema.parse(extractJson(text)), cost };
    }
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? "ollama";
  }
}
