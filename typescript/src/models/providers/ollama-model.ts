import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type GenerationKwargs,
  type GenerationResult,
} from "../base-model";
import { extractJson, importOptional, toJsonSchema } from "../utils";
import type { ModelNamespace } from "../registry";

const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";

export interface OllamaModelOptions {
  model?: string;
  baseURL?: string;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  /** Extra params merged into Ollama's `chat(..., options: {...})` bag. */
  generationKwargs?: GenerationKwargs;
}

export class OllamaModel extends DeepEvalBaseLLM {
  private readonly baseURL: string;
  private readonly generationKwargs: GenerationKwargs;
  private client?: any;
  protected registryNamespace: ModelNamespace = "ollama";

  constructor(options: OllamaModelOptions = {}) {
    super(options.model ?? process.env.OLLAMA_MODEL_NAME);
    this.baseURL =
      options.baseURL ??
      process.env.LOCAL_MODEL_BASE_URL ??
      DEFAULT_OLLAMA_BASE_URL;
    this.temperature = options.temperature;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
    this.generationKwargs = { ...options.generationKwargs };
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

    const temperature = this.resolveTemperature();
    const modelOptions: Record<string, unknown> = {
      ...(temperature !== undefined && { temperature }),
      ...this.generationKwargs,
    };

    const request: Record<string, unknown> = {
      model: this.modelName,
      messages: [{ role: "user", content: prompt }],
      ...(Object.keys(modelOptions).length > 0 && { options: modelOptions }),
    };
    if (schema) {
      request.format = toJsonSchema(schema);
    }

    const response = await client.chat(request);
    const text: string = response.message?.content ?? "";
    const cost = this.resolveCost(
      response.prompt_eval_count,
      response.eval_count,
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
