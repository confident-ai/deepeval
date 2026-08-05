import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type ExtraGenerationParams,
  type GenerationResult,
} from "@/models/base-model";
import { extractJson, importOptional, toJsonSchema } from "@/models/utils";
import { ollamaMessages } from "@/models/multimodal";
import type { ModelNamespace } from "@/models/registry";

const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";

/** Any other key is merged into Ollama's `chat(..., options: {...})` bag. */
export interface OllamaModelOptions extends ExtraGenerationParams {
  model?: string;
  baseURL?: string;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class OllamaModel extends DeepEvalBaseLLM {
  private readonly baseURL: string;
  private readonly extraParams: ExtraGenerationParams;
  private client?: any;
  protected registryNamespace: ModelNamespace = "ollama";

  constructor(options: OllamaModelOptions = {}) {
    const {
      model,
      baseURL,
      temperature,
      costPerInputToken,
      costPerOutputToken,
      ...extraParams
    } = options;

    super(model ?? process.env.OLLAMA_MODEL_NAME);
    this.baseURL =
      baseURL ?? process.env.LOCAL_MODEL_BASE_URL ?? DEFAULT_OLLAMA_BASE_URL;
    this.temperature = temperature;
    this.costPerInputToken = costPerInputToken;
    this.costPerOutputToken = costPerOutputToken;
    this.extraParams = extraParams;
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
      ...this.extraParams,
    };

    const request: Record<string, unknown> = {
      model: this.modelName,
      messages: await ollamaMessages(prompt),
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
