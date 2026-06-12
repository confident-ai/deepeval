import type { ZodType } from "zod";
import { DeepEvalBaseLLM, type GenerationResult } from "../base-model";
import {
  computeCost,
  extractJson,
  importOptional,
  requireApiKey,
} from "../utils";

const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";

export interface GeminiModelOptions {
  model?: string;
  apiKey?: string;
  temperature?: number;
  useVertexAI?: boolean;
  project?: string;
  location?: string;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class GeminiModel extends DeepEvalBaseLLM {
  private readonly apiKey: string;
  private readonly temperature?: number;
  private readonly useVertexAI: boolean;
  private readonly project?: string;
  private readonly location?: string;
  private readonly costPerInputToken?: number;
  private readonly costPerOutputToken?: number;
  private client?: any;

  constructor(options: GeminiModelOptions = {}) {
    super(
      options.model ?? process.env.GEMINI_MODEL_NAME ?? DEFAULT_GEMINI_MODEL,
    );
    this.apiKey =
      options.apiKey ??
      process.env.GOOGLE_API_KEY ??
      process.env.GEMINI_API_KEY ??
      "";
    // Only sent when explicitly set — some models (e.g. reasoning models) reject `temperature`.
    this.temperature = options.temperature;
    this.useVertexAI =
      options.useVertexAI ?? process.env.GOOGLE_GENAI_USE_VERTEXAI === "true";
    this.project = options.project ?? process.env.GOOGLE_CLOUD_PROJECT;
    this.location = options.location ?? process.env.GOOGLE_CLOUD_LOCATION;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
  }

  private async getClient(): Promise<any> {
    if (!this.client) {
      const { GoogleGenAI } = await importOptional("@google/genai", "Gemini");
      this.client = this.useVertexAI
        ? new GoogleGenAI({
            vertexai: true,
            project: this.project,
            location: this.location,
          })
        : new GoogleGenAI({
            apiKey: requireApiKey(this.apiKey, "Gemini", "GOOGLE_API_KEY"),
          });
    }
    return this.client;
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    const client = await this.getClient();

    const config: Record<string, unknown> = {
      ...(this.temperature !== undefined && { temperature: this.temperature }),
    };
    if (schema) {
      config.responseMimeType = "application/json";
    }

    const response = await client.models.generateContent({
      model: this.modelName,
      contents: prompt,
      config,
    });

    const text: string = response.text ?? "";
    const cost = computeCost(
      response.usageMetadata?.promptTokenCount,
      response.usageMetadata?.candidatesTokenCount,
      this.costPerInputToken,
      this.costPerOutputToken,
    );

    if (schema) {
      return { output: schema.parse(extractJson(text)), cost };
    }
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? DEFAULT_GEMINI_MODEL;
  }

  supportsMultimodal(): boolean {
    return true;
  }
}
