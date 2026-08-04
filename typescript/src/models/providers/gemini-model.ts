import type { ZodType } from "zod";
import { parseBool } from "@/config/utils";
import {
  DeepEvalBaseLLM,
  type GenerationKwargs,
  type GenerationResult,
} from "@/models/base-model";
import { extractJson, importOptional, requireApiKey } from "@/models/utils";
import { geminiContents } from "@/models/multimodal";
import { defaultModelName, type ModelNamespace } from "@/models/registry";

export interface GeminiModelOptions {
  model?: string;
  apiKey?: string;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
  useVertexAI?: boolean;
  project?: string;
  location?: string;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  /** Extra params merged into the `generateContent` request `config`. */
  generationKwargs?: GenerationKwargs;
}

export class GeminiModel extends DeepEvalBaseLLM {
  private readonly apiKey: string;
  private readonly useVertexAI: boolean;
  private readonly project?: string;
  private readonly location?: string;
  private readonly generationKwargs: GenerationKwargs;
  private client?: any;
  protected registryNamespace: ModelNamespace = "gemini";

  constructor(options: GeminiModelOptions = {}) {
    const useVertexAI =
      options.useVertexAI ??
      parseBool(process.env.GOOGLE_GENAI_USE_VERTEXAI) ??
      false;

    super(
      options.model ??
        // Vertex AI deployments name models differently from the Gemini API.
        (useVertexAI ? process.env.VERTEX_AI_MODEL_NAME : undefined) ??
        process.env.GEMINI_MODEL_NAME ??
        defaultModelName("gemini"),
    );
    this.apiKey =
      options.apiKey ??
      process.env.GOOGLE_API_KEY ??
      process.env.GEMINI_API_KEY ??
      "";
    this.temperature = options.temperature;
    this.useVertexAI = useVertexAI;
    this.project = options.project ?? process.env.GOOGLE_CLOUD_PROJECT;
    this.location = options.location ?? process.env.GOOGLE_CLOUD_LOCATION;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
    this.generationKwargs = { ...options.generationKwargs };
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

    const temperature = this.resolveTemperature();
    const config: Record<string, unknown> = {
      ...(temperature !== undefined && { temperature }),
    };
    if (schema) {
      config.responseMimeType = "application/json";
    }
    Object.assign(config, this.generationKwargs);

    const response = await client.models.generateContent({
      model: this.modelName,
      contents: await geminiContents(prompt),
      config,
    });

    const text: string = response.text ?? "";
    const cost = this.resolveCost(
      response.usageMetadata?.promptTokenCount,
      response.usageMetadata?.candidatesTokenCount,
    );

    if (schema) {
      return { output: schema.parse(extractJson(text)), cost };
    }
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? defaultModelName("gemini");
  }

  supportsMultimodal(): boolean {
    return this.modelData.supportsMultimodal ?? true;
  }
}
