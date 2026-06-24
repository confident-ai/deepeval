import { DeepEvalBaseEmbeddingModel, EmbeddingResult } from "./base-embedding-model";
import { computeCost, requireApiKey } from "../utils";

export interface OpenAIEmbeddingModelOptions {
  model?: string;
  apiKey?: string;
  baseURL?: string;
  dimensions?: number;
  costPerInputToken?: number;
}

const DEFAULT_MODEL = "text-embedding-3-small";

export class OpenAIEmbeddingModel extends DeepEvalBaseEmbeddingModel {
  private apiKey: string;
  private baseURL?: string;
  private dimensions?: number;
  private costPerInputToken?: number;
  private client?: any;

  protected providerLabel = "OpenAI Embedding";
  protected apiKeyEnvVar = "OPENAI_API_KEY";

  constructor(options: OpenAIEmbeddingModelOptions = {}) {
    super(options.model ?? process.env.OPENAI_EMBEDDING_MODEL_NAME ?? DEFAULT_MODEL);
    this.apiKey = options.apiKey ?? "";
    this.baseURL = options.baseURL;
    this.dimensions = options.dimensions;
    this.costPerInputToken = options.costPerInputToken;
  }

  private async getClient(): Promise<any> {
    if (!this.client) {
      const { default: OpenAI } = (await import("openai")) as any;
      this.client = new OpenAI({
        apiKey: requireApiKey(this.apiKey, this.providerLabel, this.apiKeyEnvVar),
        baseURL: this.baseURL,
      });
    }
    return this.client;
  }

  async embedText(text: string): Promise<EmbeddingResult> {
    const client = await this.getClient();
    const request: Record<string, unknown> = {
      model: this.modelName,
      input: text,
    };
    if (this.dimensions != null) {
      request.dimensions = this.dimensions;
    }
    const response = await client.embeddings.create(request);
    const embedding: number[] = response.data[0].embedding;
    const cost = computeCost(
      response.usage?.prompt_tokens,
      0,
      this.costPerInputToken,
      0,
    );
    return { embedding, cost };
  }

  async embedTexts(texts: string[]): Promise<EmbeddingResult[]> {
    const client = await this.getClient();
    const request: Record<string, unknown> = {
      model: this.modelName,
      input: texts,
    };
    if (this.dimensions != null) {
      request.dimensions = this.dimensions;
    }
    const response = await client.embeddings.create(request);
    const batchSize = texts.length;
    const results: EmbeddingResult[] = [];
    for (const item of response.data) {
      const cost = computeCost(
        response.usage?.prompt_tokens,
        0,
        this.costPerInputToken,
        0,
      );
      const perItemCost = cost != null ? cost / batchSize : null;
      results.push({
        embedding: item.embedding,
        cost: perItemCost,
      });
    }
    return results;
  }

  getModelName(): string {
    return this.modelName ?? this.providerLabel;
  }
}
