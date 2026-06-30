export interface EmbeddingResult {
  embedding: number[];
  cost: number | null;
}

export abstract class DeepEvalBaseEmbeddingModel {
  modelName?: string;

  constructor(modelName?: string) {
    this.modelName = modelName;
  }

  abstract embedText(text: string): Promise<EmbeddingResult>;
  abstract embedTexts(texts: string[]): Promise<EmbeddingResult[]>;
  abstract getModelName(): string;
}
