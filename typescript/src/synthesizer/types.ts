export enum Evolution {
  Reasoning = "Reasoning",
  MultiContext = "Multi-context",
  Concretizing = "Concretizing",
  Constrained = "Constrained",
  Comparative = "Comparative",
  Hypothetical = "Hypothetical",
  InBreadth = "In-Breadth",
}

export enum PromptEvolution {
  Reasoning = "Reasoning",
  Concretizing = "Concretizing",
  Constrained = "Constrained",
  Comparative = "Comparative",
  Hypothetical = "Hypothetical",
  InBreadth = "In-Breadth",
}

export interface FiltrationConfig {
  syntheticInputQualityThreshold?: number;
  maxQualityRetries?: number;
}

export interface EvolutionConfig {
  numEvolutions?: number;
  evolutions?: Partial<Record<Evolution, number>>;
}

export interface StylingConfig {
  scenario?: string;
  task?: string;
  inputFormat?: string;
  expectedOutputFormat?: string;
}

export interface ConversationalStylingConfig {
  scenarioContext?: string;
  conversationalTask?: string;
  participantRoles?: string;
  scenarioFormat?: string;
  expectedOutcomeFormat?: string;
}

export interface ContextConstructionConfig {
  maxContextsPerDocument?: number;
  minContextsPerDocument?: number;
  maxContextLength?: number;
  minContextLength?: number;
  chunkSize?: number;
  chunkOverlap?: number;
  contextQualityThreshold?: number;
  contextSimilarityThreshold?: number;
  maxRetries?: number;
  allowCrossFileContexts?: boolean;
  targetFilesPerContext?: number | null;
  maxFilesPerContext?: number;
}

export interface SynthesizerOptions {
  asyncMode?: boolean;
  maxConcurrent?: number;
  filtrationConfig?: FiltrationConfig;
  evolutionConfig?: EvolutionConfig;
  stylingConfig?: StylingConfig;
  conversationalStylingConfig?: ConversationalStylingConfig;
}
