import { DeepEvalBaseLLM } from "../models";
import { Golden, ConversationalGolden } from "../dataset";
import { Prompt } from "../prompt";

export type ModelCallback = (
  prompt: Prompt,
  golden: Golden | ConversationalGolden,
) => Promise<string>;

export interface OptimizationConfig {
  iterations?: number;
  minibatchSize?: number;
  paretoSize?: number;
  patience?: number;
  randomSeed?: number;
}

export interface IterationLog {
  iteration: number;
  outcome: "accepted" | "rejected";
  reason: string;
  before?: number;
  after?: number;
}

export interface AcceptedIteration {
  parent: string;
  child: string;
  module: string;
  before: number;
  after: number;
}

export interface OptimizationReport {
  optimizationId: string;
  bestPrompt: Prompt;
  acceptedIterations: AcceptedIteration[];
  logs: IterationLog[];
}
