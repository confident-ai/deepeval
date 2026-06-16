import { z } from "zod";
import {
  Turn,
  LLMTestCase,
  ConversationalTestCase,
  ArenaTestCase,
} from "../test-case";

// Mirrors deepeval/evaluate/types.py (+ test_run MetricData), trimmed to what
// the local runner needs today. Extend toward the Python shape as we grow.

export interface MetricData {
  name: string;
  threshold: number;
  success: boolean;
  score?: number;
  reason?: string;
  strictMode: boolean;
  evaluationModel?: string;
  evaluationCost?: number;
  verboseLogs?: string;
  error?: string;
  skipped: boolean;
}

export interface TestResult {
  name: string;
  success: boolean;
  metricsData: MetricData[] | null;
  conversational: boolean;
  index?: number;
  // single-turn fields
  input?: string;
  actualOutput?: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: string[];
  // conversational field
  turns?: Turn[];
}

export interface EvaluationResult {
  testResults: TestResult[];
  // Confident AI integration is not wired yet — placeholders for parity.
  confidentLink: string | null;
  testRunId: string | null;
}

// --- Confident AI posting payloads -----------------------------------------
// Defined as zod schemas: `z.infer` gives the type and `.parse(obj)` does the
// conversion (validating + stripping extra fields like `skipped`).

export const ApiToolCallSchema = z.object({
  name: z.string(),
  description: z.string().nullish(),
  reasoning: z.string().nullish(),
  output: z.unknown(),
  inputParameters: z.record(z.string(), z.unknown()).nullish(),
});
export type ApiToolCall = z.infer<typeof ApiToolCallSchema>;

export const ApiMetricDataSchema = z.object({
  name: z.string(),
  threshold: z.number(),
  success: z.boolean(),
  score: z.number().nullish(),
  reason: z.string().nullish(),
  strictMode: z.boolean(),
  evaluationModel: z.string().nullish(),
  error: z.string().nullish(),
  evaluationCost: z.number().nullish(),
  verboseLogs: z.string().nullish(),
});
export type ApiMetricData = z.infer<typeof ApiMetricDataSchema>;

export type AnyTestCase = LLMTestCase | ConversationalTestCase;

/** A test case paired with its metric results (input to `postTestRun`). */
export interface EvaluatedCase {
  testCase: AnyTestCase;
  metricsData: MetricData[];
  runDuration: number;
  /** A serialized trace (TraceApi) to embed + link on the posted test case. */
  trace?: unknown;
}

/** A snapshot of one arena case's result, since the metric is reused (input to `postExperiment`). */
export interface ArenaCaseResult {
  testCase: ArenaTestCase;
  index: number;
  winner: string | null;
  reason?: string;
  error?: string;
  evaluationModel?: string;
  evaluationCost?: number;
  verboseLogs?: string;
  runDuration: number;
}

export interface ContestantRun {
  identifier: string;
  testCases: Record<string, unknown>[];
  scores: number[];
  passes: number;
  fails: number;
  errors: number;
  testPassed: number;
  testFailed: number;
  runDuration: number;
  evaluationCost: number;
  hasCost: boolean;
  hyperparameters: Record<string, unknown>;
}
