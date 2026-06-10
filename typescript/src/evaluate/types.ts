import { Turn } from "../test-case";

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
