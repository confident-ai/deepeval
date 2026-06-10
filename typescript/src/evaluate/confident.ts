import { Api, Endpoints, HttpMethods } from "../confident/api";
import {
  LLMTestCase,
  ConversationalTestCase,
  ToolCall,
  Turn,
  resolveRetrievalContext,
} from "../test-case";
import { MetricData } from "./types";

type AnyTestCase = LLMTestCase | ConversationalTestCase;

export interface EvaluatedCase {
  testCase: AnyTestCase;
  metricsData: MetricData[];
  runDuration: number;
}

interface ApiToolCall {
  name: string;
  description?: string;
  reasoning?: string;
  output?: unknown;
  inputParameters?: Record<string, unknown>;
}

interface ApiMetricData {
  name: string;
  threshold: number;
  success: boolean;
  score?: number;
  reason?: string;
  strictMode?: boolean;
  evaluationModel?: string;
  error?: string;
  evaluationCost?: number;
  verboseLogs?: string;
}

function convertTool(t: ToolCall): ApiToolCall {
  return {
    name: t.name,
    description: t.description,
    reasoning: t.reasoning,
    output: t.output,
    inputParameters: t.inputParameters,
  };
}

function convertMetricData(m: MetricData): ApiMetricData {
  return {
    name: m.name,
    threshold: m.threshold,
    success: m.success,
    score: m.score,
    reason: m.reason,
    strictMode: m.strictMode,
    evaluationModel: m.evaluationModel,
    error: m.error,
    evaluationCost: m.evaluationCost,
    verboseLogs: m.verboseLogs,
  };
}

/** Sum of the (known) per-metric costs, or undefined if none are known. */
function caseCost(metricsData: MetricData[]): number | undefined {
  const costs = metricsData
    .map((m) => m.evaluationCost)
    .filter((c): c is number => c != null);
  return costs.length ? costs.reduce((s, c) => s + c, 0) : undefined;
}

function convertTurn(turn: Turn, order: number) {
  return {
    role: turn.role,
    content: turn.content,
    order,
    userId: turn.userId,
    retrievalContext: resolveRetrievalContext(turn.retrievalContext),
    toolsCalled: turn.toolsCalled?.map(convertTool),
  };
}

/** Per-metric aggregate scores across all evaluated cases (Python `MetricScores`). */
function buildMetricsScores(cases: EvaluatedCase[]) {
  const map = new Map<
    string,
    { scores: number[]; passes: number; fails: number; errors: number }
  >();
  for (const { metricsData } of cases) {
    for (const m of metricsData) {
      if (m.skipped) continue;
      const e = map.get(m.name) ?? {
        scores: [],
        passes: 0,
        fails: 0,
        errors: 0,
      };
      if (m.error) {
        e.errors += 1;
      } else {
        if (m.score != null) e.scores.push(m.score);
        if (m.success) e.passes += 1;
        else e.fails += 1;
      }
      map.set(m.name, e);
    }
  }
  return [...map.entries()].map(([metric, e]) => ({ metric, ...e }));
}

/**
 * Post the evaluation results to Confident AI as a TestRun (mirrors Python's
 * `post_test_run`). No-op (returns nulls) when not logged in. Never throws —
 * a posting failure is logged and evaluation results are still returned.
 */
export async function postTestRun(
  cases: EvaluatedCase[],
  runDuration: number,
): Promise<{ link: string | null; testRunId: string | null }> {
  // Silent check (isConfident() logs a warning — not wanted on the no-op path).
  const apiKey = process.env.CONFIDENT_API_KEY;
  if (!apiKey || apiKey.trim() === "" || cases.length === 0) {
    return { link: null, testRunId: null };
  }

  const testCases: Record<string, unknown>[] = [];
  const conversationalTestCases: Record<string, unknown>[] = [];
  let testPassed = 0;
  let testFailed = 0;
  let totalCost = 0;
  let hasCost = false;

  cases.forEach(({ testCase, metricsData, runDuration }, order) => {
    const success = metricsData.every((m) => m.skipped || m.success);
    if (success) testPassed += 1;
    else testFailed += 1;

    const evaluationCost = caseCost(metricsData);
    if (evaluationCost != null) {
      totalCost += evaluationCost;
      hasCost = true;
    }
    const metricsDataApi = metricsData.map(convertMetricData);

    if (testCase instanceof ConversationalTestCase) {
      conversationalTestCases.push({
        name: testCase.name ?? `test_case_${order}`,
        success,
        metricsData: metricsDataApi,
        runDuration,
        evaluationCost,
        order,
        turns: testCase.turns.map((t, i) => convertTurn(t, i)),
        scenario: testCase.scenario,
        expectedOutcome: testCase.expectedOutcome,
        userDescription: testCase.userDescription,
        chatbotRole: testCase.chatbotRole,
      });
    } else {
      testCases.push({
        name: testCase.name ?? `test_case_${order}`,
        input: testCase.input,
        actualOutput: testCase.actualOutput,
        expectedOutput: testCase.expectedOutput,
        context: testCase.context,
        retrievalContext: resolveRetrievalContext(testCase.retrievalContext),
        toolsCalled: testCase.toolsCalled?.map(convertTool),
        expectedTools: testCase.expectedTools?.map(convertTool),
        success,
        metricsData: metricsDataApi,
        runDuration,
        evaluationCost,
        order,
      });
    }
  });

  const payload = {
    testCases,
    conversationalTestCases,
    metricsScores: buildMetricsScores(cases),
    testPassed,
    testFailed,
    runDuration,
    evaluationCost: hasCost ? totalCost : undefined,
  };

  try {
    const api = new Api();
    const result = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.TEST_RUN_ENDPOINT,
      payload,
    );
    const link = result?.link ?? null;
    const testRunId = result?.id ?? null;
    if (link) {
      console.log(`\n✅ Test run posted to Confident AI: ${link}`);
    }
    return { link, testRunId };
  } catch (e) {
    console.warn(
      `Confident AI: failed to post test run — ${(e as Error).message}`,
    );
    return { link: null, testRunId: null };
  }
}
