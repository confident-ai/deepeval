import { Api, Endpoints, HttpMethods } from "../confident/api";
import {
  ConversationalTestCase,
  ToolCall,
  Turn,
  resolveRetrievalContext,
} from "../test-case";
import {
  MetricData,
  ApiToolCall,
  ApiToolCallSchema,
  ApiMetricData,
  ApiMetricDataSchema,
  EvaluatedCase,
  ArenaCaseResult,
  ContestantRun,
} from "./types";

// --- shared leaf conversions (zod parse validates + strips extra fields) ---

const convertTool = (t: ToolCall): ApiToolCall => ApiToolCallSchema.parse(t);
const convertMetricData = (m: MetricData): ApiMetricData =>
  ApiMetricDataSchema.parse(m);

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

function caseCost(metricsData: MetricData[]): number | undefined {
  const costs = metricsData
    .map((m) => m.evaluationCost)
    .filter((c): c is number => c != null);
  return costs.length ? costs.reduce((s, c) => s + c, 0) : undefined;
}

function buildMetricsScores(cases: EvaluatedCase[]) {
  const map = new Map<
    string,
    { scores: number[]; passes: number; fails: number; errors: number }
  >();
  for (const { metricsData } of cases) {
    for (const m of metricsData) {
      if (m.skipped) continue;
      const e = map.get(m.name) ?? { scores: [], passes: 0, fails: 0, errors: 0 };
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

export async function postTestRun(
  cases: EvaluatedCase[],
  runDuration: number,
  official = false,
  silent = false,
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

  cases.forEach(({ testCase, metricsData, runDuration: caseDuration, trace }, order) => {
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
        runDuration: caseDuration,
        evaluationCost,
        order,
        turns: testCase.turns.map((t, i) => convertTurn(t, i)),
        scenario: testCase.scenario,
        expectedOutcome: testCase.expectedOutcome,
        userDescription: testCase.userDescription,
        chatbotRole: testCase.chatbotRole,
        imagesMapping: testCase.getImagesMapping(),
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
        runDuration: caseDuration,
        evaluationCost,
        order,
        imagesMapping: testCase.getImagesMapping(),
        trace,
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
    official: official || undefined,
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
    if (link && !silent) {
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

function arenaMetricData(
  r: ArenaCaseResult,
  contestantName: string,
  metricName: string,
): ApiMetricData {
  const won = r.winner === contestantName;
  return {
    name: metricName,
    threshold: 1,
    strictMode: true,
    evaluationModel: r.evaluationModel,
    evaluationCost: r.evaluationCost,
    verboseLogs: r.verboseLogs,
    ...(r.error != null
      ? { success: false, error: r.error }
      : { success: won, score: won ? 1 : 0, reason: r.reason }),
  };
}

export async function postExperiment(
  results: ArenaCaseResult[],
  metricName: string,
  name: string,
): Promise<{ link: string | null }> {
  const apiKey = process.env.CONFIDENT_API_KEY;
  if (!apiKey || apiKey.trim() === "" || results.length === 0) {
    return { link: null };
  }

  const runMap = new Map<string, ContestantRun>();
  const ensure = (n: string): ContestantRun => {
    let e = runMap.get(n);
    if (!e) {
      e = {
        identifier: n,
        testCases: [],
        scores: [],
        passes: 0,
        fails: 0,
        errors: 0,
        testPassed: 0,
        testFailed: 0,
        runDuration: 0,
        evaluationCost: 0,
        hasCost: false,
        hyperparameters: {},
      };
      runMap.set(n, e);
    }
    return e;
  };

  for (const r of results) {
    for (const contestant of r.testCase.contestants) {
      const e = ensure(contestant.name);
      const won = r.winner === contestant.name;
      const md = arenaMetricData(r, contestant.name, metricName);
      const tc = contestant.testCase;
      e.testCases.push({
        name: tc.name ?? `test_case_${r.index}`,
        input: tc.input,
        actualOutput: tc.actualOutput,
        expectedOutput: tc.expectedOutput,
        context: tc.context,
        retrievalContext: resolveRetrievalContext(tc.retrievalContext),
        toolsCalled: tc.toolsCalled?.map(convertTool),
        expectedTools: tc.expectedTools?.map(convertTool),
        success: md.success,
        metricsData: [md],
        runDuration: r.runDuration,
        evaluationCost: md.evaluationCost,
        order: r.index,
      });
      e.runDuration += r.runDuration;
      if (r.error != null) {
        e.errors += 1;
      } else {
        e.scores.push(won ? 1 : 0);
        if (won) {
          e.passes += 1;
          e.testPassed += 1;
        } else {
          e.fails += 1;
          e.testFailed += 1;
        }
      }
      if (md.evaluationCost != null) {
        e.evaluationCost += md.evaluationCost;
        e.hasCost = true;
      }
      if (contestant.hyperparameters) {
        Object.assign(e.hyperparameters, contestant.hyperparameters);
      }
    }
  }

  const testRuns = [...runMap.values()].map((e) => ({
    testCases: e.testCases,
    conversationalTestCases: [],
    metricsScores: [
      { metric: metricName, scores: e.scores, passes: e.passes, fails: e.fails, errors: e.errors },
    ],
    identifier: e.identifier,
    testPassed: e.testPassed,
    testFailed: e.testFailed,
    runDuration: e.runDuration,
    evaluationCost: e.hasCost ? e.evaluationCost : undefined,
    hyperparameters: Object.keys(e.hyperparameters).length
      ? e.hyperparameters
      : undefined,
  }));

  try {
    const api = new Api();
    const result = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.EXPERIMENT_ENDPOINT,
      { testRuns, name },
    );
    const link = result?.link ?? null;
    if (link) console.log(`\n✓ Done 🎉! View results on ${link}`);
    return { link };
  } catch (e) {
    console.warn(
      `Confident AI: failed to post experiment — ${(e as Error).message}`,
    );
    return { link: null };
  }
}
