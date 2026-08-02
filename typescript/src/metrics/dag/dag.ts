import { BaseMetric } from "../base-metrics";
import { LLMTestCase, SingleTurnParams } from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import {
  initializeModel,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "../utils";
import { Api, Endpoints, HttpMethods } from "../../confident/api";
import { DeepAcyclicGraph } from "./graph";
import { extractRequiredParams, isValidDagFromRoots } from "./utils";
import {
  buildDagFromPayload,
  constructDagUploadPayload,
} from "./serialization";

export interface DAGMetricOptions {
  name: string;
  dag: DeepAcyclicGraph;
  model?: DeepEvalBaseLLM | string;
  threshold?: number;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  includeDagSuffix?: boolean;
}

/**
 * DAG — a deterministic decision tree of LLM judgements. Each node either
 * transforms the test case (`TaskNode`) or picks a branch (`BinaryJudgementNode`,
 * `NonBinaryJudgementNode`); the `VerdictNode` the traversal lands on supplies
 * the score, either directly or by delegating to another metric.
 */
export class DAGMetric extends BaseMetric {
  dag: DeepAcyclicGraph;
  verboseSteps: string[] = [];
  metricId?: string;
  readonly metricName: string;
  private readonly includeDagSuffix: boolean;

  constructor(options: DAGMetricOptions) {
    if (!isValidDagFromRoots(options.dag.rootNodes)) {
      throw new Error("Cycle detected in DAG graph.");
    }
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });

    this.metricName = options.name;
    this.dag = options.dag;
    this.includeDagSuffix = options.includeDagSuffix ?? true;

    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      this.requiredParams = extractRequiredParams<SingleTurnParams>(
        this.dag.rootNodes,
      );
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      this.verboseSteps = [];

      await this.dag.execute(this, testCase);
      this.success = this.isSuccessful();
      this.verboseLogs = constructVerboseLogs(this, [
        ...this.verboseSteps,
        `Score: ${this.score}\nReason: ${this.reason}`,
      ]);
      return this.score!;
    } finally {
      this.stopProgress();
    }
  }

  /** Push this DAG to Confident AI as a reusable metric definition. */
  async upload(): Promise<any> {
    const api = new Api();
    const payload = constructDagUploadPayload(this.metricName, this.dag, false);
    const data = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.METRICS_ENDPOINT,
      payload,
    );
    this.metricId = data?.data?.id ?? data?.id;
    console.log(
      `✓ Metric '${this.metricName}' [DAG] uploaded successfully (id: ${this.metricId})`,
    );
    return data;
  }

  /** Replace this metric's DAG with the one stored on Confident AI. */
  async pull(): Promise<any> {
    const api = new Api();
    const data = await api.sendRequest(
      HttpMethods.GET,
      Endpoints.METRIC_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { name: this.metricName },
    );
    const body = data?.data ?? data;
    if (!body?.dag) {
      throw new Error(
        `Metric '${this.metricName}' has no DAG graph and cannot be pulled as a DAGMetric.`,
      );
    }
    this.dag = buildDagFromPayload(body.dag, false);
    this.metricId = body.id;
    console.log(`✓ Metric '${this.metricName}' [DAG] pulled successfully`);
    return data;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return this.includeDagSuffix
      ? `${this.metricName} [DAG]`
      : this.metricName;
  }
}
