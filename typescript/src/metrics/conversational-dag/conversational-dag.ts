import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import { initializeModel, constructVerboseLogs } from "@/metrics/utils";
import { checkConversationalTestCaseParams } from "@/metrics/conversational-utils";
import { Api, Endpoints, HttpMethods } from "@/confident/api";
import { DeepAcyclicGraph } from "@/metrics/dag/graph";
import {
  extractRequiredParams,
  isValidDagFromRoots,
} from "@/metrics/dag/utils";
import {
  buildDagFromPayload,
  constructDagUploadPayload,
} from "@/metrics/dag/serialization";

export interface ConversationalDAGMetricOptions {
  name: string;
  dag: DeepAcyclicGraph;
  model?: DeepEvalBaseLLM | string;
  threshold?: number | null;
  flaky?: boolean;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  includeDagSuffix?: boolean;
}

/**
 * Conversational DAG — the multi-turn counterpart of {@link DAGMetric}: the same
 * deterministic decision tree, with nodes that read turns (optionally narrowed
 * to a `turnWindow`) instead of a single-turn test case.
 */
export class ConversationalDAGMetric extends BaseConversationalMetric {
  dag: DeepAcyclicGraph;
  verboseSteps: string[] = [];
  metricId?: string;
  readonly metricName: string;
  private readonly includeDagSuffix: boolean;

  constructor(options: ConversationalDAGMetricOptions) {
    if (!isValidDagFromRoots(options.dag.rootNodes)) {
      throw new Error("Cycle detected in DAG graph.");
    }
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
    });

    this.metricName = options.name;
    this.dag = options.dag;
    this.includeDagSuffix = options.includeDagSuffix ?? true;

    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      this.requiredParams = extractRequiredParams<MultiTurnParams>(
        this.dag.rootNodes,
      );
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
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
    const payload = constructDagUploadPayload(this.metricName, this.dag, true);
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
        `Metric '${this.metricName}' has no DAG graph and cannot be pulled as a ConversationalDAGMetric.`,
      );
    }
    this.dag = buildDagFromPayload(body.dag, true);
    this.metricId = body.id;
    console.log(`✓ Metric '${this.metricName}' [DAG] pulled successfully`);
    return data;
  }

  get name(): string {
    return this.includeDagSuffix ? `${this.metricName} [DAG]` : this.metricName;
  }
}
