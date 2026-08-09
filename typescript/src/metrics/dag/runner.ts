import type { DeepAcyclicGraph } from "@/metrics/dag/graph";
import type { JudgementVerdict } from "@/metrics/dag/schema";
import {
  type AnyNode,
  type ChildMetric,
  type DagHostMetric,
  type ExecutableNode,
  type VerdictLikeNode,
  isJudgementNode,
  isNode,
  isVerdictNode,
} from "@/metrics/dag/types";

export class DeepAcyclicGraphRunner {
  private readonly remaining: Map<AnyNode, number>;
  private readonly outputs = new Map<AnyNode, unknown>();
  private readonly verdicts = new Map<AnyNode, JudgementVerdict>();
  private readonly depth = new Map<AnyNode, number>();

  constructor(private readonly graph: DeepAcyclicGraph) {
    this.remaining = new Map(graph.indegree);
  }

  async run(metric: DagHostMetric, testCase: any): Promise<void> {
    await Promise.all(
      this.graph.rootNodes.map((root) => this.visit(root, metric, testCase, 0)),
    );
  }

  private checkRemaining(node: AnyNode): boolean {
    const left = (this.remaining.get(node) ?? 0) - 1;
    this.remaining.set(node, left);
    return left <= 0;
  }

  private verdictMatches(node: VerdictLikeNode): boolean {
    const parent = this.graph.parents.get(node)?.[0];
    if (parent && isJudgementNode(parent)) {
      return this.verdicts.get(parent)?.verdict === node.verdict;
    }
    return true;
  }

  private storeResult(node: ExecutableNode, result: unknown): void {
    if (node.nodeKind === "task") {
      this.outputs.set(node, result);
    } else {
      this.verdicts.set(node, result as JudgementVerdict);
    }
  }

  private async visit(
    node: AnyNode,
    metric: DagHostMetric,
    testCase: any,
    depth: number,
  ): Promise<void> {
    if (isVerdictNode(node)) {
      await this.visitVerdict(node, metric, testCase, depth);
      return;
    }
    this.depth.set(node, Math.max(0, this.depth.get(node) ?? 0, depth));
    if (!this.checkRemaining(node)) return;

    const result = await node.execute({
      metric,
      testCase,
      parents: this.graph.parents.get(node),
      outputs: this.outputs,
    });
    this.storeResult(node, result);

    const nodeDepth = this.depth.get(node)!;
    metric.verboseSteps.push(node.verboseLog(nodeDepth, result));
    await Promise.all(
      node.children.map((child) =>
        this.visit(child, metric, testCase, nodeDepth + 1),
      ),
    );
  }

  private async visitVerdict(
    node: VerdictLikeNode,
    metric: DagHostMetric,
    testCase: any,
    depth: number,
  ): Promise<void> {
    if (!this.checkRemaining(node)) return;
    if (!this.verdictMatches(node)) return;

    const child = node.child;
    if (child == null) {
      metric.verboseSteps.push(node.verboseLog(depth));
      metric.score = node.score! / 10;
      if (metric.includeReason) {
        metric.reason = await node.generateReason(metric);
      }
    } else if (isNode(child)) {
      await this.visit(child, metric, testCase, depth);
    } else {
      const childMetric = await node.runChildMetric(metric, testCase);
      this.applyChildMetric(node, childMetric, metric, depth);
    }
  }

  private applyChildMetric(
    node: VerdictLikeNode,
    childMetric: ChildMetric,
    metric: DagHostMetric,
    depth: number,
  ): void {
    metric.verboseSteps.push(node.verboseLog(depth, childMetric));
    metric.score = childMetric.score;
    if (metric.includeReason) metric.reason = childMetric.reason;
    metric.accrueCost(childMetric.evaluationCost ?? null);
  }
}
