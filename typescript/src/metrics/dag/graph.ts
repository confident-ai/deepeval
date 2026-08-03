import {
  type AnyNode,
  type DagHostMetric,
  isJudgementNode,
} from "@/metrics/dag/types";
import { edgesOf } from "@/metrics/dag/utils";
import { DeepAcyclicGraphRunner } from "@/metrics/dag/runner";

export interface DeepAcyclicGraphOptions {
  rootNodes: AnyNode[];
}

export class DeepAcyclicGraph {
  readonly rootNodes: AnyNode[];
  readonly multiturn: boolean;
  readonly indegree = new Map<AnyNode, number>();
  readonly parents = new Map<AnyNode, AnyNode[]>();

  constructor(options: DeepAcyclicGraphOptions) {
    const rootNodes = options.rootNodes;
    if (!rootNodes || rootNodes.length === 0) {
      throw new Error("A DeepAcyclicGraph must have at least one root node.");
    }
    if (rootNodes.some((node) => node.multiTurn !== rootNodes[0].multiTurn)) {
      throw new Error("You cannot mix multi and single turn nodes");
    }
    if (rootNodes.length > 1 && rootNodes.some(isJudgementNode)) {
      throw new Error(
        "You cannot provide more than one root node when a " +
          "BinaryJudgementNode or NonBinaryJudgementNode is a root.",
      );
    }

    this.rootNodes = rootNodes;
    this.multiturn = rootNodes[0].multiTurn;
    this.buildGraph();
  }

  private buildGraph(): void {
    const visited = new Set<AnyNode>();
    const stack = new Set<AnyNode>();

    const visit = (node: AnyNode): void => {
      if (stack.has(node)) {
        throw new Error("Cycle detected in DAG graph.");
      }
      if (visited.has(node)) return;
      visited.add(node);
      node.validate();
      stack.add(node);
      if (!this.indegree.has(node)) this.indegree.set(node, 0);
      for (const child of edgesOf(node)) {
        this.indegree.set(child, (this.indegree.get(child) ?? 0) + 1);
        const parents = this.parents.get(child) ?? [];
        parents.push(node);
        this.parents.set(child, parents);
        visit(child);
      }
      stack.delete(node);
    };

    this.rootNodes.forEach(visit);
  }

  async execute(metric: DagHostMetric, testCase: any): Promise<void> {
    await new DeepAcyclicGraphRunner(this).run(metric, testCase);
  }
}
