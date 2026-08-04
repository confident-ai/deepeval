import type { BaseMetricCore } from "@/metrics/base-metrics";
import type { SingleTurnParams, MultiTurnParams } from "@/test-case";

export type EvalParam = SingleTurnParams | MultiTurnParams;

export type NodeKind =
  | "task"
  | "binaryJudgement"
  | "nonBinaryJudgement"
  | "verdict";

/** A metric that can host a DAG: the base metric plus the traversal log. */
export type DagHostMetric = BaseMetricCore & { verboseSteps: string[] };

/** A metric a verdict can hand off to (single-turn or conversational). */
export type ChildMetric = BaseMetricCore & {
  measure(testCase: any, ...args: any[]): number | Promise<number>;
};

export interface NodeContext {
  metric: DagHostMetric;
  testCase: any;
  parents?: AnyNode[];
  outputs: Map<AnyNode, unknown>;
}

/** A task or judgement node — it runs, produces a result, and has children. */
export interface ExecutableNode {
  readonly nodeKind: "task" | "binaryJudgement" | "nonBinaryJudgement";
  readonly multiTurn: boolean;
  readonly children: AnyNode[];
  label?: string;
  evaluationParams?: EvalParam[];
  validate(): void;
  execute(ctx: NodeContext): Promise<unknown>;
  verboseLog(depth: number, result: unknown): string;
}

/** A terminal branch: either a fixed score, or a handoff to a node or metric. */
export interface VerdictLikeNode {
  readonly nodeKind: "verdict";
  readonly multiTurn: boolean;
  readonly verdict: string | boolean;
  readonly score?: number;
  readonly child?: AnyNode | ChildMetric;
  validate(): void;
  generateReason(metric: DagHostMetric): Promise<string>;
  runChildMetric(metric: DagHostMetric, testCase: any): Promise<ChildMetric>;
  verboseLog(depth: number, childMetric?: ChildMetric): string;
}

export type AnyNode = ExecutableNode | VerdictLikeNode;

export function isNode(value: unknown): value is AnyNode {
  return (
    value != null &&
    typeof (value as AnyNode).nodeKind === "string" &&
    typeof (value as AnyNode).validate === "function"
  );
}

export function isVerdictNode(node: AnyNode): node is VerdictLikeNode {
  return node.nodeKind === "verdict";
}

export function isExecutableNode(node: AnyNode): node is ExecutableNode {
  return node.nodeKind !== "verdict";
}

export function isJudgementNode(node: AnyNode): boolean {
  return (
    node.nodeKind === "binaryJudgement" ||
    node.nodeKind === "nonBinaryJudgement"
  );
}
