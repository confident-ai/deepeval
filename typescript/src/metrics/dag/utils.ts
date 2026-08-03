import { GEval } from "@/metrics/g-eval/g-eval";
import { ConversationalGEval } from "@/metrics/conversational-g-eval/conversational-g-eval";
import {
  type AnyNode,
  type ChildMetric,
  type DagHostMetric,
  type EvalParam,
  isNode,
  isVerdictNode,
} from "@/metrics/dag/types";

/** Every node reachable from a verdict's `child` or a node's `children`. */
export function edgesOf(node: AnyNode): AnyNode[] {
  if (isVerdictNode(node)) {
    return node.child != null && isNode(node.child) ? [node.child] : [];
  }
  return [...node.children];
}

export function isValidDagFromRoots(rootNodes: AnyNode[]): boolean {
  const visited = new Set<AnyNode>();
  const stack = new Set<AnyNode>();

  const visit = (node: AnyNode): boolean => {
    if (stack.has(node)) return false;
    if (visited.has(node)) return true;
    visited.add(node);
    stack.add(node);
    for (const child of edgesOf(node)) {
      if (!visit(child)) return false;
    }
    stack.delete(node);
    return true;
  };

  return rootNodes.every(visit);
}

export function extractRequiredParams<T extends EvalParam>(
  rootNodes: AnyNode[],
): T[] {
  const params = new Set<T>();
  const seen = new Set<AnyNode>();

  const visit = (node: AnyNode): void => {
    if (seen.has(node)) return;
    seen.add(node);
    if (!isVerdictNode(node) && node.evaluationParams) {
      for (const param of node.evaluationParams) params.add(param as T);
    }
    for (const child of edgesOf(node)) visit(child);
  };

  rootNodes.forEach(visit);
  return [...params];
}

/**
 * Run a verdict's child metric on a copy, so the user's instance keeps its own
 * score and logs. Replaces Python's `copy_metrics`, which reconstructs from the
 * constructor signature — unavailable at runtime in TypeScript.
 *
 * G-Eval children adopt the host metric's model, matching Python's
 * `_build_child_metric`.
 */
export function cloneChildMetric<T extends ChildMetric>(
  child: T,
  host: DagHostMetric,
): T {
  const copy = Object.create(Object.getPrototypeOf(child)) as T;
  Object.assign(copy, child);
  copy.verboseMode = false;
  copy.showIndicator = false;
  if (child instanceof GEval || child instanceof ConversationalGEval) {
    copy.model = host.model;
    copy.evaluationModel = host.model?.getModelName();
  }
  return copy;
}

export function validateVerdictBranch(
  score: number | undefined,
  child: unknown,
  className: string,
): void {
  if (score != null && child != null) {
    throw new Error(
      `A ${className} can have either a 'score' or a 'child', but not both.`,
    );
  }
  if (score == null && child == null) {
    throw new Error(`A ${className} must have either a 'score' or a 'child'.`);
  }
  if (score != null && (score < 0 || score > 10)) {
    throw new Error("The score must be between 0 and 10, inclusive.");
  }
}
