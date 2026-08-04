import { randomUUID } from "node:crypto";
import { MultiTurnParams, SingleTurnParams } from "@/test-case";
import { GEval } from "@/metrics/g-eval/g-eval";
import { ConversationalGEval } from "@/metrics/conversational-g-eval/conversational-g-eval";
import {
  BinaryJudgementNode,
  NonBinaryJudgementNode,
  TaskNode,
  VerdictNode,
} from "@/metrics/dag/nodes";
import {
  ConversationalBinaryJudgementNode,
  ConversationalNonBinaryJudgementNode,
  ConversationalTaskNode,
  ConversationalVerdictNode,
  type TurnWindow,
} from "@/metrics/conversational-dag/nodes";
import { DeepAcyclicGraph } from "@/metrics/dag/graph";
import {
  type AnyNode,
  type ChildMetric,
  type ExecutableNode,
  type VerdictLikeNode,
  isNode,
  isVerdictNode,
} from "@/metrics/dag/types";
import { edgesOf, isValidDagFromRoots } from "@/metrics/dag/utils";

export enum NodeType {
  TASK = "TaskNode",
  BINARY_JUDGEMENT = "BinaryJudgementNode",
  NON_BINARY_JUDGEMENT = "NonBinaryJudgementNode",
  VERDICT = "VerdictNode",
}

export enum ChildType {
  NODE = "node",
  GEVAL = "geval",
  METRIC = "metric",
}

const NODE_TYPES: Record<AnyNode["nodeKind"], NodeType> = {
  task: NodeType.TASK,
  binaryJudgement: NodeType.BINARY_JUDGEMENT,
  nonBinaryJudgement: NodeType.NON_BINARY_JUDGEMENT,
  verdict: NodeType.VERDICT,
};

type NodeConstructors = Record<NodeType, any>;

const NODE_CLASSES: Record<"single" | "multi", NodeConstructors> = {
  single: {
    [NodeType.TASK]: TaskNode,
    [NodeType.BINARY_JUDGEMENT]: BinaryJudgementNode,
    [NodeType.NON_BINARY_JUDGEMENT]: NonBinaryJudgementNode,
    [NodeType.VERDICT]: VerdictNode,
  },
  multi: {
    [NodeType.TASK]: ConversationalTaskNode,
    [NodeType.BINARY_JUDGEMENT]: ConversationalBinaryJudgementNode,
    [NodeType.NON_BINARY_JUDGEMENT]: ConversationalNonBinaryJudgementNode,
    [NodeType.VERDICT]: ConversationalVerdictNode,
  },
};

/**
 * Metric classes a `metric`-typed verdict child can be rebuilt from. TypeScript
 * has no import-by-name, so anything beyond G-Eval must be registered by the
 * caller before deserializing.
 */
export const METRIC_CLASS_REGISTRY: Record<string, new (options: any) => any> =
  {
    GEval,
    ConversationalGEval,
  };

export function registerMetricClass(
  name: string,
  metricClass: new (options: any) => any,
): void {
  METRIC_CLASS_REGISTRY[name] = metricClass;
}

// ---------------------------------------------------------------- serializing

/** BFS from each root, every reachable node exactly once, roots first. */
export function walkNodes(rootNodes: AnyNode[]): AnyNode[] {
  const seen = new Set<AnyNode>();
  const ordered: AnyNode[] = [];
  const queue = [...rootNodes];
  while (queue.length > 0) {
    const node = queue.shift()!;
    if (seen.has(node)) continue;
    seen.add(node);
    ordered.push(node);
    queue.push(...edgesOf(node));
  }
  return ordered;
}

function assignIds(nodes: AnyNode[]): Map<AnyNode, string> {
  return new Map(nodes.map((node) => [node, randomUUID()]));
}

/**
 * Params travel snake_cased so a DAG serialized here can be read by the Python
 * SDK, whose enum values are the snake_case spellings of these same fields.
 */
function toWireParam(value: string): string {
  return value.replace(/[A-Z]/g, (char) => `_${char.toLowerCase()}`);
}

function fromWireParam(value: string): string {
  return value.replace(/_([a-z])/g, (_, char: string) => char.toUpperCase());
}

function serializeEvalParams(
  params?: Array<SingleTurnParams | MultiTurnParams>,
): string[] | null {
  return params == null ? null : params.map((param) => toWireParam(param));
}

/** Public metric options we can round-trip; anything else is dropped. */
function serializeMetricOptions(metric: any): Record<string, unknown> {
  const options: Record<string, unknown> = {};
  const put = (key: string, value: unknown) => {
    if (value != null) options[key] = value;
  };
  put("name", metric.metricName);
  put("criteria", metric.criteria);
  put("evaluation_steps", metric.evaluationSteps);
  put("evaluation_params", serializeEvalParams(metric.evaluationParams));
  put("threshold", metric.threshold);
  put("strict_mode", metric.strictMode);
  return options;
}

function serializeVerdictChild(
  child: AnyNode | ChildMetric,
  ids: Map<AnyNode, string>,
): Record<string, unknown> {
  if (isNode(child)) {
    return { type: ChildType.NODE, ref: ids.get(child) };
  }
  if (child instanceof GEval || child instanceof ConversationalGEval) {
    return { type: ChildType.GEVAL, ...serializeMetricOptions(child) };
  }
  return {
    type: ChildType.METRIC,
    metric_class: child.constructor.name,
    kwargs: serializeMetricOptions(child),
  };
}

type ChildSerializer = (
  child: AnyNode | ChildMetric,
  ids: Map<AnyNode, string>,
) => Record<string, unknown>;

function serializeNode(
  node: AnyNode,
  ids: Map<AnyNode, string>,
  serializeChild: ChildSerializer,
): Record<string, unknown> {
  const type = NODE_TYPES[node.nodeKind];

  if (isVerdictNode(node)) {
    const out: Record<string, unknown> = { type, verdict: node.verdict };
    if (node.score != null) out.score = node.score;
    if (node.child != null) {
      out.child = serializeChild(node.child, ids);
    }
    return out;
  }

  const turnWindow = (node as { turnWindow?: TurnWindow }).turnWindow;
  const out: Record<string, unknown> = {
    type,
    label: node.label ?? null,
    evaluation_params: serializeEvalParams(node.evaluationParams),
    children: node.children.map((child) => ids.get(child)),
  };
  if (node.nodeKind === "task") {
    const task = node as unknown as TaskNode;
    out.instructions = task.instructions;
    out.output_label = task.outputLabel;
  } else {
    out.criteria = (node as unknown as BinaryJudgementNode).criteria;
  }
  if (turnWindow != null) out.turn_window = [...turnWindow];
  return out;
}

function serializeDag(
  dag: DeepAcyclicGraph,
  serializeChild: ChildSerializer,
): Record<string, unknown> {
  if (!isValidDagFromRoots(dag.rootNodes)) {
    throw new Error("Cycle detected in DAG graph; cannot serialize.");
  }
  const ordered = walkNodes(dag.rootNodes);
  const ids = assignIds(ordered);
  const nodes: Record<string, unknown> = {};
  for (const node of ordered) {
    nodes[ids.get(node)!] = serializeNode(node, ids, serializeChild);
  }
  return { nodes };
}

export function dagToDict(dag: DeepAcyclicGraph): Record<string, unknown> {
  return serializeDag(dag, serializeVerdictChild);
}

export function dagToJson(dag: DeepAcyclicGraph, indent = 2): string {
  return JSON.stringify(dagToDict(dag), null, indent);
}

// -------------------------------------------------------------- deserializing

function collectReferencedIds(nodesSpec: Record<string, any>): Set<string> {
  const referenced = new Set<string>();
  for (const spec of Object.values(nodesSpec)) {
    for (const id of spec.children ?? []) referenced.add(id);
    if (
      spec.child?.type === ChildType.NODE &&
      typeof spec.child.ref === "string"
    ) {
      referenced.add(spec.child.ref);
    }
  }
  return referenced;
}

function deserializeEvalParams(
  values: unknown,
  multiturn: boolean,
): Array<SingleTurnParams | MultiTurnParams> | undefined {
  if (values == null) return undefined;
  const valid: string[] = Object.values(
    multiturn ? MultiTurnParams : SingleTurnParams,
  );
  return (values as string[]).map((value) => {
    const param = fromWireParam(value);
    if (!valid.includes(param)) {
      throw new Error(
        `Unknown evaluation param '${value}'. Expected one of: ${valid
          .map(toWireParam)
          .join(", ")}.`,
      );
    }
    return param as SingleTurnParams | MultiTurnParams;
  });
}

function nodeOptions(spec: any, multiturn: boolean): Record<string, unknown> {
  const options: Record<string, unknown> = {
    label: spec.label ?? undefined,
    evaluationParams: deserializeEvalParams(spec.evaluation_params, multiturn),
  };
  if (spec.type === NodeType.TASK) {
    options.instructions = spec.instructions;
    options.outputLabel = spec.output_label;
  } else {
    options.criteria = spec.criteria;
  }
  if (multiturn && spec.turn_window != null) {
    options.turnWindow = [spec.turn_window[0], spec.turn_window[1]];
  }
  return options;
}

function metricOptions(
  spec: Record<string, any>,
  multiturn: boolean,
): Record<string, unknown> {
  return {
    name: spec.name,
    criteria: spec.criteria,
    evaluationSteps: spec.evaluation_steps,
    evaluationParams: deserializeEvalParams(spec.evaluation_params, multiturn),
    threshold: spec.threshold,
    strictMode: spec.strict_mode,
  };
}

function buildGEval(spec: any, multiturn: boolean): ChildMetric {
  const options = metricOptions(spec, multiturn);
  if (!options.name) {
    throw new Error("A 'geval' verdict child requires a 'name'.");
  }
  if (
    !Array.isArray(options.evaluationParams) ||
    options.evaluationParams.length === 0
  ) {
    throw new Error(
      "A 'geval' verdict child requires a non-empty 'evaluation_params'.",
    );
  }
  const cls = multiturn ? ConversationalGEval : GEval;
  return new cls(options as any);
}

function buildMetric(spec: any, multiturn: boolean): ChildMetric {
  const metricClass = spec.metric_class;
  if (typeof metricClass !== "string" || metricClass.length === 0) {
    throw new Error("A 'metric' verdict child requires a 'metric_class'.");
  }
  const cls = METRIC_CLASS_REGISTRY[metricClass];
  if (cls == null) {
    throw new Error(
      `Unknown metric_class '${metricClass}'. Register it first with ` +
        "registerMetricClass() so it can be reconstructed.",
    );
  }
  return new cls(metricOptions(spec.kwargs ?? {}, multiturn));
}

interface BuildOptions {
  multiturn: boolean;
  /** How a `metric`-typed verdict child is rebuilt from its spec. */
  buildChildMetric: (spec: any, multiturn: boolean) => ChildMetric;
}

function buildNodes(
  nodesSpec: Record<string, any>,
  options: BuildOptions,
): AnyNode[] {
  if (nodesSpec == null || Object.keys(nodesSpec).length === 0) {
    throw new Error(
      "Invalid DAG document: 'nodes' must be a non-empty object.",
    );
  }
  const validTypes: string[] = Object.values(NodeType);
  for (const [id, spec] of Object.entries(nodesSpec)) {
    if (spec == null || spec.type == null) {
      throw new Error(`Node '${id}' is missing required 'type' field.`);
    }
    if (!validTypes.includes(spec.type)) {
      throw new Error(
        `Node '${id}' has unknown type '${spec.type}'. Expected one of: ${validTypes.join(", ")}.`,
      );
    }
  }

  const referenced = collectReferencedIds(nodesSpec);
  const rootIds = Object.keys(nodesSpec).filter((id) => !referenced.has(id));
  if (rootIds.length === 0) {
    throw new Error(
      "No root nodes detected (every node is referenced as a child); " +
        "graph would be empty or contain a cycle.",
    );
  }

  const classes = NODE_CLASSES[options.multiturn ? "multi" : "single"];
  const built = new Map<string, AnyNode>();

  const build = (id: string, stack: Set<string>): AnyNode => {
    const existing = built.get(id);
    if (existing) return existing;
    if (stack.has(id)) {
      throw new Error(`Cycle detected in DAG refs involving node '${id}'.`);
    }
    const spec = nodesSpec[id];
    if (spec == null) {
      throw new Error(`Reference to unknown node id '${id}'.`);
    }

    stack.add(id);
    const cls = classes[spec.type as NodeType];
    let node: AnyNode;
    if (spec.type === NodeType.VERDICT) {
      node = buildVerdict(spec, cls, stack, build, options);
    } else {
      const children = (spec.children ?? []).map((childId: string) =>
        build(childId, stack),
      );
      node = new cls(nodeOptions(spec, options.multiturn)) as ExecutableNode;
      (node as ExecutableNode).children.push(...children);
    }
    stack.delete(id);
    built.set(id, node);
    return node;
  };

  return rootIds.map((id) => build(id, new Set()));
}

function buildVerdict(
  spec: any,
  cls: any,
  stack: Set<string>,
  build: (id: string, stack: Set<string>) => AnyNode,
  options: BuildOptions,
): VerdictLikeNode {
  if (spec.score != null) {
    return new cls({ verdict: spec.verdict, score: spec.score });
  }
  const childSpec = spec.child;
  if (childSpec == null || childSpec.type == null) {
    throw new Error(
      "VerdictNode spec must have either 'score' or a 'child' object.",
    );
  }
  let child: AnyNode | ChildMetric;
  switch (childSpec.type) {
    case ChildType.NODE:
      if (typeof childSpec.ref !== "string") {
        throw new Error("VerdictNode child of type 'node' requires 'ref'.");
      }
      child = build(childSpec.ref, stack);
      break;
    case ChildType.GEVAL:
      child = buildGEval(childSpec, options.multiturn);
      break;
    case ChildType.METRIC:
      child = options.buildChildMetric(childSpec, options.multiturn);
      break;
    default:
      throw new Error(
        `VerdictNode child has unknown type '${childSpec.type}'. ` +
          `Expected one of: ${Object.values(ChildType).join(", ")}.`,
      );
  }
  return new cls({ verdict: spec.verdict, child });
}

export function dagFromDict(
  data: Record<string, any>,
  multiturn = false,
): DeepAcyclicGraph {
  if (data == null || data.nodes == null) {
    throw new Error(
      "Invalid DAG document: expected an object with a 'nodes' key.",
    );
  }
  const rootNodes = buildNodes(data.nodes, {
    multiturn,
    buildChildMetric: buildMetric,
  });
  return new DeepAcyclicGraph({ rootNodes });
}

export function dagFromJson(json: string, multiturn = false): DeepAcyclicGraph {
  return dagFromDict(JSON.parse(json), multiturn);
}

// ----------------------------------------------------- Confident AI payloads

/**
 * The upload shape: identical to {@link dagToDict} except a metric child is
 * referenced by name, since the platform stores metrics separately.
 */
export function serializeDagToPayload(
  dag: DeepAcyclicGraph,
): Record<string, unknown> {
  return serializeDag(dag, (child, ids) =>
    isNode(child)
      ? { type: ChildType.NODE, ref: ids.get(child) }
      : { type: ChildType.METRIC, metric_name: child.name },
  );
}

export function constructDagUploadPayload(
  name: string,
  dag: DeepAcyclicGraph,
  multiTurn = false,
): Record<string, unknown> {
  return {
    name,
    algorithm: "DAG",
    multiTurn,
    dag: serializeDagToPayload(dag),
  };
}

/**
 * Rebuild a DAG from a Confident AI payload. Metric children are referenced by
 * name only; the TypeScript G-Eval metrics have no `pull()`, so those must be
 * re-attached in code.
 */
export function buildDagFromPayload(
  payload: Record<string, any>,
  multiturn = false,
): DeepAcyclicGraph {
  if (payload == null || payload.nodes == null) {
    throw new Error(
      "Invalid DAG document: expected an object with a 'nodes' key.",
    );
  }
  const rootNodes = buildNodes(payload.nodes, {
    multiturn,
    buildChildMetric: (spec) => {
      throw new Error(
        `Metric child '${spec.metric_name ?? spec.metric_class}' cannot be pulled: ` +
          "resolving a metric by name is not supported in the TypeScript SDK. " +
          "Rebuild that branch in code instead.",
      );
    },
  });
  return new DeepAcyclicGraph({ rootNodes });
}
