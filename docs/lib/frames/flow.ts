import { z } from "zod";
import { type ChartColor, chartColor } from "@/lib/charts/schema";

const nonEmpty = z.string().trim().min(1);
const localId = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, "use lowercase-kebab-case ids");

export const NODE_KINDS = ["input", "llm", "tool", "store", "eval", "output", "decision", "actor"] as const;
export type NodeKind = (typeof NODE_KINDS)[number];

/** A node's kind picks its color unless the node names one. */
export const KIND_COLORS: Record<NodeKind, ChartColor> = {
  input: "blue",
  llm: "violet",
  tool: "ember",
  store: "teal",
  eval: "pink",
  output: "blue",
  decision: "yellow",
  actor: "muted",
};

/**
 * Lucide icons a node may show, by name (lucide.dev/icons). A short, deliberate
 * list keeps diagrams in one visual language; add a name here (and its import
 * in components/frames/FlowPlot/_icons.ts) when a diagram needs one.
 */
export const FLOW_ICONS = [
  "activity",
  "app-window",
  "bell",
  "binary",
  "book-open",
  "bot",
  "boxes",
  "brain",
  "bug",
  "calendar",
  "clipboard-check",
  "code",
  "cpu",
  "database",
  "file-search",
  "file-text",
  "filter",
  "gauge",
  "gavel",
  "globe",
  "hard-drive",
  "image",
  "inbox",
  "key",
  "layers",
  "layout-template",
  "library",
  "list-checks",
  "lock",
  "mail",
  "message-square",
  "message-square-reply",
  "network",
  "plug",
  "scale",
  "search",
  "send",
  "server",
  "shield-alert",
  "shield-check",
  "sparkles",
  "table",
  "target",
  "terminal",
  "text-cursor-input",
  "user",
  "users",
  "workflow",
  "wrench",
  "zap",
] as const;
export type FlowIcon = (typeof FLOW_ICONS)[number];

/** A node's kind picks its icon unless the node names one, or "none". */
export const KIND_ICONS: Partial<Record<NodeKind, FlowIcon>> = {
  input: "message-square",
  llm: "brain",
  tool: "wrench",
  store: "database",
  eval: "gauge",
  output: "message-square-reply",
  actor: "user",
};

const icon = z.union([z.enum(FLOW_ICONS), z.literal("none")]);

export const NODE_SHAPES = ["box", "pill", "diamond", "circle", "store"] as const;
export type NodeShape = (typeof NODE_SHAPES)[number];
/** Shapes that hold a question or a score, not a component: drawn without icons. */
const TEXT_SHAPES: readonly NodeShape[] = ["diamond", "circle"];

export const graphData = z.strictObject({
  direction: z.enum(["TB", "LR"]).default("TB"),
  nodes: z
    .array(
      z.strictObject({
        id: localId,
        label: nonEmpty,
        kind: z.enum(NODE_KINDS).optional(),
        color: chartColor.optional(),
        icon: icon.optional(),
        shape: z.enum(NODE_SHAPES).default("box"),
        group: localId.optional(),
      })
    )
    .min(1),
  groups: z
    .array(z.strictObject({ id: localId, label: nonEmpty, badge: nonEmpty.optional(), parent: localId.optional() }))
    .default([]),
  edges: z
    .array(
      z.strictObject({
        from: localId,
        to: localId,
        label: nonEmpty.optional(),
        dashed: z.boolean().default(false),
        both: z.boolean().default(false),
      })
    )
    .default([]),
});

export const sequenceData = z.strictObject({
  participants: z
    .array(
      z.strictObject({
        id: localId,
        label: nonEmpty,
        kind: z.enum(NODE_KINDS).optional(),
        color: chartColor.optional(),
        icon: icon.optional(),
      })
    )
    .min(2),
  messages: z
    .array(z.strictObject({ from: localId, to: localId, label: nonEmpty, dashed: z.boolean().default(false) }))
    .min(1),
});

export type GraphData = z.infer<typeof graphData>;
export type GraphNode = GraphData["nodes"][number];
export type SequenceData = z.infer<typeof sequenceData>;
export type FlowData = { shape: "graph"; data: GraphData } | { shape: "sequence"; data: SequenceData };

export const nodeColor = (n: { kind?: NodeKind; color?: ChartColor }): ChartColor =>
  n.color ?? (n.kind ? KIND_COLORS[n.kind] : "muted");

/** The icon a node or participant shows, if any. */
export const nodeIcon = (n: { kind?: NodeKind; icon?: FlowIcon | "none"; shape?: NodeShape }): FlowIcon | undefined => {
  if (n.icon === "none" || (n.shape && TEXT_SHAPES.includes(n.shape))) return undefined;
  return n.icon ?? (n.kind ? KIND_ICONS[n.kind] : undefined);
};

export const parseFlowData = (raw: unknown) =>
  typeof raw === "object" && raw !== null && "participants" in raw
    ? ({ shape: "sequence", result: sequenceData.safeParse(raw) } as const)
    : ({ shape: "graph", result: graphData.safeParse(raw) } as const);

export const FLOW_TYPES = ["flow", "tree", "sequence"] as const;
export type FlowType = (typeof FLOW_TYPES)[number];

const duplicates = (ids: string[]) => ids.filter((id, i) => ids.indexOf(id) !== i);

export const flowProblems = (flow: FlowData): string[] => {
  const problems: string[] = [];
  if (flow.shape === "sequence") {
    const ids = flow.data.participants.map(p => p.id);
    for (const id of new Set(duplicates(ids))) problems.push(`participant id "${id}" is used twice`);
    flow.data.messages.forEach((m, i) => {
      for (const end of ["from", "to"] as const) {
        if (!ids.includes(m[end])) problems.push(`"messages.${i}.${end}" names unknown participant "${m[end]}"`);
      }
    });
    return problems;
  }
  const { nodes, groups, edges } = flow.data;
  const nodeIds = nodes.map(n => n.id);
  const groupIds = groups.map(g => g.id);
  for (const id of new Set(duplicates([...nodeIds, ...groupIds]))) problems.push(`id "${id}" is used twice`);
  nodes.forEach((n, i) => {
    if (n.group && !groupIds.includes(n.group)) problems.push(`"nodes.${i}.group" names unknown group "${n.group}"`);
    if (n.icon && n.icon !== "none" && TEXT_SHAPES.includes(n.shape)) {
      problems.push(`"nodes.${i}" is a ${n.shape}, which holds text and draws no icon; remove "icon"`);
    }
  });
  groups.forEach((g, i) => {
    if (g.parent && !groupIds.includes(g.parent)) problems.push(`"groups.${i}.parent" names unknown group "${g.parent}"`);
    const seen = new Set([g.id]);
    for (let p = g.parent; p; p = groups.find(x => x.id === p)?.parent) {
      if (seen.has(p)) {
        problems.push(`group "${g.id}" is nested inside itself`);
        break;
      }
      seen.add(p);
    }
    if (!nodes.some(n => n.group === g.id) && !groups.some(x => x.parent === g.id)) problems.push(`group "${g.id}" is empty`);
  });
  edges.forEach((e, i) => {
    for (const end of ["from", "to"] as const) {
      if (!nodeIds.includes(e[end])) problems.push(`"edges.${i}.${end}" names unknown node "${e[end]}"`);
    }
    if (e.from === e.to) problems.push(`"edges.${i}" connects "${e.from}" to itself`);
  });
  return problems;
};

export const flowTypeProblems = (type: FlowType, flow: FlowData): string[] => {
  if (type === "sequence") return flow.shape === "sequence" ? [] : ['sequence needs "participants" and "messages"'];
  if (flow.shape !== "graph") return [`${type} needs "nodes" and "edges", this data has "participants"`];
  if (type === "flow") return [];
  const { nodes, edges } = flow.data;
  const incoming = new Map(nodes.map(n => [n.id, edges.filter(e => e.to === n.id).length]));
  const roots = nodes.filter(n => incoming.get(n.id) === 0);
  const problems = roots.length === 1 ? [] : [`tree needs exactly one root (a node with no incoming edge), this data has ${roots.length}`];
  for (const [id, count] of incoming) if (count > 1) problems.push(`tree nodes have one parent, "${id}" has ${count}`);
  if (edges.some(e => e.both)) problems.push("tree edges point one way; remove \"both\"");
  return problems;
};
