import dagre from "@dagrejs/dagre";
import { type GraphData, type GraphNode, nodeColor, nodeIcon, type SequenceData } from "@/lib/frames/flow";
import {
  type Box,
  boundary,
  type FlowScene,
  GROUP_HEAD,
  GROUP_PAD,
  groupHeaderWidth,
  iconRoom,
  LINE,
  normalize,
  PAD_X,
  PAD_Y,
  type Point,
  type SceneEdge,
  type SceneGroup,
  type SceneNode,
  SMALL,
  SMALL_LINE,
  smoothPath,
  textWidth,
  wrapWords,
} from "@/components/frames/FlowPlot/_scene";

// FlowFrame's layouts: dagre for flows and trees, columns and rows for sequences.

const nodeSize = (node: GraphNode, wrap: number) => {
  const lines = wrapWords(node.label, wrap);
  const text = Math.max(...lines.map(l => textWidth(l))) + iconRoom(nodeIcon(node));
  const tall = lines.length * LINE;
  if (node.shape === "circle") {
    const d = Math.max(40, text + 20, tall + 16);
    return { lines, w: d, h: d };
  }
  if (node.shape === "diamond") return { lines, w: text * 1.5 + 32, h: tall * 1.6 + 28 };
  if (node.shape === "store") return { lines, w: text + PAD_X * 2, h: tall + PAD_Y * 2 + 10 };
  if (node.shape === "pill") return { lines, w: text + PAD_X * 2 + 10, h: tall + PAD_Y * 2 };
  return { lines, w: Math.max(56, text + PAD_X * 2), h: tall + PAD_Y * 2 };
};

export const layoutGraph = (data: GraphData, direction: "TB" | "LR", wrap: number): FlowScene => {
  const g = new dagre.graphlib.Graph({ compound: true, multigraph: true });
  // Groups need room for their header row: above them in TB, beside them in LR.
  const grouped = data.groups.length > 0;
  g.setGraph({
    rankdir: direction,
    nodesep: direction === "LR" ? (grouped ? GROUP_HEAD + 24 : 24) : 32,
    ranksep: direction === "TB" && grouped ? GROUP_HEAD + 44 : 44,
    edgesep: 14,
    marginx: 0,
    marginy: 0,
  });
  g.setDefaultEdgeLabel(() => ({}));

  const sizes = new Map(data.nodes.map(n => [n.id, nodeSize(n, wrap)]));
  for (const n of data.nodes) {
    const { w, h } = sizes.get(n.id)!;
    g.setNode(n.id, { width: w, height: h });
  }
  for (const group of data.groups) g.setNode(group.id, {});
  for (const group of data.groups) if (group.parent) g.setParent(group.id, group.parent);
  for (const n of data.nodes) if (n.group) g.setParent(n.id, n.group);

  const edgeLabels = data.edges.map(e => (e.label ? wrapWords(e.label, 120, SMALL) : undefined));
  data.edges.forEach((e, i) => {
    const lines = edgeLabels[i];
    const label = lines ? { width: Math.max(...lines.map(l => textWidth(l, SMALL))) + 8, height: lines.length * SMALL_LINE + 4, labelpos: "c" } : {};
    g.setEdge(e.from, e.to, label, `e${i}`);
  });

  dagre.layout(g);

  const nodes: SceneNode[] = data.nodes.map(n => {
    const laid = g.node(n.id);
    const { w, h, lines } = sizes.get(n.id)!;
    return { id: n.id, x: laid.x - w / 2, y: laid.y - h / 2, w, h, lines, color: nodeColor(n), shape: n.shape, icon: nodeIcon(n) };
  });
  const byId = new Map(nodes.map(n => [n.id, n]));

  // Group boxes are measured from their members rather than taken from dagre,
  // so each gets a header row for its label and badge. Inner groups first.
  const depth = (id: string): number => {
    const parent = data.groups.find(x => x.id === id)?.parent;
    return parent ? 1 + depth(parent) : 0;
  };
  const boxes = new Map<string, Box>();
  for (const group of [...data.groups].sort((a, b) => depth(b.id) - depth(a.id))) {
    const members: Box[] = [
      ...nodes.filter(n => data.nodes.find(x => x.id === n.id)?.group === group.id),
      ...data.groups.filter(x => x.parent === group.id).map(x => boxes.get(x.id)!),
    ];
    const left = Math.min(...members.map(b => b.x)) - GROUP_PAD;
    const right = Math.max(...members.map(b => b.x + b.w)) + GROUP_PAD;
    const top = Math.min(...members.map(b => b.y)) - GROUP_HEAD - 6;
    const bottom = Math.max(...members.map(b => b.y + b.h)) + GROUP_PAD;
    const minW = groupHeaderWidth(group.label, group.badge ? textWidth(group.badge, SMALL) + 14 : 0);
    const grow = Math.max(0, minW - (right - left)) / 2;
    boxes.set(group.id, { x: left - grow, y: top, w: right - left + grow * 2, h: bottom - top });
  }
  const groups: SceneGroup[] = data.groups.map(group => ({
    id: group.id,
    label: group.label,
    badge: group.badge ? { text: group.badge, w: textWidth(group.badge, SMALL) + 14 } : undefined,
    ...boxes.get(group.id)!,
  }));

  const edges: SceneEdge[] = data.edges.map((e, i) => {
    const laid = g.edge({ v: e.from, w: e.to, name: `e${i}` });
    const pts: Point[] = laid.points.map((p: Point) => ({ x: p.x, y: p.y }));
    const from = byId.get(e.from)!;
    const to = byId.get(e.to)!;
    pts[0] = boundary(from, pts[1] ?? { x: to.x + to.w / 2, y: to.y + to.h / 2 });
    pts[pts.length - 1] = boundary(to, pts[pts.length - 2] ?? { x: from.x + from.w / 2, y: from.y + from.h / 2 });
    const lines = edgeLabels[i];
    return {
      d: smoothPath(pts),
      dashed: e.dashed,
      both: e.both,
      label: lines && laid.x !== undefined ? { x: laid.x, y: laid.y, lines } : undefined,
    };
  });

  return normalize({ width: 0, height: 0, nodes, groups, edges, lifelines: [] });
};

const SEQ_GAP = 32;

/** Participants as columns, messages as numbered arrows down the page. */
export const layoutSequence = (data: SequenceData, wrap: number): FlowScene => {
  const index = new Map(data.participants.map((p, i) => [p.id, i]));
  const heads = data.participants.map(p => {
    const lines = wrapWords(p.label, Math.max(wrap, 96));
    const icon = nodeIcon(p);
    return { lines, icon, w: Math.max(88, Math.max(...lines.map(l => textWidth(l))) + iconRoom(icon) + PAD_X * 2) };
  });
  const headH = Math.max(...heads.map(h => h.lines.length)) * LINE + PAD_Y * 2;

  const messages = data.messages.map((m, i) => ({
    ...m,
    a: index.get(m.from)!,
    b: index.get(m.to)!,
    lines: wrapWords(`${i + 1}. ${m.label}`, wrap, SMALL),
  }));
  const labelW = (lines: string[]) => Math.max(...lines.map(l => textWidth(l, SMALL)));

  // Column gaps: room for both heads, then widened until each label fits over its span.
  const gaps = heads.slice(1).map((h, i) => (heads[i].w + h.w) / 2 + SEQ_GAP);
  const sorted = [...messages].filter(m => m.a !== m.b).sort((p, q) => Math.abs(p.b - p.a) - Math.abs(q.b - q.a));
  for (const m of sorted) {
    const lo = Math.min(m.a, m.b);
    const hi = Math.max(m.a, m.b);
    const span = gaps.slice(lo, hi).reduce((s, g) => s + g, 0);
    const need = labelW(m.lines) + 24;
    if (span < need) for (let k = lo; k < hi; k++) gaps[k] += (need - span) / (hi - lo);
  }
  const xs = [heads[0].w / 2];
  for (const g of gaps) xs.push(xs.at(-1)! + g);

  const nodes: SceneNode[] = data.participants.map((p, i) => ({
    id: p.id,
    x: xs[i] - heads[i].w / 2,
    y: 0,
    w: heads[i].w,
    h: headH,
    lines: heads[i].lines,
    color: nodeColor(p),
    shape: "box",
    icon: heads[i].icon,
  }));

  const edges: SceneEdge[] = [];
  let y = headH + 18;
  for (const m of messages) {
    const textH = m.lines.length * SMALL_LINE;
    const x1 = xs[m.a];
    const x2 = xs[m.b];
    if (m.a === m.b) {
      const top = y + textH + 2;
      edges.push({
        d: `M${x1},${top}L${x1 + 28},${top}L${x1 + 28},${top + 16}L${x1 + 2},${top + 16}`,
        dashed: m.dashed,
        both: false,
        label: { x: x1 + 8 + labelW(m.lines) / 2, y: y + textH / 2, lines: m.lines },
      });
      y = top + 16 + 20;
    } else {
      const arrowY = y + textH + 6;
      const dir = Math.sign(x2 - x1);
      edges.push({
        d: `M${x1 + dir * 2},${arrowY}L${x2 - dir * 2},${arrowY}`,
        dashed: m.dashed,
        both: false,
        label: { x: (x1 + x2) / 2, y: y + textH / 2, lines: m.lines },
      });
      y = arrowY + 18;
    }
  }
  const lifelines = xs.map(x => ({ x, y1: headH, y2: y - 4 }));
  return normalize({ width: 0, height: 0, nodes, groups: [], edges, lifelines });
};

