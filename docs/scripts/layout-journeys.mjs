// Lays out each benchmark journey's graph once, so the page needs no layout
// library: positions and edge routes are stored next to the journey.
import { readFileSync, writeFileSync } from "node:fs";
import * as dagre from "dagre-d3-es/src/dagre/index.js";
import * as graphlib from "dagre-d3-es/src/graphlib/index.js";

const FILE = "lib/benchmarks/mock/voice-skills-journeys.json";
const NODE_WIDTH = 132;
const NODE_HEIGHT = 108;
const CONDITION_SIZE = 22;

const backEdges = (graph) => {
  const out = new Map();
  graph.edges.forEach((edge) => out.set(edge.from, [...(out.get(edge.from) ?? []), edge]));
  const seen = new Set();
  const onStack = new Set();
  const back = new Set();
  const visit = (id) => {
    seen.add(id);
    onStack.add(id);
    (out.get(id) ?? []).forEach((edge) => {
      if (onStack.has(edge.to)) back.add(edge.id);
      else if (!seen.has(edge.to)) visit(edge.to);
    });
    onStack.delete(id);
  };
  visit(graph.start);
  return back;
};

// Fewest caller lines from the start to an expected ending.
const mainPath = (graph) => {
  const cost = new Map(graph.nodes.map((node) => [node.id, node.lineIds.length]));
  const kind = new Map(graph.nodes.map((node) => [node.id, node.kind]));
  const best = new Map([[graph.start, { cost: cost.get(graph.start), via: null }]]);
  const queue = [graph.start];
  while (queue.length) {
    queue.sort((a, b) => best.get(a).cost - best.get(b).cost);
    const id = queue.shift();
    for (const edge of graph.edges.filter((edge) => edge.from === id)) {
      if (kind.get(edge.to) === "failure" || kind.get(edge.to) === "correction") continue;
      const next = best.get(id).cost + cost.get(edge.to);
      if (!best.has(edge.to) || next < best.get(edge.to).cost) {
        best.set(edge.to, { cost: next, via: edge });
        queue.push(edge.to);
      }
    }
  }
  const ends = graph.nodes
    .filter((node) => node.kind === "end" && best.has(node.id))
    .sort((a, b) => best.get(a.id).cost - best.get(b.id).cost);
  const edges = new Set();
  for (let via = best.get(ends[0]?.id)?.via; via; via = best.get(via.from).via) edges.add(via.id);
  return edges;
};

const journeys = JSON.parse(readFileSync(FILE, "utf8"));
for (const journey of journeys) {
  const { graph } = journey;
  const back = backEdges(graph);
  const main = mainPath(graph);
  const layout = new graphlib.Graph({ multigraph: true });
  layout.setGraph({ rankdir: "LR", nodesep: 28, ranksep: 56, edgesep: 18, marginx: 24, marginy: 24 });
  layout.setDefaultEdgeLabel(() => ({}));
  graph.nodes.forEach((node) => layout.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT }));
  graph.edges.forEach((edge) => {
    const loop = back.has(edge.id);
    layout.setEdge(
      loop ? edge.to : edge.from,
      loop ? edge.from : edge.to,
      { width: CONDITION_SIZE, height: CONDITION_SIZE, labelpos: "c", weight: main.has(edge.id) ? 4 : 1 },
      edge.id,
    );
  });
  dagre.layout(layout);

  const round = (value) => Math.round(value);
  graph.nodes.forEach((node) => {
    const placed = layout.node(node.id);
    node.x = round(placed.x);
    node.y = round(placed.y);
    node.main = graph.edges.some((edge) => main.has(edge.id) && (edge.from === node.id || edge.to === node.id));
  });
  graph.edges.forEach((edge) => {
    const loop = back.has(edge.id);
    const placed = layout.edge(loop ? edge.to : edge.from, loop ? edge.from : edge.to, edge.id);
    const points = placed.points.map((point) => [round(point.x), round(point.y)]);
    edge.points = loop ? points.reverse() : points;
    edge.label = [round(placed.x), round(placed.y)];
    edge.main = main.has(edge.id);
    edge.back = loop;
  });
  const size = layout.graph();
  graph.width = round(size.width);
  graph.height = round(size.height);
}
writeFileSync(FILE, JSON.stringify(journeys, null, 1));
console.log(journeys.length, "journeys laid out; widest", Math.max(...journeys.map((j) => j.graph.width)));
