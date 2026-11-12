"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CornerDownLeft,
  Flag,
  ListChecks,
  Maximize,
  Minus,
  PhoneOff,
  Play,
  Plus,
  User,
  X,
  Zap,
} from "lucide-react";
import type {
  JourneyGraph as Graph,
  JourneyGraphEdge,
  JourneyLine,
  JourneyNodeKind,
} from "@/lib/benchmarks/types";
import ClipPlayer from "./ClipPlayer";
import styles from "./JourneyGraph.module.scss";

type JourneyGraphProps = {
  graph: Graph;
  lines: JourneyLine[];
};

type View = { x: number; y: number; scale: number };

const TILE = 56;
const MIN_SCALE = 0.3;
const MAX_SCALE = 1.6;
const GRAPH_SHARE = 0.62;

const ICONS: Record<JourneyNodeKind, React.ReactNode> = {
  start: <Play />,
  state: <User />,
  group: <ListChecks />,
  correction: <CornerDownLeft />,
  end: <Flag />,
  failure: <PhoneOff />,
};

const LEGEND: { kind: JourneyNodeKind; label: string }[] = [
  { kind: "start", label: "Start" },
  { kind: "state", label: "Line" },
  { kind: "group", label: "Any order" },
  { kind: "correction", label: "Correction" },
  { kind: "end", label: "Good ending" },
  { kind: "failure", label: "Agent failed" },
];

// A smooth line through the route the layout gave the edge.
const pathThrough = (points: number[][]) => {
  const [first, ...rest] = points;
  let path = `M ${first[0]} ${first[1]}`;
  for (let index = 0; index < rest.length - 1; index++) {
    const [x, y] = rest[index];
    const [nextX, nextY] = rest[index + 1];
    path += ` Q ${x} ${y} ${(x + nextX) / 2} ${(y + nextY) / 2}`;
  }
  const last = rest[rest.length - 1];
  return `${path} L ${last[0]} ${last[1]}`;
};

const clamp = (scale: number) =>
  Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));

const JourneyGraph: React.FC<JourneyGraphProps> = ({ graph, lines }) => {
  const [selectedId, setSelectedId] = useState<string | null>(graph.start);
  const [edgeId, setEdgeId] = useState<string | null>(null);
  const selected = graph.nodes.find((node) => node.id === selectedId);
  const condition = graph.edges.find((edge) => edge.id === edgeId);
  const labelOf = (id: string) =>
    graph.nodes.find((node) => node.id === id)?.label ?? id;
  const said = (selected?.lineIds ?? []).flatMap(
    (id) => lines.find((line) => line.id === id) ?? [],
  );
  const canvas = useRef<HTMLDivElement>(null);
  const drag = useRef<{
    x: number;
    y: number;
    moved: boolean;
    nodeId?: string;
    edgeId?: string;
  } | null>(null);
  const [view, setView] = useState<View>({ x: 0, y: 0, scale: 1 });
  const [hoveredEdge, setHoveredEdge] = useState<JourneyGraphEdge | null>(
    null,
  );

  // Opens on the start of the call at a readable size, not the whole graph.
  const reset = useCallback(() => {
    const box = canvas.current?.getBoundingClientRect();
    if (!box) return;
    // The graph sits in the upper part of the canvas, which leaves the lower
    // part free for the card that opens on a click.
    const room = box.height * GRAPH_SHARE;
    // Never so small that the labels stop being readable.
    const scale = Math.max(0.8, Math.min(1, room / graph.height));
    const start = graph.nodes.find((node) => node.id === graph.start);
    setView({
      scale,
      x: 32 - ((start?.x ?? 0) - 66) * scale,
      y: Math.max(16, (room - graph.height * scale) / 2),
    });
  }, [graph]);

  const fit = useCallback(() => {
    const box = canvas.current?.getBoundingClientRect();
    if (!box) return;
    const room = box.height * GRAPH_SHARE;
    const scale = clamp(Math.min(box.width / graph.width, room / graph.height));
    setView({
      scale,
      x: (box.width - graph.width * scale) / 2,
      y: Math.max(16, (room - graph.height * scale) / 2),
    });
  }, [graph]);

  useEffect(reset, [reset]);

  const zoomAt = useCallback((factor: number, clientX?: number, clientY?: number) => {
    const box = canvas.current?.getBoundingClientRect();
    if (!box) return;
    const originX = (clientX ?? box.left + box.width / 2) - box.left;
    const originY = (clientY ?? box.top + box.height / 2) - box.top;
    setView((current) => {
      const scale = clamp(current.scale * factor);
      const ratio = scale / current.scale;
      return {
        scale,
        x: originX - (originX - current.x) * ratio,
        y: originY - (originY - current.y) * ratio,
      };
    });
  }, []);

  // Two fingers or the wheel scroll the graph; pinch or ctrl + wheel zooms.
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      if (event.ctrlKey || event.metaKey) {
        zoomAt(Math.exp(-event.deltaY * 0.01), event.clientX, event.clientY);
        return;
      }
      setView((current) => ({
        ...current,
        x: current.x - event.deltaX,
        y: current.y - event.deltaY,
      }));
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  const onPointerDown = (event: React.PointerEvent) => {
    // Read now: once the pointer is captured, later events no longer say
    // which tile they started on.
    const target = event.target as HTMLElement;
    drag.current = {
      x: event.clientX,
      y: event.clientY,
      moved: false,
      nodeId: target.closest<HTMLElement>("[data-node]")?.dataset.node,
      edgeId: target.closest<HTMLElement>("[data-edge]")?.dataset.edge,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const from = drag.current;
    if (!from) return;
    const dx = event.clientX - from.x;
    const dy = event.clientY - from.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) from.moved = true;
    drag.current = { ...from, x: event.clientX, y: event.clientY };
    setView((current) => ({ ...current, x: current.x + dx, y: current.y + dy }));
  };

  const onPointerUp = () => {
    const from = drag.current;
    drag.current = null;
    if (!from || from.moved) return;
    if (from.nodeId) {
      setSelectedId(from.nodeId);
      setEdgeId(null);
    } else if (from.edgeId) {
      setEdgeId(from.edgeId);
      setSelectedId(null);
    }
  };

  const isLit = (edge: JourneyGraphEdge) =>
    edge.from === selectedId ||
    edge.id === edgeId ||
    edge.id === hoveredEdge?.id;

  return (
    <div className={styles.frame}>
      <ul className={styles.legend}>
        {LEGEND.map((entry) => (
          <li key={entry.kind} data-kind={entry.kind}>
            <span className={styles.swatch}>{ICONS[entry.kind]}</span>
            {entry.label}
          </li>
        ))}
        <li>
          <span className={styles.route} />
          Best route
        </li>
        <li data-kind="happens">
          <span className={styles.swatch}>
            <Zap />
          </span>
          Noise, pause or aside
        </li>
        <li>
          <span className={styles.dot} />
          Condition, click to read
        </li>
      </ul>
      <div className={styles.body}>
      <div
        ref={canvas}
        className={styles.canvas}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => (drag.current = null)}
      >
        <div
          className={styles.stage}
          style={{
            width: graph.width,
            height: graph.height,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
          }}
        >
          <svg
            className={styles.edges}
            width={graph.width}
            height={graph.height}
            aria-hidden
          >
            <defs>
              <marker
                id="journey-arrow"
                viewBox="0 0 8 8"
                refX="7"
                refY="4"
                markerWidth="7"
                markerHeight="7"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 8 4 L 0 8 z" fill="context-stroke" />
              </marker>
            </defs>
            {graph.edges.map((edge) => (
              <path
                key={edge.id}
                d={pathThrough(edge.points)}
                className={styles.edge}
                data-main={edge.main}
                data-lit={isLit(edge)}
                markerEnd="url(#journey-arrow)"
              />
            ))}
          </svg>

          {graph.edges.map((edge) => (
            <span
              key={edge.id}
              className={styles.condition}
              data-edge={edge.id}
              data-main={edge.main}
              data-lit={isLit(edge)}
              style={{ left: edge.label[0], top: edge.label[1] }}
              onPointerEnter={() => setHoveredEdge(edge)}
              onPointerLeave={() => setHoveredEdge(null)}
            />
          ))}

          {graph.nodes.map((node) => (
            <button
              key={node.id}
              type="button"
              data-node={node.id}
              data-kind={node.kind}
              data-main={node.main}
              aria-pressed={node.id === selectedId}
              className={styles.node}
              style={{ left: node.x, top: node.y - 22 }}
            >
              <span className={styles.tile} style={{ width: TILE, height: TILE }}>
                {ICONS[node.kind]}
                {node.lineIds.length > 1 && (
                  <span className={styles.count}>{node.lineIds.length}</span>
                )}
                {node.events > 0 && (
                  <span className={styles.happens} aria-label="Something else happens here">
                    <Zap />
                  </span>
                )}
              </span>
              <span className={styles.label}>{node.label}</span>
            </button>
          ))}

          {hoveredEdge && hoveredEdge.id !== edgeId && (
            <span
              className={styles.tooltip}
              style={{ left: hoveredEdge.label[0], top: hoveredEdge.label[1] }}
            >
              The agent {hoveredEdge.when}
            </span>
          )}
        </div>
      </div>

      {condition && (
        <div className={styles.card}>
          <div className={styles.cardHead}>
            <span>
              {labelOf(condition.from)} → {labelOf(condition.to)}
              <small>The caller goes this way when</small>
            </span>
            <button
              type="button"
              onClick={() => setEdgeId(null)}
              aria-label="Close"
            >
              <X />
            </button>
          </div>
          <p className={styles.when}>The agent {condition.when}.</p>
        </div>
      )}

      {selected && (
        <div className={styles.card}>
          <div className={styles.cardHead}>
            <span>
              {selected.label}
              <small>What the caller says</small>
            </span>
            <button
              type="button"
              onClick={() => setSelectedId(null)}
              aria-label="Close"
            >
              <X />
            </button>
          </div>
          {said.map((line) => (
            <div key={line.id} className={styles.said}>
              <ClipPlayer src={line.audio} label={line.say} />
              <p>{line.say}</p>
              {line.events.length > 0 && (
                <ul className={styles.events}>
                  {line.events.map((event, index) => (
                    <li key={index}>
                      <Zap />
                      <span>{event.label}</span>
                      {event.audio && (
                        <ClipPlayer src={event.audio} label={event.label} />
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}

      <div className={styles.controls}>
        <button type="button" onClick={() => zoomAt(1.25)} aria-label="Zoom in">
          <Plus />
        </button>
        <button type="button" onClick={() => zoomAt(0.8)} aria-label="Zoom out">
          <Minus />
        </button>
        <button type="button" onClick={fit} aria-label="Fit the whole graph">
          <Maximize />
        </button>
      </div>
      </div>
    </div>
  );
};

export default JourneyGraph;
