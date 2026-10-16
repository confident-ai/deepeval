import type * as React from "react";
import FlowPlot from "@/components/frames/FlowPlot/FlowPlot";
import { type FlowScene, MIN_SCALE } from "@/components/frames/FlowPlot/_scene";
import FrameShell from "@/components/frames/FrameShell/FrameShell";
import type { FlowData } from "@/lib/frames/flow";
import { FLOW_KIND } from "@/lib/frames/kinds";
import { layoutGraph, layoutSequence } from "./_layout";

interface FlowFrameProps {
  id: string;
  flow?: FlowData;

  type: string;
  title: string;

  description: string;
  kicker?: string;
  subtitle?: string;
  note?: string;
  caption?: string;
}

// Label wrap widths for the wide (desktop) and narrow (phone) drawings.
const WIDE_WRAP = 150;
const NARROW_WRAP = 110;
// The wide drawing's budget: the chart canvas width. A left-to-right flow wider
// than this is laid out top to bottom instead.
const WIDE_MAX = 640;

const scenes = (type: string, flow: FlowData): [FlowScene, FlowScene] => {
  if (flow.shape === "sequence") return [layoutSequence(flow.data, WIDE_WRAP + 40), layoutSequence(flow.data, NARROW_WRAP)];
  const authored = layoutGraph(flow.data, flow.data.direction, WIDE_WRAP);
  const tall = flow.data.direction === "LR" && authored.width > WIDE_MAX ? layoutGraph(flow.data, "TB", WIDE_WRAP) : null;
  const wide = tall && tall.width < authored.width ? tall : authored;
  return [wide, layoutGraph(flow.data, "TB", NARROW_WRAP)];
};

const FlowFrame: React.FC<FlowFrameProps> = ({ id, flow, type, title, description, kicker, subtitle, note, caption }) => {
  const problems = !flow
    ? ["unknown or invalid data"]
    : !FLOW_KIND.types.includes(type)
      ? [`unknown type "${type}"`]
      : FLOW_KIND.typeProblems(type, flow);
  if (!flow || problems.length) {
    console.warn(`[FlowFrame] Can't draw "${id}" as "${type}": ${problems.join("; ")}. Run yarn check:content for details.`);
    return null;
  }
  const [wide, narrow] = scenes(type, flow);
  return (
    <FrameShell
      data={{ "data-flow": id, "data-flow-type": type }}
      title={title}
      description={description}
      kicker={kicker}
      subtitle={subtitle}
      caption={caption}
      note={note}
    >
      <FlowPlot scene={wide} id={id} variant="wide" minScale={MIN_SCALE.wide} />
      <FlowPlot scene={narrow} id={id} variant="narrow" minScale={MIN_SCALE.narrow} />
    </FrameShell>
  );
};

/** Bind MDX to this article's flows.json; authors only supply the local ID. */
export const createFlowFrame = (data: Record<string, FlowData>) =>
  function ArticleFlowFrame(props: Omit<FlowFrameProps, "flow">) {
    return <FlowFrame {...props} flow={Object.hasOwn(data, props.id) ? data[props.id] : undefined} />;
  };

export default FlowFrame;
