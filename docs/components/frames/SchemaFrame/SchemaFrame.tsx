import type * as React from "react";
import FlowPlot from "@/components/frames/FlowPlot/FlowPlot";
import { MIN_SCALE } from "@/components/frames/FlowPlot/_scene";
import FrameShell from "@/components/frames/FrameShell/FrameShell";
import { SCHEMA_KIND } from "@/lib/frames/kinds";
import type { SchemaCardData, SchemaType } from "@/lib/frames/schema-card";
import { layoutSchema } from "./_layout";

interface SchemaFrameProps {
  id: string;
  schema?: SchemaCardData;

  type: string;
  title: string;

  description: string;
  kicker?: string;
  subtitle?: string;
  note?: string;
  caption?: string;
}

const SchemaFrame: React.FC<SchemaFrameProps> = ({ id, schema, type, title, description, kicker, subtitle, note, caption }) => {
  const problems = !schema
    ? ["unknown or invalid data"]
    : !SCHEMA_KIND.types.includes(type)
      ? [`unknown type "${type}"`]
      : SCHEMA_KIND.typeProblems(type, schema);
  if (!schema || problems.length) {
    console.warn(`[SchemaFrame] Can't draw "${id}" as "${type}": ${problems.join("; ")}. Run yarn check:content for details.`);
    return null;
  }
  return (
    <FrameShell
      data={{ "data-schema": id, "data-schema-type": type }}
      title={title}
      description={description}
      kicker={kicker}
      subtitle={subtitle}
      caption={caption}
      note={note}
    >
      <FlowPlot scene={layoutSchema(type as SchemaType, schema, "wide")} id={`schema-${id}`} variant="wide" minScale={MIN_SCALE.wide} />
      <FlowPlot scene={layoutSchema(type as SchemaType, schema, "narrow")} id={`schema-${id}`} variant="narrow" minScale={MIN_SCALE.narrow} />
    </FrameShell>
  );
};

/** Bind MDX to this article's schemas.json; authors only supply the local ID. */
export const createSchemaFrame = (data: Record<string, SchemaCardData>) =>
  function ArticleSchemaFrame(props: Omit<SchemaFrameProps, "schema">) {
    return <SchemaFrame {...props} schema={Object.hasOwn(data, props.id) ? data[props.id] : undefined} />;
  };

export default SchemaFrame;
