import fs from "node:fs";
import path from "node:path";
import type { ChartData } from "@/lib/charts/schema";
import type { BoardData } from "./board";
import type { ChatData } from "./chat";
import type { FlowData } from "./flow";
import { BOARD_KIND, CHART_KIND, CHAT_KIND, FLOW_KIND, type FrameKind, SCHEMA_KIND, TRANSCRIPT_KIND } from "./kinds";
import type { SchemaCardData } from "./schema-card";
import type { TranscriptData } from "./transcript";

export interface FrameData {
  charts: Record<string, ChartData>;
  transcripts: Record<string, TranscriptData>;
  schemas: Record<string, SchemaCardData>;
  flows: Record<string, FlowData>;
  boards: Record<string, BoardData>;
  chats: Record<string, ChatData>;
}

export const EMPTY_FRAME_DATA: FrameData = { charts: {}, transcripts: {}, schemas: {}, flows: {}, boards: {}, chats: {} };

/** Frame IDs are local to the JSON file beside the content file. */
export const loadFrameFile = <D>(kind: FrameKind<D>, sourceFile: string): Record<string, D> => {
  const file = path.join(path.dirname(sourceFile), kind.file);
  if (!fs.existsSync(file)) return {};
  const raw: unknown = JSON.parse(fs.readFileSync(file, "utf8"));
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error(`${file}: expected an object keyed by ${kind.noun} IDs`);
  }
  return Object.fromEntries(
    Object.entries(raw).map(([id, value]) => {
      const { data, problems } = kind.parse(value);
      if (!data) throw new Error(`${file}: invalid ${kind.noun} "${id}": ${problems.join("; ")}`);
      return [id, data];
    })
  );
};

export const loadFrameData = (sourceFile: string): FrameData => ({
  charts: loadFrameFile(CHART_KIND, sourceFile),
  transcripts: loadFrameFile(TRANSCRIPT_KIND, sourceFile),
  schemas: loadFrameFile(SCHEMA_KIND, sourceFile),
  flows: loadFrameFile(FLOW_KIND, sourceFile),
  boards: loadFrameFile(BOARD_KIND, sourceFile),
  chats: loadFrameFile(CHAT_KIND, sourceFile),
});
