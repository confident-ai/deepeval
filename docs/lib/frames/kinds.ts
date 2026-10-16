import type { z } from "zod";
import { CHART_TYPE_NAMES, type ChartData, dataProblems, isChartType, parseChartData, typeProblems } from "@/lib/charts/schema";
import { BOARD_TYPES, type BoardData, type BoardType, boardData, boardProblems, boardTypeProblems } from "./board";
import { CHAT_TYPES, type ChatData, type ChatType, chatData, chatProblems, chatTypeProblems } from "./chat";
import { FLOW_TYPES, type FlowData, type FlowType, flowProblems, flowTypeProblems, parseFlowData } from "./flow";
import { SCHEMA_TYPES, type SchemaCardData, type SchemaType, schemaCardData, schemaProblems, schemaTypeProblems } from "./schema-card";
import {
  TRANSCRIPT_TYPES,
  type TranscriptData,
  type TranscriptType,
  transcriptData,
  transcriptProblems,
  transcriptTypeProblems,
} from "./transcript";

const formatIssues = (error: z.ZodError): string[] =>
  error.issues.map(issue => {
    const field = issue.path.join(".");
    const message =
      issue.code === "unrecognized_keys"
        ? `unknown field${issue.keys.length > 1 ? "s" : ""} ${issue.keys.map(k => `"${k}"`).join(", ")}`
        : issue.code === "invalid_type" && issue.message.endsWith("received undefined")
          ? "is required"
          : issue.message;
    return field ? `"${field}" ${message}` : message;
  });

/**
 * One entry per frame component. Each frame reads its data from its own JSON
 * file beside the content file, keyed by IDs local to that folder; the loader,
 * the validator and /dev/frames all work from this list.
 */
export interface FrameKind<D> {
  key: "charts" | "transcripts" | "schemas" | "flows" | "boards" | "chats";
  component: string;
  file: string;
  /** Singular, for messages: `chart "scores" …`. */
  noun: string;
  parse: (raw: unknown) => { data: D; problems: [] } | { data: null; problems: string[] };
  types: readonly string[];
  typeProblems: (type: string, data: D) => string[];
}

const issues = (error: z.ZodError, suffix = "") => formatIssues(error).map(issue => issue + suffix);

const zodKind = <D>(schema: z.ZodType<D>, check: (data: D) => string[]) => (raw: unknown) => {
  const result = schema.safeParse(raw);
  if (!result.success) return { data: null, problems: issues(result.error) };
  const problems = check(result.data);
  return problems.length ? { data: null, problems } : { data: result.data, problems: [] as [] };
};

export const CHART_KIND: FrameKind<ChartData> = {
  key: "charts",
  component: "ChartFrame",
  file: "charts.json",
  noun: "chart",
  parse: raw => {
    const { shape, result } = parseChartData(raw);
    if (!result.success) return { data: null, problems: issues(result.error, ` (read as ${shape} data)`) };
    const chart = { shape, data: result.data } as ChartData;
    const problems = dataProblems(chart);
    return problems.length ? { data: null, problems } : { data: chart, problems: [] };
  },
  types: CHART_TYPE_NAMES,
  typeProblems: (type, chart) => (isChartType(type) ? typeProblems(type, chart) : []),
};

export const TRANSCRIPT_KIND: FrameKind<TranscriptData> = {
  key: "transcripts",
  component: "TranscriptFrame",
  file: "transcripts.json",
  noun: "transcript",
  parse: zodKind(transcriptData, transcriptProblems),
  types: TRANSCRIPT_TYPES,
  typeProblems: (type, data) => transcriptTypeProblems(type as TranscriptType, data),
};

export const SCHEMA_KIND: FrameKind<SchemaCardData> = {
  key: "schemas",
  component: "SchemaFrame",
  file: "schemas.json",
  noun: "schema",
  parse: zodKind(schemaCardData, schemaProblems),
  types: SCHEMA_TYPES,
  typeProblems: (type, data) => schemaTypeProblems(type as SchemaType, data),
};

export const FLOW_KIND: FrameKind<FlowData> = {
  key: "flows",
  component: "FlowFrame",
  file: "flows.json",
  noun: "flow",
  parse: raw => {
    const { shape, result } = parseFlowData(raw);
    if (!result.success) return { data: null, problems: issues(result.error, ` (read as ${shape} data)`) };
    const flow = { shape, data: result.data } as FlowData;
    const problems = flowProblems(flow);
    return problems.length ? { data: null, problems } : { data: flow, problems: [] };
  },
  types: FLOW_TYPES,
  typeProblems: (type, data) => flowTypeProblems(type as FlowType, data),
};

export const BOARD_KIND: FrameKind<BoardData> = {
  key: "boards",
  component: "BoardFrame",
  file: "boards.json",
  noun: "board",
  parse: zodKind(boardData, boardProblems),
  types: BOARD_TYPES,
  typeProblems: (type, data) => boardTypeProblems(type as BoardType, data),
};

export const CHAT_KIND: FrameKind<ChatData> = {
  key: "chats",
  component: "ChatFrame",
  file: "chats.json",
  noun: "chat",
  parse: zodKind(chatData, chatProblems),
  types: CHAT_TYPES,
  typeProblems: (type, data) => chatTypeProblems(type as ChatType, data),
};

export const FRAME_KINDS = [CHART_KIND, TRANSCRIPT_KIND, SCHEMA_KIND, FLOW_KIND, BOARD_KIND, CHAT_KIND] as const;
export const FRAME_FILES: readonly string[] = FRAME_KINDS.map(k => k.file);

export const FRAME_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
