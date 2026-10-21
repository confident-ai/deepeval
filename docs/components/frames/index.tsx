import { createBoardFrame } from "@/components/frames/BoardFrame/BoardFrame";
import { createChartFrame } from "@/components/frames/ChartFrame/ChartFrame";
import { createChatFrame } from "@/components/frames/ChatFrame/ChatFrame";
import { createFlowFrame } from "@/components/frames/FlowFrame/FlowFrame";
import { createSchemaFrame } from "@/components/frames/SchemaFrame/SchemaFrame";
import { createTranscriptFrame } from "@/components/frames/TranscriptFrame/TranscriptFrame";
import { EMPTY_FRAME_DATA, type FrameData } from "@/lib/frames/store";

/** Every frame component, bound to one content file's frame data (loadFrameData). */
export const createFrameComponents = (data: FrameData = EMPTY_FRAME_DATA) => ({
  ChartFrame: createChartFrame(data.charts),
  TranscriptFrame: createTranscriptFrame(data.transcripts),
  SchemaFrame: createSchemaFrame(data.schemas),
  FlowFrame: createFlowFrame(data.flows),
  BoardFrame: createBoardFrame(data.boards),
  ChatFrame: createChatFrame(data.chats),
});
