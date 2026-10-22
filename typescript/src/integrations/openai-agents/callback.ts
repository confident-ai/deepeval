/**
 * deepeval's OpenAI Agents integration, built on confident-trace.
 *
 * `DeepEvalTracingProcessor` is confident-trace's Agents trace processor:
 * register it with `addTraceProcessor(...)`. It adds the run's workflow name,
 * group and metadata as the trace's name, thread and metadata.
 */

import { recordTracingIntegration } from "@/telemetry";
import { Integration } from "@/tracing/integrations";
import { withStaged } from "@/tracing/pending-context";
import {
  confidentTraceModule,
  ensureConfidentTrace,
} from "@/tracing/otel/setup";

type AgentsAdapter = typeof import("confident-trace/openai-agents");

const { ConfidentOpenAIAgentsProcessor } =
  confidentTraceModule<AgentsAdapter>("openai-agents");

type AgentsTrace = Parameters<
  InstanceType<typeof ConfidentOpenAIAgentsProcessor>["onTraceStart"]
>[0];

function traceFields(trace: AgentsTrace): Record<string, unknown> {
  const exported =
    (trace as { toJSON?: () => Record<string, unknown> }).toJSON?.() ?? {};
  const fields = trace as unknown as Record<string, unknown>;
  return {
    name: trace.name,
    threadId: fields.groupId ?? exported.group_id,
    metadata: fields.metadata ?? exported.metadata,
  };
}

export class DeepEvalTracingProcessor extends ConfidentOpenAIAgentsProcessor {
  constructor() {
    ensureConfidentTrace();
    super();
    recordTracingIntegration(Integration.OPENAI_AGENTS);
  }

  override async onTraceStart(trace: AgentsTrace): Promise<void> {
    // The run's root span opens synchronously inside, and takes these.
    return withStaged({ trace: traceFields(trace) }, () =>
      super.onTraceStart(trace),
    );
  }
}
