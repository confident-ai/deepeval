import type { BaseMetric } from "@/metrics/base-metrics";
import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { checkSingleTurnParams } from "@/metrics/utils";
import { checkConversationalTestCaseParams } from "@/metrics/conversational-utils";
import type { ConversationalTestCase, LLMTestCase } from "@/test-case";

/**
 * What every measure does before judging: validate the test case against the
 * metric's required params and zero its cost. Shared with the batched System
 * One path in `evaluate()`, which judges the metric without calling its
 * `measure`. Mirrors Python's `prepare_measure`.
 */
export function prepareMeasure(
  metric: BaseMetric | BaseConversationalMetric,
  testCase: LLMTestCase | ConversationalTestCase,
): void {
  if (metric instanceof BaseConversationalMetric) {
    checkConversationalTestCaseParams(
      testCase as ConversationalTestCase,
      metric.requiredParams,
      metric,
      {
        requireChatbotRole: metric.requiresChatbotRole,
        requireMcpServers: metric.requiresMcpServers,
      },
    );
  } else {
    checkSingleTurnParams(
      testCase as LLMTestCase,
      metric.requiredParams,
      metric,
    );
  }
  metric.evaluationCost = metric.usingNativeModel ? 0 : undefined;
}
