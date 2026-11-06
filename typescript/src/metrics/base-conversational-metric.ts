import { ConversationalTestCase, MultiTurnParams } from "@/test-case";
import { BaseMetricCore } from "@/metrics/base-metrics";

export abstract class BaseConversationalMetric extends BaseMetricCore {
  /** @internal Read by `prepareMeasure`. */
  requiredParams: Array<MultiTurnParams> = [];
  /** @internal Read by `prepareMeasure`. */
  requiresChatbotRole: boolean = false;
  /** @internal Read by `prepareMeasure`. */
  requiresMcpServers: boolean = false;

  abstract measure(
    testCase: ConversationalTestCase,
    ...args: any[]
  ): number | Promise<number>;

  get name(): string {
    return "Base Conversational Metric";
  }
}
