import { ConversationalTestCase, MultiTurnParams } from "../test-case";
import { BaseMetricCore } from "./base-metrics";

export abstract class BaseConversationalMetric extends BaseMetricCore {
  protected requiredParams: Array<MultiTurnParams> = [];

  abstract measure(
    testCase: ConversationalTestCase,
    ...args: any[]
  ): number | Promise<number>;

  get name(): string {
    return "Base Conversational Metric";
  }
}
