// Deprecated: `ArgumentCorrectnessMetric` is now `ArgumentRelevancyMetric`.
// It judges whether tool-call arguments fit the input and never compares them
// with expected values; for that, use `ToolCorrectnessMetric` with
// `evaluationParams: [ToolCallParams.INPUT_PARAMETERS]`.
import {
  ArgumentRelevancyMetric,
  type ArgumentRelevancyMetricOptions,
  type ArgumentRelevancyTemplateOverride,
} from "@/metrics/argument-relevancy";

let warned = false;

/** @deprecated Use `ArgumentRelevancyMetric`. */
export class ArgumentCorrectnessMetric extends ArgumentRelevancyMetric {
  constructor(options: ArgumentRelevancyMetricOptions = {}) {
    if (!warned) {
      warned = true;
      console.warn(
        "ArgumentCorrectnessMetric is deprecated and will be removed in a future release. Use ArgumentRelevancyMetric instead.",
      );
    }
    super(options);
  }
}

/** @deprecated Use `ArgumentRelevancyMetricOptions`. */
export type ArgumentCorrectnessMetricOptions = ArgumentRelevancyMetricOptions;
/** @deprecated Use `ArgumentRelevancyTemplateOverride`. */
export type ArgumentCorrectnessTemplateOverride =
  ArgumentRelevancyTemplateOverride;
