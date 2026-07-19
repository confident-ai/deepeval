import { Mastra } from "@mastra/core/mastra";
import { Observability } from "@mastra/observability";
import type { DeepEvalExporter } from "../../../../src/integrations/mastra";

export function buildMastra(
  exporter: DeepEvalExporter,
  config: Record<string, any>,
) {
  return new Mastra({
    ...config,
    observability: new Observability({
      configs: {
        deepeval: {
          serviceName: "deepeval-mastra-test",
          exporters: [exporter as any],
        },
      },
    }),
  } as any);
}
