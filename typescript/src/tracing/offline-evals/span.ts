import { Api, Endpoints, HttpMethods } from "@/confident/api";
import {
  EvaluateSpanRequestBody,
  EvaluateSpanRequestBodySchema,
} from "@/tracing/offline-evals/api";

function getApi(): Api {
  return new Api();
}

export function evaluateSpan({
  spanUuid,
  metricCollection,
  overwriteMetrics = false,
  projectId,
}: EvaluateSpanRequestBody & {
  spanUuid: string;
  projectId?: string;
}): Promise<any> {
  const api = getApi();

  const body = EvaluateSpanRequestBodySchema.parse({
    metricCollection,
    overwriteMetrics,
  });

  return api.sendRequest(
    HttpMethods.POST,
    Endpoints.EVALUATE_SPAN_ENDPOINT,
    body,
    undefined,
    undefined,
    { spanUuid },
    projectId,
  );
}
