import { Api, Endpoints, HttpMethods } from "@/confident/api";
import {
  EvaluateTraceRequestBody,
  EvaluateTraceRequestBodySchema,
} from "@/tracing/offline-evals/api";

function getApi(): Api {
  return new Api();
}

export function evaluateTrace({
  traceUuid,
  metricCollection,
  overwriteMetrics = false,
  projectId,
}: EvaluateTraceRequestBody & {
  traceUuid: string;
  projectId?: string;
}): Promise<any> {
  const api = getApi();

  const body = EvaluateTraceRequestBodySchema.parse({
    metricCollection,
    overwriteMetrics,
  });

  return api.sendRequest(
    HttpMethods.POST,
    Endpoints.EVALUATE_TRACE_ENDPOINT,
    body,
    undefined,
    undefined,
    { traceUuid },
    projectId,
  );
}
