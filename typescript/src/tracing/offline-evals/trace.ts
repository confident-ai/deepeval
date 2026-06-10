import { Api, Endpoints, HttpMethods } from "../../confident/api";
import { getCurrentTrace } from "../tracing";
import {
  EvaluateTraceRequestBody,
  EvaluateTraceRequestBodySchema,
} from "./api";

function getApi(): Api {
  const trace = getCurrentTrace();
  return new Api(trace?.confidentApiKey);
}

export function evaluateTrace({
  traceUuid,
  metricCollection,
  overwriteMetrics = false,
}: EvaluateTraceRequestBody & { traceUuid: string }): Promise<any> {
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
  );
}
