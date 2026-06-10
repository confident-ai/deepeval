import { Api, Endpoints, HttpMethods } from "../../confident/api";
import { getCurrentTrace } from "../tracing";
import { EvaluateSpanRequestBody, EvaluateSpanRequestBodySchema } from "./api";

function getApi(): Api {
  const trace = getCurrentTrace();
  return new Api(trace?.confidentApiKey);
}

export function evaluateSpan({
  spanUuid,
  metricCollection,
  overwriteMetrics = false,
}: EvaluateSpanRequestBody & { spanUuid: string }): Promise<any> {
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
  );
}
