import { Api, Endpoints, HttpMethods } from "../../confident/api";
import { getCurrentTrace } from "../tracing";
import {
  EvaluateThreadRequestBody,
  EvaluateThreadRequestBodySchema,
} from "./api";

function getApi(): Api {
  const trace = getCurrentTrace();
  return new Api(trace?.confidentApiKey);
}

export function evaluateThread({
  threadId,
  chatbotRole,
  metricCollection,
  overwriteMetrics = false,
}: EvaluateThreadRequestBody & { threadId: string }): Promise<any> {
  const api = getApi();

  const body = EvaluateThreadRequestBodySchema.parse({
    metricCollection,
    chatbotRole,
    overwriteMetrics,
  });

  return api.sendRequest(
    HttpMethods.POST,
    Endpoints.EVALUATE_THREAD_ENDPOINT,
    body,
    undefined,
    undefined,
    { threadId },
  );
}
