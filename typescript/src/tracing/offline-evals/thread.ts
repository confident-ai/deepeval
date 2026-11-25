import { Api, Endpoints, HttpMethods } from "@/confident/api";
import {
  EvaluateThreadRequestBody,
  EvaluateThreadRequestBodySchema,
} from "@/tracing/offline-evals/api";

function getApi(): Api {
  return new Api();
}

export function evaluateThread({
  threadId,
  chatbotRole,
  metricCollection,
  overwriteMetrics = false,
  projectId,
}: EvaluateThreadRequestBody & {
  threadId: string;
  projectId?: string;
}): Promise<any> {
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
    projectId,
  );
}
