import { validateAPIAnnotation } from "@/annotation/utils";
import { APIAnnotation } from "@/annotation/api";
import { Api, Endpoints, HttpMethods } from "@/confident";

export * from "@/annotation/api";

export const sendAnnotation = async ({
  rating,
  traceUuid,
  spanUuid,
  threadId,
  expectedOutput,
  expectedOutcome,
  explanation,
  type,
  userId,
  projectId,
}: APIAnnotation & { projectId?: string }) => {
  const apiAnnotation = validateAPIAnnotation({
    rating,
    traceUuid,
    spanUuid,
    threadId,
    expectedOutput,
    expectedOutcome,
    explanation,
    type,
    userId,
  });
  const api = new Api();

  await api.sendRequest(
    HttpMethods.POST,
    Endpoints.ANNOTATION_ENDPOINT,
    apiAnnotation,
    undefined,
    undefined,
    undefined,
    projectId,
  );
};
