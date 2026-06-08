import { validateAPIAnnotation } from "./utils";
import { APIAnnotation } from "./api";
import { Api, Endpoints, HttpMethods } from "../confident";

export * from "./api";

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
}: APIAnnotation) => {
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
  );
};
