import { APIAnnotation, AnnotationType } from "./api";

export const validateAPIAnnotation = (
  apiAnnotation: APIAnnotation,
): APIAnnotation => {
  const hasTrace = !!apiAnnotation.traceUuid;
  const hasSpan = !!apiAnnotation.spanUuid;
  const hasThread = !!apiAnnotation.threadId;

  const idCount = [hasTrace, hasSpan, hasThread].filter(Boolean).length;
  if (idCount > 1) {
    throw new Error(
      "Only one of 'traceUuid', 'spanUuid', or 'threadId' should be provided.",
    );
  }
  if (idCount === 0) {
    throw new Error(
      "One of 'traceUuid', 'spanUuid', or 'threadId' must be provided.",
    );
  }

  if (
    apiAnnotation.type === AnnotationType.FIVE_STAR_RATING &&
    apiAnnotation.rating !== undefined &&
    (apiAnnotation.rating < 1 || apiAnnotation.rating > 5)
  ) {
    throw new Error("Five star rating must be between 1 and 5.");
  }

  if (
    apiAnnotation.type === AnnotationType.THUMBS_RATING &&
    apiAnnotation.rating !== undefined &&
    (apiAnnotation.rating < 0 || apiAnnotation.rating > 1)
  ) {
    throw new Error("Thumbs rating must be either 0 or 1.");
  }

  if (apiAnnotation.threadId && apiAnnotation.expectedOutput) {
    throw new Error("Expected output cannot be provided for threads.");
  }

  if (!apiAnnotation.threadId && apiAnnotation.expectedOutcome) {
    throw new Error("Expected outcome cannot be provided for traces or spans.");
  }

  return apiAnnotation;
};
