/** Annotate retrieval context lists with node IDs for multimodal metric prompts. */

import { convertToMultiModalArray, MLLMImage } from "@/test-case/mllm-image";

/**
 * Label each node `Node N:` so verdicts can cite images positionally. Images stay
 * `MLLMImage`s: the resolver renders them as slugs the model layer then expands.
 */
export function idRetrievalContext(
  retrievalContext: string[],
): (string | MLLMImage)[] {
  const annotated: (string | MLLMImage)[] = [];
  const seq = convertToMultiModalArray(retrievalContext);
  seq.forEach((context, i) => {
    if (typeof context === "string") {
      annotated.push(`Node ${i + 1}: ${context}`);
    } else {
      annotated.push(`Node ${i + 1}:`);
      annotated.push(context);
    }
  });
  return annotated;
}

/** Python's `_contextual_recall_verdict_kwargs`, shared by both turn variants. */
export function contextualRecallVerdictVars(
  retrievalContext: string[],
  multimodal: boolean,
): Record<string, unknown> {
  return {
    content_type: multimodal ? "sentence and image" : "sentence",
    content_type_plural: multimodal ? "sentences and images" : "sentences",
    content_or: multimodal ? "sentence or image" : "sentence",
    context_to_display: multimodal
      ? idRetrievalContext(retrievalContext)
      : retrievalContext,
    node_instruction: multimodal
      ? " A node is either a string or image, but not both (so do not group " +
        "images and texts in the same nodes)."
      : "",
  };
}

export function contextualRecallReasonContentType(multimodal: boolean): string {
  return multimodal ? "sentence or image" : "sentence";
}

/** Python's `_contextual_relevancy_verdict_kwargs`. */
export function contextualRelevancyVerdictVars(
  multimodal: boolean,
): Record<string, string> {
  return {
    context_type: multimodal ? "context (image or string)" : "context",
    statement_or_image: multimodal ? "statement or image" : "statement",
    extraction_instructions: multimodal
      ? "If the context is textual, you should first extract the statements found in the context if the context, which are high level information found in the context, before deciding on a verdict and optionally a reason for each statement.\n" +
        "If the context is an image, `statement` should be a description of the image. Do not assume any information not visibly available."
      : "You should first extract statements found in the context, which are " +
        "high level information found in the context, before deciding on a " +
        "verdict and optionally a reason for each statement.",
    empty_context_instruction: multimodal
      ? ""
      : "\nIf provided context contains no actual content or statements then: " +
        'give "no" as a "verdict",\nput context into "statement", and ' +
        '"No statements found in provided context." into "reason".',
  };
}
