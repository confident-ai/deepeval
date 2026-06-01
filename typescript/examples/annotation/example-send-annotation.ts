import { sendAnnotation, AnnotationType } from "../../src/annotation";

sendAnnotation({
  threadId: "test_thread_id_1",
  type: AnnotationType.FIVE_STAR_RATING,
  rating: 5,
});
