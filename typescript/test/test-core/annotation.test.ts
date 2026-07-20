import { config } from "dotenv";

import { AnnotationType, sendAnnotation } from "../../src/annotation";

config();

const hasApiKey = !!process.env.CONFIDENT_API_KEY;

describe("Annotation Module", () => {
  (hasApiKey ? test : test.skip)(
    "should send a FIVE_STAR_RATING annotation without throwing",
    async () => {
      await expect(
        sendAnnotation({
          threadId: "thread-123",
          type: AnnotationType.FIVE_STAR_RATING,
          rating: 4,
        }),
      ).resolves.not.toThrow();
    }
  );
});
