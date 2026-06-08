import {
  observe,
  updateCurrentSpan,
  updateCurrentTrace,
} from "../../src/tracing/tracing";

const llmApp = () => {
  // Add span-level metadata
  updateCurrentSpan({
    metadata: {
      source: "knowledge_base_1",
      retrieved_documents: 3,
    },
  });

  // Add trace-level metadata
  updateCurrentTrace({
    metadata: {
      user_id: "user-456",
      app_version: "1.2.3",
    },
  });
};

const observedLlmApp = observe({ fn: llmApp });

observedLlmApp();
