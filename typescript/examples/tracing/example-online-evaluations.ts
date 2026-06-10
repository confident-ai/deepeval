import {
  observe,
  updateCurrentSpan,
  updateCurrentTrace,
} from "../../src/tracing/tracing";

const generate = async (prompt: string) => {
  updateCurrentSpan({
    input: prompt,
    output: "LLM response",
  });
  updateCurrentTrace({
    input: prompt,
    output: "LLM response",
    metricCollection: "Task Completion Collection",
  });
  return "LLM response";
};

const observedGenerate = observe({
  type: "llm",
  model: "gpt-4",
  metricCollection: "My Metrics",
  fn: generate,
});

observedGenerate("Write response");
