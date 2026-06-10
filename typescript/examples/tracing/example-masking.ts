import { observe, traceManager } from "../../src/tracing/tracing";

const maskingFunction = (data: any): any => {
  if (typeof data === "string") {
    return data.replace(/\b(?:\d{4}[- ]?){3}\d{4}\b/g, "[REDACTED CARD]");
  }
  return data;
};

traceManager.configure({ mask: maskingFunction });

const agent = (_query: string): string => {
  return "4242-4242-4242-4242";
};

const observedAgent = observe({
  type: "agent",
  fn: agent,
});

observedAgent("Test Masking");
