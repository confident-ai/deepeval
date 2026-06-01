import { OpenAI } from "openai";
import * as path from "path";

import { Prompt } from "../../../src/prompt";
import { setTracingContext } from "../../../src/tracing/trace-context";
import { traceManager } from "../../../src/tracing/tracing";
import { Environment } from "../../../src/tracing/utils";
import { instrumentOpenAI } from "../../../src/openai";
import { unpatchOpenAI } from "../../../src/openai/patch";

import { generateTraceJson, assertTraceJson } from "../utils";

const client = new OpenAI();

const prompt = new Prompt({ alias: "asd" });
prompt.version = "00.00.01";
prompt.label = "test-label";
prompt.hash = "bab04ec";

const FIXTURES_DIR = path.join(__dirname, "fixtures");

const GENERATE_SCHEMAS = process.env.GENERATE_SCHEMAS === "true";

describe("OpenAI Basic Integration Tests", () => {
  beforeAll(() => {
    instrumentOpenAI(client);
  });

  afterAll(() => {
    unpatchOpenAI(client);
  });

  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
    });
  });

  test("Should capture basic chat completion without explicit trace wrapper", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_openai_without_trace.json",
    );

    const executeTest = async () => {
      await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Hello, how are you?" }],
      });
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  });

  test("Should capture basic chat completion with setTracingContext", async () => {
    const jsonPath = path.join(FIXTURES_DIR, "expected_openai_with_trace.json");

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "test_thread_id_1",
          name: "test_name_1",
          tags: ["test_tag_1"],
          metadata: { test_metadata_1: "test_value_1" },
          userId: "test_user_id_1",
          llmSpanContext: {
            prompt: prompt,
            metricCollection: "test_collection_1",
          },
        },
        async () => {
          await client.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [{ role: "user", content: "Hello, how are you?" }],
          });
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  });

  test("Should capture Responses API creation with setTracingContext", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_response_create_with_trace.json",
    );

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "test_thread_id_1",
          name: "test_name_1",
          tags: ["test_tag_1"],
          metadata: { test_metadata_1: "test_value_1" },
          userId: "test_user_id_1",
          llmSpanContext: {
            prompt: prompt,
            metricCollection: "test_collection_1",
          },
        },
        async () => {
          await client.responses.create({
            model: "gpt-4o-mini",
            instructions:
              "You are a helpful assistant. Always generate a string response.",
            input: "Hello, how are you?",
          });
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  });

  test("Should capture all Confident AI specific attributes for OpenAI", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_openai_confident_attributes.json",
    );

    const mockPrompt = new Prompt({ alias: "openai-test-prompt" });
    mockPrompt.version = "00.00.01";
    mockPrompt.label = "test-label";
    mockPrompt.hash = "bab04ec";

    const executeTest = async () => {
      await setTracingContext(
        {
          threadId: "openai-thread-404",
          name: "OpenAI Attributes Trace",
          tags: ["openai", "production", "v2"],
          metadata: { customer_tier: "enterprise" },
          userId: "user-beta-888",
          metricCollection: "global-openai-metrics",

          llmSpanContext: {
            prompt: mockPrompt,
            metricCollection: "llm-openai-metrics",
          },
        },
        async () => {
          await client.chat.completions.create({
            model: "gpt-4o-mini",
            temperature: 0,
            messages: [
              {
                role: "user",
                content: "Tell me a fun fact about space. Be brief.",
              },
            ],
          });
        },
      );
    };

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  });
});
