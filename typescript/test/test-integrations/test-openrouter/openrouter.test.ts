// Trace-shape tests for the OpenRouter integration, against real API calls.
//
// Mirrors the other integration suites: each app under `apps/` is run once and
// the serialized trace is compared to a committed fixture. Behavioural edge
// cases that a fixture cannot express (streaming passthrough, unpatching) live
// in `openrouter-sdk.test.ts`, which stubs HTTP and needs no key.
//
// Regenerate the fixtures with:
//   OPENROUTER_API_KEY=... GENERATE_SCHEMAS=true npx jest -c jest.openrouter.config.cjs test/test-integrations/test-openrouter/openrouter.test.ts

import * as path from "path";

import { instrumentOpenAI } from "@/openai";
import { unpatchOpenAI } from "@/openai/patch";
import { instrumentOpenRouter } from "@/openrouter";
import { unpatchOpenRouter } from "@/openrouter/patch";
import { traceManager } from "@/tracing/tracing";
import { Environment } from "@/tracing/utils";

import { assertTraceJson, generateTraceJson } from "../utils";

import {
  getOpenAIRouteClient,
  getOpenRouterClient,
  hasApiKey,
} from "./apps/openrouter-clients";
import { invokeSimpleApp } from "./apps/openrouter-simple-app";
import { invokeToolApp } from "./apps/openrouter-tool-app";
import { invokeAgentApp } from "./apps/openrouter-agent-app";
import { invokeMetricCollectionApp } from "./apps/openrouter-metric-collection-app";
import { invokeOpenAIRouteApp } from "./apps/openrouter-openai-route-app";

const FIXTURES_DIR = path.join(__dirname, "fixtures");
const GENERATE_SCHEMAS = process.env.GENERATE_SCHEMAS === "true";

const traceTest = async (
  schemaName: string,
  executeFn: () => Promise<void>,
) => {
  const jsonPath = path.join(FIXTURES_DIR, schemaName);
  if (GENERATE_SCHEMAS) {
    await generateTraceJson(jsonPath, executeFn);
  } else {
    await assertTraceJson(jsonPath, executeFn);
  }
};

// The whole suite talks to OpenRouter, so skip rather than fail without a key.
const describeIfKey = hasApiKey() ? describe : describe.skip;

describeIfKey("OpenRouter Integration Tests", () => {
  beforeAll(() => {
    instrumentOpenRouter(getOpenRouterClient());
    instrumentOpenAI(getOpenAIRouteClient());
  });

  afterAll(() => {
    unpatchOpenRouter(getOpenRouterClient() as any);
    unpatchOpenAI(getOpenAIRouteClient());
  });

  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({
      environment: Environment.TESTING,
      tracingEnabled: true,
    });
  });

  test("Should capture a simple chat.send trace", async () => {
    await traceTest("openrouter_simple_schema.json", async () => {
      await invokeSimpleApp("Say hello in one short sentence.");
    });
  }, 60000);

  test("Should capture a tool-calling trace", async () => {
    await traceTest("openrouter_tool_schema.json", async () => {
      await invokeToolApp("What's the weather in Paris? Use the tool.");
    });
  }, 60000);

  test("Should nest LLM and retriever spans under an agent span", async () => {
    await traceTest("openrouter_agent_schema.json", async () => {
      await invokeAgentApp("What's the capital of France?");
    });
  }, 60000);

  test("Should carry prompt and metric collection onto the LLM span", async () => {
    await traceTest("openrouter_metric_collection_schema.json", async () => {
      await invokeMetricCollectionApp("What's the capital of France?");
    });
  }, 60000);

  test("Should capture OpenRouter reached through the OpenAI SDK", async () => {
    await traceTest("openrouter_openai_route_schema.json", async () => {
      await invokeOpenAIRouteApp("Say hello in one short sentence.");
    });
  }, 60000);
});
