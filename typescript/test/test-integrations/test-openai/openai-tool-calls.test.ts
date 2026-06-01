import { OpenAI } from "openai";
import * as path from "path";

import { traceManager, observe, SpanType } from "../../../src/tracing/tracing";
import { Environment } from "../../../src/tracing/utils";
import { instrumentOpenAI } from "../../../src/openai";
import { unpatchOpenAI } from "../../../src/openai/patch";
import { generateTraceJson, assertTraceJson } from "../utils";

const client = new OpenAI();
const FIXTURES_DIR = path.join(__dirname, "fixtures");
const GENERATE_SCHEMAS = process.env.GENERATE_SCHEMAS === "true";

const getWeather = observe({
  type: SpanType.TOOL,
  name: "get_weather",
  fn: async (location: string, unit: string = "c") => {
    const data: Record<string, any> = {
      "San Francisco": { temp_c: 18, condition: "partly cloudy" },
      "New York": { temp_c: 22, condition: "sunny" },
      London: { temp_c: 15, condition: "light rain" },
    };

    const city = location.trim();
    const entry = data[city] || { temp_c: 20, condition: "clear" };

    if (unit.toLowerCase() === "f") {
      const temp = Math.round((entry.temp_c * 9) / 5 + 32);
      return {
        location: city,
        temperature: temp,
        unit: "F",
        condition: entry.condition,
      };
    }

    return {
      location: city,
      temperature: entry.temp_c,
      unit: "C",
      condition: entry.condition,
    };
  },
});

const COMPLETION_TOOLS: any[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a city.",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "City name, e.g. 'San Francisco'",
          },
          unit: {
            type: "string",
            enum: ["c", "f"],
            description: "Temperature unit",
          },
        },
        required: ["location"],
        additionalProperties: false,
      },
    },
  },
];

const RESPONSES_TOOLS: any[] = [
  {
    type: "function",
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameters: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "City name, e.g. 'San Francisco'",
        },
        unit: {
          type: "string",
          enum: ["c", "f"],
          description: "Temperature unit",
        },
      },
      required: ["location"],
      additionalProperties: false,
    },
  },
];

describe("OpenAI Tool Calling Flow Tests", () => {
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

  test("Should capture tool call flow using Chat Completions API", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_tool_call_flow_completion.json",
    );

    const executeTest = observe({
      type: SpanType.CUSTOM,
      name: "run_main_completion",
      fn: async () => {
        const systemPrompt =
          "You are a helpful assistant. Use tools when they are needed to get accurate data.";
        const userPrompt =
          "What's the weather in San Francisco in celsius? Then give a one-sentence travel tip that fits the weather.";

        const messages: any[] = [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ];

        // 1st LLM Call
        const first = await client.chat.completions.create({
          model: "gpt-4o-mini",
          messages: messages,
          tools: COMPLETION_TOOLS,
          tool_choice: "auto",
          temperature: 0,
        });

        const assistantMsg = first.choices[0].message;
        const toolCalls = assistantMsg.tool_calls || [];

        if (toolCalls.length > 0) {
          messages.push(assistantMsg);

          // Execute tools locally
          for (const tc of toolCalls) {
            if (tc.type === "function") {
              const name = tc.function.name;
              const args = JSON.parse(tc.function.arguments || "{}");

              let result;
              if (name === "get_weather") {
                result = await getWeather(args.location, args.unit);
              } else {
                result = { error: `Unknown tool '${name}'` };
              }

              messages.push({
                role: "tool",
                tool_call_id: tc.id,
                name: name,
                content: JSON.stringify(result),
              });
            }
          }

          // 2nd LLM Call (Final answer)
          await client.chat.completions.create({
            model: "gpt-4o-mini",
            messages: messages,
            temperature: 0,
          });
        }
      },
    });

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 60_000);

  test("Should capture tool call flow using Responses API", async () => {
    const jsonPath = path.join(
      FIXTURES_DIR,
      "expected_tool_call_flow_response.json",
    );

    const executeTest = observe({
      type: SpanType.CUSTOM,
      name: "run_main_response",
      fn: async () => {
        const systemPrompt =
          "You are a helpful assistant. Use tools when they are needed to get accurate data.";
        const userPrompt =
          "What's the weather in San Francisco in celsius? Then give a one-sentence travel tip that fits the weather.";

        const first = await client.responses.create({
          model: "gpt-4o-mini",
          instructions: systemPrompt,
          input: userPrompt,
          tools: RESPONSES_TOOLS,
          temperature: 0,
        });

        const toolCalls: any[] = [];
        for (const item of (first as any).output || []) {
          if (item.type === "function_call") {
            toolCalls.push(item);
          }
        }

        if (toolCalls.length > 0) {
          const functionCallOutputs: any[] = [];

          for (const tc of toolCalls) {
            const name = tc.name;
            const args = JSON.parse(tc.arguments || "{}");

            let result;
            if (name === "get_weather") {
              result = await getWeather(args.location, args.unit);
            } else {
              result = { error: `Unknown tool '${name}'` };
            }

            functionCallOutputs.push({
              type: "function_call_output",
              call_id: tc.call_id,
              output: JSON.stringify(result),
            });
          }

          await client.responses.create({
            model: "gpt-4o-mini",
            previous_response_id: first.id,
            input: functionCallOutputs,
            temperature: 0,
          });
        }
      },
    });

    if (GENERATE_SCHEMAS) {
      await generateTraceJson(jsonPath, executeTest);
    } else {
      await assertTraceJson(jsonPath, executeTest);
    }
  }, 60_000);
});
