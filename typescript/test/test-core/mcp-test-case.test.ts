import {
  ConversationalTestCase,
  LLMTestCase,
  MCPPromptCall,
  MCPResourceCall,
  MCPServer,
  MCPToolCall,
  Turn,
} from "@/test-case";

const toolResult = {
  content: [{ type: "text", text: "72F and sunny" }],
  isError: false,
  _meta: { "io.modelcontextprotocol/related-task": { taskId: "t1" } },
};

const testCase = (mcpToolsCalled: MCPToolCall[]) =>
  new LLMTestCase({
    input: "What's the weather?",
    actualOutput: "72F and sunny.",
    mcpServers: [new MCPServer({ serverName: "weather" })],
    mcpToolsCalled,
  });

describe("MCP call validation", () => {
  it("accepts a result carrying content, structured content, or both", () => {
    expect(() =>
      testCase([
        new MCPToolCall({ name: "get_weather", args: {}, result: toolResult }),
      ]),
    ).not.toThrow();
    expect(() =>
      testCase([
        new MCPToolCall({
          name: "get_weather",
          args: {},
          result: { structuredContent: { tempF: 72 } },
        }),
      ]),
    ).not.toThrow();
  });

  it("rejects a result that is not an MCP CallToolResult", () => {
    for (const result of ["72F and sunny", {}, { foo: "bar" }, null]) {
      expect(() =>
        testCase([new MCPToolCall({ name: "get_weather", args: {}, result })]),
      ).toThrow(/'mcpToolsCalled'.*'CallToolResult'/);
    }
  });

  it("distinguishes resource and prompt results", () => {
    const conversation = (
      mcpResourcesCalled: MCPResourceCall[],
      mcpPromptsCalled: MCPPromptCall[],
    ) =>
      new ConversationalTestCase({
        turns: [
          new Turn({ role: "user", content: "Summarize the docs." }),
          new Turn({
            role: "assistant",
            content: "Here you go.",
            mcpResourcesCalled,
            mcpPromptsCalled,
          }),
        ],
        mcpServers: [new MCPServer({ serverName: "docs" })],
      });

    expect(() =>
      conversation(
        [
          new MCPResourceCall({
            uri: "file:///docs.md",
            result: { contents: [{ uri: "file:///docs.md", text: "# Docs" }] },
          }),
        ],
        [
          new MCPPromptCall({
            name: "summarize",
            result: {
              messages: [
                { role: "user", content: { type: "text", text: "hi" } },
              ],
            },
          }),
        ],
      ),
    ).not.toThrow();

    // A resource result carries `contents`, not a tool result's `content`.
    expect(() =>
      conversation(
        [new MCPResourceCall({ uri: "file:///docs.md", result: toolResult })],
        [],
      ),
    ).toThrow(/'mcpResourcesCalled'.*'ReadResourceResult'/);
    expect(() =>
      conversation(
        [],
        [new MCPPromptCall({ name: "summarize", result: toolResult })],
      ),
    ).toThrow(/'mcpPromptsCalled'.*'GetPromptResult'/);
  });
});
