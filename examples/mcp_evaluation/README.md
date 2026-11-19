# Parallel Search MCP example

Capture free web search and page fetching from [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) in a DeepEval `LLMTestCase`. The anonymous endpoint needs no Parallel account or API key and uses Fast mode for search, with free-tier rate limits.

From the repository root, use Python 3.10 or newer:

```bash
python -m venv .venv-parallel
source .venv-parallel/bin/activate
python -m pip install -r examples/mcp_evaluation/requirements-parallel.txt
python examples/mcp_evaluation/parallel_search.py \
  "Find how DeepEval records MCP tool calls" \
  --query "DeepEval MCP tool calls" \
  --url https://deepeval.com/docs/evaluation-mcp
```

The script connects with Streamable HTTP, discovers the server's tools, calls `web_search`, and optionally calls `web_fetch` for the supplied URLs. Repeat `--query` for additional queries or `--url` for additional pages (up to 20). It prints a JSON test case with the discovered `MCPServer`, each `MCPToolCall` including its arguments and full result, and the combined tool text as `actual_output`. Both calls share a session identifier. Tool errors stop the script, and the connection closes on completion or failure.

This example calls tools directly and captures their output; it does not generate an LLM answer, run an agent loop, or score the test case. To evaluate your own MCP agent, use its final answer as `actual_output` and retain the server metadata and tool calls as shown here. See [DeepEval's MCP evaluation guide](https://deepeval.com/docs/evaluation-mcp) for running MCP metrics with a configured judge model.
