import { Agent, tool } from "@openai/agents";
import { z } from "zod";

const getCompanyInfo = tool({
  name: "get_company_info",
  description: "Get company information for a ticker symbol.",
  parameters: z.object({
    symbol: z.string().describe("The ticker symbol, e.g. AAPL"),
  }),
  execute: async ({ symbol }) => {
    const info: Record<string, string> = {
      AAPL: "Apple Inc. - Technology company",
      GOOGL: "Alphabet Inc. - Technology company",
      MSFT: "Microsoft Corporation - Technology company",
    };
    return (
      info[symbol.toUpperCase()] || `Company info not available for ${symbol}`
    );
  },
});

export const streamingAgent = new Agent({
  name: "StreamingAgent",
  instructions: `You are a helpful assistant. 
    If asked for company info, use the tool. 
    If asked a general question, write a short poem about it to generate many tokens.`,
  model: "gpt-4o",
  tools: [getCompanyInfo],
});
