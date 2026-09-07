// Client construction for the OpenRouter apps.
//
// Both routes into OpenRouter are covered: its own SDK, and the OpenAI SDK
// pointed at OpenRouter's base URL. Clients are memoized rather than built at
// module scope so importing an app never throws when `OPENROUTER_API_KEY` is
// absent — the suite skips instead. Instrumenting is the suite's job, since
// both patches are module-global and belong in `beforeAll`/`afterAll`.

import { OpenAI } from "openai";
import { OpenRouter } from "@openrouter/sdk";

export const MODEL = "openai/gpt-4o-mini";

export const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

export function hasApiKey(): boolean {
  return Boolean(process.env.OPENROUTER_API_KEY);
}

let openRouter: OpenRouter | undefined;
let openAIRoute: OpenAI | undefined;

export function getOpenRouterClient(): OpenRouter {
  openRouter ??= new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });
  return openRouter;
}

export function getOpenAIRouteClient(): OpenAI {
  openAIRoute ??= new OpenAI({
    baseURL: OPENROUTER_BASE_URL,
    apiKey: process.env.OPENROUTER_API_KEY,
  });
  return openAIRoute;
}
