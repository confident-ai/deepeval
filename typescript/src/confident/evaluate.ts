import { Api, Endpoints, API_BASE_URL, HttpMethods } from "./api";
import { LLMTestCase, ConversationalTestCase, Turn } from "../test-case";
import { isConfident } from "../utils";
import { Prompt } from "../prompt";

interface ApiToolCall {
  name: string;
  description?: string;
  reasoning?: string;
  output?: any;
  inputParameters?: Record<string, any>;
}

interface ApiTurn {
  role: "user" | "assistant";
  content: string;
  userId?: string;
  retrievalContext?: string[];
  toolsCalled?: ApiToolCall[];
  additionalMetadata?: Record<string, any>;
}

interface ApiTestCase {
  input: string;
  actualOutput: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: string[];
  additionalMetadata?: Record<string, any>;
  comments?: string;
  toolsCalled?: ApiToolCall[];
  expectedTools?: ApiToolCall[];
  reasoning?: string;
  tokenCost?: number;
  completionTime?: number;
  name?: string;
}

interface ApiConversationalTestCase {
  turns: ApiTurn[];
  scenario?: string;
  expectedOutcome?: string;
  userDescription?: string;
  chatbotRole?: string;
}

interface PromptApi {
  id: string;
  type: string;
}

interface ConfidentEvaluateRequestData {
  metricCollection: string;
  llmTestCases?: ApiTestCase[];
  conversationalTestCases?: ApiConversationalTestCase[];
  hyperparameters?: { [key: string]: string | PromptApi };
  identifier?: string;
}

interface ConfidentEvaluateResponseData {
  link: string;
}

function convertTurn(turn: Turn): ApiTurn {
  const toolsCalled = turn.toolsCalled
    ? turn.toolsCalled.map((tool) => ({
        name: tool.name,
        description: tool.description,
        reasoning: tool.reasoning,
        output: tool.output,
        inputParameters: tool.inputParameters,
      }))
    : undefined;

  return {
    role: turn.role,
    content: turn.content,
    userId: turn.userId,
    retrievalContext: turn.retrievalContext,
    toolsCalled: toolsCalled,
    additionalMetadata: turn.additionalMetadata,
  };
}

function convertLLMTestCase(testCase: LLMTestCase): ApiTestCase {
  const toolsCalled = testCase.toolsCalled
    ? testCase.toolsCalled.map((tool) => ({
        name: tool.name,
        description: tool.description,
        reasoning: tool.reasoning,
        output: tool.output,
        inputParameters: tool.inputParameters,
      }))
    : undefined;

  const expectedTools = testCase.expectedTools
    ? testCase.expectedTools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        reasoning: tool.reasoning,
        output: tool.output,
        inputParameters: tool.inputParameters,
      }))
    : undefined;

  return {
    input: testCase.input,
    actualOutput: testCase.actualOutput,
    expectedOutput: testCase.expectedOutput,
    context: testCase.context,
    retrievalContext: testCase.retrievalContext,
    additionalMetadata: testCase.additionalMetadata,
    comments: testCase.comments,
    toolsCalled: toolsCalled,
    expectedTools: expectedTools,
    reasoning: testCase.reasoning,
    tokenCost: testCase.tokenCost,
    completionTime: testCase.completionTime,
    name: testCase.name,
  };
}

function convertConversationalTestCase(
  testCase: ConversationalTestCase,
): ApiConversationalTestCase {
  const turns = testCase.turns.map(convertTurn);

  return {
    turns: turns,
    scenario: testCase.scenario || undefined,
    expectedOutcome: testCase.expectedOutcome || undefined,
    userDescription: testCase.userDescription || undefined,
    chatbotRole: testCase.chatbotRole || undefined,
  };
}

async function processHyperparameters(hyperparameters: {
  [key: string]: string | number | boolean | Prompt;
}): Promise<{ [key: string]: string | PromptApi }> {
  const processed: { [key: string]: string | PromptApi } = {};

  for (const [key, value] of Object.entries(hyperparameters)) {
    if (value instanceof Prompt) {
      try {
        if (!value.hash || value.hash === "latest" || !value.type) {
          await value.push();
        }
        processed[key] = {
          id: value.hash,
          type: value.type || (value.textTemplate !== null ? "TEXT" : "LIST"),
        };
      } catch (e) {
        console.warn(`Failed to process prompt hyperparameter '${key}':`, e);
        processed[key] = "Error processing prompt";
      }
    } else {
      processed[key] = String(value);
    }
  }

  return processed;
}

export async function evaluate(params: {
  metricCollection: string;
  llmTestCases?: LLMTestCase[];
  conversationalTestCases?: ConversationalTestCase[];
  hyperparameters?: { [key: string]: string | number | boolean | Prompt };
  identifier?: string;
}): Promise<void> {
  const {
    metricCollection,
    llmTestCases,
    conversationalTestCases,
    hyperparameters,
    identifier,
  } = params;

  /////////////////////////////////////////////////////////
  /// Type Checking
  /////////////////////////////////////////////////////////

  if (
    (llmTestCases?.length ?? 0) === 0 &&
    (conversationalTestCases?.length ?? 0) === 0
  ) {
    throw new Error(
      "You must provide either a non-empty array of 'llmTestCases' or 'conversationalTestCases'",
    );
  }
  const testCaseLength =
    (llmTestCases?.length ?? 0) + (conversationalTestCases?.length ?? 0);

  ////////////////////////////////////////////////////////
  /// Posting Data
  /////////////////////////////////////////////////////////

  if (isConfident()) {
    console.log(`Sending ${testCaseLength} test case(s) to Confident AI...`);
    const startTime = performance.now();

    try {
      const api = new Api(undefined, API_BASE_URL);

      let processedHyperparameters:
        | { [key: string]: string | PromptApi }
        | undefined;
      if (hyperparameters) {
        processedHyperparameters =
          await processHyperparameters(hyperparameters);
      }

      let confidentRequestData: ConfidentEvaluateRequestData;
      if (llmTestCases) {
        const convertedTestCases = llmTestCases.map(convertLLMTestCase);
        confidentRequestData = {
          metricCollection,
          llmTestCases: convertedTestCases,
          hyperparameters: processedHyperparameters,
          identifier,
        };
      } else if (conversationalTestCases) {
        const convertedTestCases = conversationalTestCases.map(
          convertConversationalTestCase,
        );
        confidentRequestData = {
          metricCollection,
          conversationalTestCases: convertedTestCases,
          hyperparameters: processedHyperparameters,
          identifier,
        };
      } else {
        throw new Error(
          "You must provide either a non-empty array of 'llmTestCases' or 'conversationalTestCases'",
        );
      }

      const result = await api.sendRequest(
        HttpMethods.POST,
        Endpoints.EVALUATE_ENDPOINT,
        confidentRequestData,
      );

      const endTime = performance.now();
      const timeTaken = ((endTime - startTime) / 1000).toFixed(2);

      if (result) {
        const response: ConfidentEvaluateResponseData = {
          link: result.link,
        };
        console.log(`Done! (${timeTaken}s)`);
        console.log(
          `✓ Evaluation of metric collection '${metricCollection}' started! View progress on ${response.link}`,
        );
      }
    } catch (error) {
      const endTime = performance.now();
      const timeTaken = ((endTime - startTime) / 1000).toFixed(2);
      console.error(`Error! (${timeTaken}s)`);
      throw error;
    }
  } else {
    throw new Error(
      "To run evaluations on Confident AI, run `deepeval login`.",
    );
  }
}
