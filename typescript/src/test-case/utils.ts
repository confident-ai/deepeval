import { LLMTestCase, ToolCall } from "./llm-test-case";


// Render a span/trace value as a test case's text field.git 
export function asTestCaseString(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    // Circular or otherwise unserialisable — better than throwing mid-run.
    return String(value);
  }
}

export function asToolCalls(
  value: ToolCall[] | undefined,
): ToolCall[] | undefined {
  if (value === undefined || value === null || !Array.isArray(value)) {
    return undefined;
  }
  return value.map((call) => {
    if (call instanceof ToolCall) return call;
    // Typed as ToolCall but shaped like one at runtime — read it loosely.
    const loose = call as Partial<ToolCall> | null | undefined;
    return new ToolCall({
      name: String(loose?.name ?? "unknown"),
      description: loose?.description,
      type: loose?.type,
      reasoning: loose?.reasoning,
      output: loose?.output,
      inputParameters: loose?.inputParameters,
    });
  });
}

/**
 * Deep copy function for TypeScript.
 * @param obj - The object to deep copy
 * @returns A deep copy of the object
 */
export function deepcopy<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

/**
 * Check if the test cases are of valid types.
 * @param testCases - Array of test cases to check
 * @throws Error if there's a mixture of LLMTestCase/MLLMTestCase and ConversationalTestCase
 */
export function checkValidTestCasesType(testCases: Array<LLMTestCase>): void {
  let llmTestCaseCount = 0;
  let conversationalTestCaseCount = 0;

  for (const testCase of testCases) {
    if (testCase instanceof LLMTestCase) {
      llmTestCaseCount += 1;
    } else {
      conversationalTestCaseCount += 1;
    }
  }

  if (llmTestCaseCount > 0 && conversationalTestCaseCount > 0) {
    throw new Error(
      "You cannot supply a mixture of `LLMTestCase`/`MLLMTestCase`(s) and `ConversationalTestCase`(s) as the list of test cases.",
    );
  }
}
