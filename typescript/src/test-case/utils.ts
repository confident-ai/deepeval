import { LLMTestCase } from "./llm-test-case";

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
