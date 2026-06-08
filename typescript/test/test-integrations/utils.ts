import * as fs from "fs";
import * as path from "path";
import { traceManager } from "../../src/tracing/tracing";

function isUnorderedPath(pathStr: string): boolean {
  return (
    pathStr === "root.baseSpans" ||
    pathStr === "root.llmSpans" ||
    pathStr === "root.toolSpans" ||
    pathStr.endsWith(".toolsCalled") ||
    pathStr.endsWith(".tool_calls")
  );
}

function normalizeSpanOrTool(item: any, isTool: boolean): string {
  if (!item || typeof item !== "object") return "unknown";

  let key = "";
  if (isTool) {
    let name = item.name || "";
    let argsObj = item.inputParameters || item.args || {};

    if (!name && item.function) {
      name = item.function.name || "";
      if (typeof item.function.arguments === "string") {
        try {
          argsObj = JSON.parse(item.function.arguments);
        } catch (e) {
          argsObj = {};
        }
      } else if (
        typeof item.function.arguments === "object" &&
        item.function.arguments !== null
      ) {
        argsObj = item.function.arguments;
      }
    }

    const argKeys = Object.keys(argsObj).sort().join(",");
    key = `${name}-${argKeys}`;
  } else {
    const name = item.name || "";
    const type = item.type || item.spanType || "span";

    if (name) {
      key = name;
    } else {
      key = `${type}-unnamed`;
    }
  }
  return key;
}

export function assertJsonObjectStructure(
  expected: any,
  actual: any,
  currentPath: string = "root",
): void {
  // Handle explicit nulls
  if (expected === null) {
    expect({ path: currentPath, value: actual }).toEqual({
      path: currentPath,
      value: null,
    });
    return;
  }

  if (typeof expected === "object" && !Array.isArray(expected)) {
    expect({
      path: currentPath,
      type: typeof actual,
      isArray: Array.isArray(actual),
    }).toEqual({ path: currentPath, type: "object", isArray: false });
    expect(actual).not.toBeNull();

    const keysToIgnore = new Set([
      "tokenIntervals",
      "lc",
      "id",
      "kwargs",
      "prompt_cache_retention",
      "moderation",
    ]);

    const expectedKeys = Object.keys(expected).filter(
      (k) => !keysToIgnore.has(k),
    );
    const actualKeys = Object.keys(actual).filter((k) => !keysToIgnore.has(k));

    for (const key of expectedKeys) {
      if (
        currentPath === "root" &&
        key === "toolsCalled" &&
        !actualKeys.includes(key)
      ) {
        continue; // Handle root toolsCalled drift
      }
      // Responses API: instructions may appear on output but not in traced input args
      if (
        key === "instructions" &&
        currentPath.includes(".input.") &&
        !actualKeys.includes(key)
      ) {
        continue;
      }
      expect({
        path: `${currentPath}.${key}`,
        exists: actualKeys.includes(key),
      }).toEqual({ path: `${currentPath}.${key}`, exists: true });

      assertJsonObjectStructure(
        expected[key],
        actual[key],
        `${currentPath}.${key}`,
      );
    }
    return;
  }

  if (Array.isArray(expected)) {
    expect({ path: currentPath, isArray: Array.isArray(actual) }).toEqual({
      path: currentPath,
      isArray: true,
    });
    expect({ path: currentPath, length: actual.length }).toEqual({
      path: currentPath,
      length: expected.length,
    });

    if (isUnorderedPath(currentPath)) {
      const isToolList =
        currentPath.endsWith(".toolsCalled") ||
        currentPath.endsWith(".tool_calls");

      const expectedKeys = expected.map((e: any) =>
        normalizeSpanOrTool(e, isToolList),
      );
      const actualKeys = actual.map((a: any) =>
        normalizeSpanOrTool(a, isToolList),
      );

      const matchedActualIndices = new Set<number>();
      const matchErrors: Error[] = [];

      for (let i = 0; i < expected.length; i++) {
        const expKey = expectedKeys[i];
        let foundMatch = false;

        for (let j = 0; j < actual.length; j++) {
          if (matchedActualIndices.has(j)) continue;

          if (expKey === actualKeys[j]) {
            try {
              assertJsonObjectStructure(
                expected[i],
                actual[j],
                `${currentPath}[${i}]`,
              );
              matchedActualIndices.add(j);
              foundMatch = true;
              break;
            } catch (e: any) {
              matchErrors.push(e);
            }
          }
        }

        if (!foundMatch) {
          if (matchErrors.length > 0) {
            console.error(
              `\n❌ Structural mismatches found for key '${expKey}' at path '${currentPath}':`,
            );
            matchErrors.forEach((err) => console.error(err.message));
          }
          throw new Error(
            `No matching element found in unordered list at path: ${currentPath} for item with key: ${expKey}`,
          );
        }
      }
      return;
    }

    for (let i = 0; i < expected.length; i++) {
      assertJsonObjectStructure(expected[i], actual[i], `${currentPath}[${i}]`);
    }
    return;
  }

  if (typeof expected === "number" && typeof actual === "number") {
    return;
  }

  expect({ path: currentPath, type: typeof actual }).toEqual({
    path: currentPath,
    type: typeof expected,
  });
}

export async function generateTraceJson<T>(
  jsonPath: string,
  testFn: () => Promise<T> | T,
): Promise<T> {
  traceManager.clearTraces();

  const result = await testFn();

  const traces = traceManager.getAllTraces();
  if (traces.length === 0) {
    throw new Error(
      "No traces were generated during the test function execution.",
    );
  }
  const actualDict = (traceManager as any).createTraceApi(traces[0]);

  fs.mkdirSync(path.dirname(jsonPath), { recursive: true });
  fs.writeFileSync(jsonPath, JSON.stringify(actualDict, null, 2), "utf-8");

  return result;
}

export async function assertTraceJson<T>(
  jsonPath: string,
  testFn: () => Promise<T> | T,
): Promise<T> {
  traceManager.clearTraces();

  const result = await testFn();

  const traces = traceManager.getAllTraces();
  if (traces.length === 0) {
    throw new Error(
      "No traces were generated during the test function execution.",
    );
  }

  const actualDict = (traceManager as any).createTraceApi(traces[0]);

  if (!fs.existsSync(jsonPath)) {
    throw new Error(
      `Expected trace JSON file not found at: ${jsonPath}. Run generateTraceJson first.`,
    );
  }

  const expectedDict = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

  // Assert using our ported structural checker
  assertJsonObjectStructure(expectedDict, actualDict);

  return result;
}
