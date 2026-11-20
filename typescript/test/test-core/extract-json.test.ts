// extractJson recovers the JSON object from a model's raw reply. Mirrors
// Python's tests/test_core/test_trim_and_load_json.py: the trailing-comma
// cleanup is only a fallback, so it must never rewrite a reply that parses.

import { extractJson } from "@/models/utils";

describe("extractJson", () => {
  it.each([
    '{"verdict": "no", "reason": "The list [a, b, ] has a trailing comma"}',
    '{"output": "items = [1, 2, ]"}',
    '{"note": "the set is {x, y, } here"}',
  ])("leaves valid JSON string values unchanged: %s", (raw) => {
    expect(extractJson(raw)).toEqual(JSON.parse(raw));
  });

  it("still strips trailing commas when the reply does not parse", () => {
    expect(extractJson('{"a": [1, 2,], }')).toEqual({ a: [1, 2] });
  });

  it("slices the JSON object out of surrounding prose", () => {
    expect(
      extractJson('Sure, here is the verdict:\n{"score": 7}\nHope that helps.'),
    ).toEqual({ score: 7 });
  });

  it("throws when the reply has no JSON object", () => {
    expect(() => extractJson("no json here")).toThrow(
      "did not return valid JSON",
    );
  });

  it("throws when the reply is still invalid after the cleanup", () => {
    expect(() => extractJson('{"a": }')).toThrow(SyntaxError);
  });
});
