/**
 * Language-aware inline code terms.
 *
 * Each entry pairs the Python and TypeScript spelling of a single inline
 * code identifier, keyed by a `namespace::localName` id. By deliberate
 * convention the local part mirrors the Python spelling (Python is the
 * canonical source language for these docs) and the `namespace::` prefix
 * categorizes the id and signals "this is a key, not a raw value". The TS
 * form is always an explicit value here — never derived from Python at
 * runtime, because the differences aren't just casing.
 *
 * Scope: single-token inline identifiers in prose only. Anything with
 * parens, operators, dotted access, or assignment stays a raw code span,
 * as do fenced code blocks, CLI commands, and external contract strings
 * (env vars, headers, URLs) which are identical across languages.
 *
 * TypeScript values are placeholders/best-guess for now — TS does not yet
 * have full parity with Python, so they are not guaranteed correct.
 */

export type Language = "python" | "typescript";

export interface Term {
  python: string;
  typescript: string;
}

export const TERMS = {
  // modules / import paths
  "module::deepeval": { python: "deepeval", typescript: "deepeval" }, // self-map
  "module::deepeval.metrics": {
    python: "deepeval.metrics",
    typescript: "deepeval",
  }, // differs, not casing

  // functions (self-map: same in both — documents intent, keeps fail-loud meaningful)
  "fn::evaluate": { python: "evaluate", typescript: "evaluate" },

  // class names (often identical -> self-map)
  "class::LLMTestCase": { python: "LLMTestCase", typescript: "LLMTestCase" },
  "class::ConversationalTestCase": {
    python: "ConversationalTestCase",
    typescript: "ConversationalTestCase",
  },
  "class::ToolCall": { python: "ToolCall", typescript: "ToolCall" },

  // variables / params / fields (differ by naming convention, NOT derived)
  "var::test_case": { python: "test_case", typescript: "testCase" },
  "field::test_cases": { python: "test_cases", typescript: "testCases" }, // plural is a distinct key from var::test_case
  "field::actual_output": {
    python: "actual_output",
    typescript: "actualOutput",
  },
  "field::expected_output": {
    python: "expected_output",
    typescript: "expectedOutput",
  },
  "field::retrieval_context": {
    python: "retrieval_context",
    typescript: "retrievalContext",
  },
  "field::tools_called": { python: "tools_called", typescript: "toolsCalled" },

  // literals
  "literal::True": { python: "True", typescript: "true" },
  "literal::None": { python: "None", typescript: "null" },

  // file names (extension swap)
  "file::test_example.py": {
    python: "test_example.py",
    typescript: "test_example.ts",
  },
} as const satisfies Record<string, Term>;

export type TermId = keyof typeof TERMS;

/**
 * Resolve a term id to its spelling in the given language.
 *
 * Fails loud on unknown ids: this throws during static generation (page
 * SSG and the `/llms.*` routes), turning a typo into a hard `next build`
 * error rather than a silently-dropped term.
 */
export function getTerm(id: string, lang: Language): string {
  const entry = (TERMS as Record<string, Term>)[id];
  if (!entry) {
    throw new Error(
      `[lang-terms] Unknown inline term id: "${id}". Add it to lib/lang/terms.ts or use a raw code span.`,
    );
  }
  return entry[lang];
}
