import type { DeepEvalBaseLLM } from "@/models";

export interface ExpectationsOptions {
  must?: string[];
  mustNot?: string[];
  model?: string | DeepEvalBaseLLM | null;
  evalMode?: "llm" | "system_one" | null;
}

/** Required and prohibited behavior for a case or entire conversation. */
export class Expectations {
  must: string[];
  mustNot: string[];
  model?: string | DeepEvalBaseLLM | null;
  evalMode?: "llm" | "system_one" | null;

  constructor(options: ExpectationsOptions = {}) {
    for (const key of Object.keys(options)) {
      if (!["must", "mustNot", "model", "evalMode"].includes(key)) {
        throw new Error(`Unknown Expectations parameter: ${key}`);
      }
    }
    const conditions = (value: string[] = []) => {
      if (
        !Array.isArray(value) ||
        value.some((s) => typeof s !== "string" || !s.trim())
      ) {
        throw new Error("Expectations must contain non-empty conditions.");
      }
      return value.map((s) => s.trim());
    };
    this.must = conditions(options.must);
    this.mustNot = conditions(options.mustNot);
    if (
      options.evalMode != null &&
      !["llm", "system_one"].includes(options.evalMode)
    ) {
      throw new Error("Expectations evalMode must be llm or system_one.");
    }
    if (
      options.model != null &&
      typeof options.model !== "string" &&
      (typeof options.model.generate !== "function" ||
        typeof options.model.getModelName !== "function")
    ) {
      throw new Error("model must be a model name or DeepEvalBaseLLM instance");
    }
    this.model = options.model;
    this.evalMode = options.evalMode;
  }

  get hasConditions(): boolean {
    return this.must.length + this.mustNot.length > 0;
  }

  toJSON(): ExpectationsOptions {
    return {
      must: this.must,
      mustNot: this.mustNot,
      model: typeof this.model === "string" ? this.model : null,
      evalMode: this.evalMode ?? null,
    };
  }
}

export function resolveExpectations(
  value?: Expectations | ExpectationsOptions | null,
): Expectations | undefined {
  return value == null
    ? undefined
    : value instanceof Expectations
      ? value
      : new Expectations(value);
}
