// Port of `deepeval/telemetry/judge.py`. The provider is the model's own class
// name, bounded by checking the class against the ones this package exports;
// the model name is checked against the generated registry. An unfiltered
// `modelName` is an unbounded-cardinality leak -- a custom subclass can set it
// to anything.

import { GENERATED_MODEL_DATA } from "@/models/registry";
import { CUSTOM_PROVIDER, UNKNOWN_MODEL } from "@/telemetry/properties";

// Pooled across namespaces: the check only proves a name is one of ours.
let knownModelNames: Set<string> | null = null;

function isKnownModelName(name: unknown): string {
  if (typeof name !== "string" || name === "") return UNKNOWN_MODEL;
  if (knownModelNames === null) {
    knownModelNames = new Set(
      Object.values(GENERATED_MODEL_DATA).flatMap((models) =>
        Object.keys(models),
      ),
    );
  }
  return knownModelNames.has(name) ? name : UNKNOWN_MODEL;
}

/**
 * The classes this package exports, by constructor identity, so a new provider
 * needs only an export from `@/models`. Required lazily: importing the barrel
 * at module scope would pull every vendor SDK into every traced application.
 */
let shippedClasses: Set<unknown> | null = null;

function isShippedClass(modelClass: unknown): boolean {
  if (shippedClasses === null) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const models = require("../models") as Record<string, unknown>;
      shippedClasses = new Set(
        Object.values(models).filter((value) => typeof value === "function"),
      );
    } catch {
      shippedClasses = new Set();
    }
  }
  return shippedClasses.has(modelClass);
}

export interface JudgeDescription {
  provider?: string;
  model?: string;
}

export function describeJudge(model: unknown): JudgeDescription {
  if (model === null || model === undefined) return {};
  try {
    const modelClass = (model as object).constructor;
    if (!isShippedClass(modelClass)) {
      return { provider: CUSTOM_PROVIDER, model: UNKNOWN_MODEL };
    }
    return {
      provider: modelClass.name,
      model: isKnownModelName((model as { modelName?: unknown }).modelName),
    };
  } catch {
    return { provider: CUSTOM_PROVIDER, model: UNKNOWN_MODEL };
  }
}
