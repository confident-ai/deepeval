import { z, type ZodType } from "zod";

export function computeCost(
  inputTokens: number | null | undefined,
  outputTokens: number | null | undefined,
  costPerInputToken?: number,
  costPerOutputToken?: number,
): number | null {
  if (costPerInputToken == null || costPerOutputToken == null) {
    return null;
  }
  return (
    (inputTokens ?? 0) * costPerInputToken +
    (outputTokens ?? 0) * costPerOutputToken
  );
}

export function extractJson(text: string): unknown {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) {
    throw new Error(
      "The evaluation model did not return valid JSON. Please use a more capable evaluation model.",
    );
  }
  const candidate = text.slice(start, end + 1).replace(/,\s*([}\]])/g, "$1");
  return JSON.parse(candidate);
}

export function toJsonSchema(schema: ZodType): Record<string, unknown> {
  return z.toJSONSchema(schema, {
    unrepresentable: "any",
  }) as Record<string, unknown>;
}

export function requireApiKey(
  key: string | undefined,
  provider: string,
  envVar: string,
): string {
  if (!key) {
    throw new Error(
      `${provider} API key is not configured. Set ${envVar} in your environment or pass \`apiKey\`.`,
    );
  }
  return key;
}

export async function importOptional(
  pkg: string,
  provider: string,
): Promise<any> {
  try {
    return await import(pkg);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    if (code === "MODULE_NOT_FOUND" || code === "ERR_MODULE_NOT_FOUND") {
      throw new Error(
        `The '${pkg}' package is required to use ${provider} with deepeval. Install it with: npm install ${pkg}`,
      );
    }
    throw err;
  }
}
