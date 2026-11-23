/** Read OTel attribute values as the content confident-trace encodes in them. */

import type { Attributes } from "@opentelemetry/api";

import { ToolCall } from "@/test-case";

/** An attribute that isn't there, as opposed to one that holds null. */
export const MISSING: unique symbol = Symbol("missing");
export type Missing = typeof MISSING;

/** Parse a JSON string attribute; other values stay, invalid JSON is undefined. */
export function structuredValue(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

/** The first argument that isn't undefined. */
export function firstDefined<T>(...values: (T | undefined)[]): T | undefined {
  return values.find((value) => value !== undefined);
}

/**
 * Decode one JSON layer, so a JSON document sent as text stays text.
 *
 * confident-trace JSON-encodes every content field, plain strings included.
 */
export function attrJsonContent(
  attributes: Attributes,
  key: string,
): unknown | Missing {
  if (!(key in attributes)) return MISSING;
  const value = attributes[key];
  if (Array.isArray(value)) return [...value];
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

/** An attribute decoded once, or undefined; plain text stays text. */
export function attrDecoded(attributes: Attributes, key: string): unknown {
  const value = attrJsonContent(attributes, key);
  return value === MISSING ? undefined : value;
}

export function attrString(
  attributes: Attributes,
  key: string,
): string | undefined {
  const value = attributes[key];
  return typeof value === "string" && value !== "" ? value : undefined;
}

export function attrNumber(
  attributes: Attributes,
  key: string,
): number | undefined {
  const value = attributes[key];
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

export function attrStringList(
  attributes: Attributes,
  key: string,
): string[] | undefined {
  const value = attributes[key];
  return Array.isArray(value) ? value.map(String) : undefined;
}

/** A string list, sent natively or as JSON. */
export function attrCoercedStringList(
  attributes: Attributes,
  key: string,
): string[] | undefined {
  const value = firstDefined(
    attrStringList(attributes, key),
    structuredValue(attributes[key]) as string[] | undefined,
  );
  return Array.isArray(value) ? value.map(String) : undefined;
}

export function attrJsonObject(
  attributes: Attributes,
  key: string,
): Record<string, unknown> | undefined {
  const value = structuredValue(attributes[key]);
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

export function attrJsonStringList(
  attributes: Attributes,
  key: string,
): string[] | undefined {
  const value = attrDecoded(attributes, key);
  return Array.isArray(value) ? value.map(String) : undefined;
}

export function toolCallFrom(candidate: unknown): ToolCall | undefined {
  if (!candidate || typeof candidate !== "object") return undefined;
  const fields = candidate as Record<string, unknown>;
  if (typeof fields.name !== "string" || fields.name === "") return undefined;
  const parameters = fields.inputParameters ?? fields.input_parameters;
  return new ToolCall({
    name: fields.name,
    ...(typeof fields.description === "string"
      ? { description: fields.description }
      : {}),
    ...(parameters && typeof parameters === "object"
      ? { inputParameters: parameters as Record<string, any> }
      : {}),
    ...(fields.output !== undefined ? { output: fields.output } : {}),
  });
}

export function attrToolCalls(
  attributes: Attributes,
  key: string,
): ToolCall[] | undefined {
  const value = attrDecoded(attributes, key);
  if (!Array.isArray(value)) return undefined;
  const calls = value
    .map(toolCallFrom)
    .filter((call): call is ToolCall => call !== undefined);
  return calls.length > 0 ? calls : undefined;
}

// What confident-trace records in place of content it won't serialize.
const UNSUPPORTED = new Set(["[unsupported]", "[truncated]"]);

/** Whether confident-trace serialized none of `value`. */
export function unsupported(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0 && value.every(unsupported);
  if (value && typeof value === "object") {
    const values = Object.values(value);
    return values.length > 0 && values.every(unsupported);
  }
  return typeof value === "string" && UNSUPPORTED.has(value);
}
