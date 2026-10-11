import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import {
  SchemaDataType,
  OutputSchemaField,
  OutputSchemaFieldSchema,
  OutputSchema,
} from "@/prompt/types";

export function interpolateMustache(
  text: string,
  kwargs: { [key: string]: any },
): string {
  return text.replace(/\{\{(\w+)\}\}/g, (match, variableName) => {
    return kwargs[variableName] !== undefined
      ? String(kwargs[variableName])
      : match;
  });
}

export function interpolateMustacheWithSpace(
  text: string,
  kwargs: { [key: string]: any },
): string {
  return text.replace(/\{\{ (\w+) \}\}/g, (match, variableName) => {
    return kwargs[variableName] !== undefined
      ? String(kwargs[variableName])
      : match;
  });
}

export function interpolateFString(
  text: string,
  kwargs: { [key: string]: any },
): string {
  return text.replace(/\{(\w+)\}/g, (match, variableName) => {
    return kwargs[variableName] !== undefined
      ? String(kwargs[variableName])
      : match;
  });
}

export function interpolateDollarBrackets(
  text: string,
  kwargs: { [key: string]: any },
): string {
  return text.replace(/\$\{(\w+)\}/g, (match, variableName) => {
    return kwargs[variableName] !== undefined
      ? String(kwargs[variableName])
      : match;
  });
}

export function interpolateText(
  interpolationType: string,
  text: string,
  kwargs: { [key: string]: any },
): string {
  switch (interpolationType) {
    case "MUSTACHE":
      return interpolateMustache(text, kwargs);
    case "MUSTACHE_WITH_SPACE":
      return interpolateMustacheWithSpace(text, kwargs);
    case "FSTRING":
      return interpolateFString(text, kwargs);
    case "DOLLAR_BRACKETS":
      return interpolateDollarBrackets(text, kwargs);
    default:
      throw new Error(`Unsupported interpolation type: ${interpolationType}`);
  }
}

// ==========================================
// Schema Generation
// ==========================================

export function generateOutputSchema(
  name: string,
  definition: Record<string, any>,
  parentId: string | null = null,
): OutputSchemaField[] {
  const fields: OutputSchemaField[] = [];

  for (const [key, value] of Object.entries(definition)) {
    const fieldId = uuidv4();
    let type: (typeof SchemaDataType)[keyof typeof SchemaDataType] =
      SchemaDataType.STRING;
    const required = true;

    if (typeof value === "string") {
      const lowerVal = value.toLowerCase();
      if (lowerVal === "string") type = SchemaDataType.STRING;
      else if (lowerVal === "integer") type = SchemaDataType.INTEGER;
      else if (lowerVal === "float" || lowerVal === "number")
        type = SchemaDataType.FLOAT;
      else if (lowerVal === "boolean") type = SchemaDataType.BOOLEAN;
    } else if (Array.isArray(value) && value.length === 1) {
      type = SchemaDataType.ARRAY;
    } else if (typeof value === "object" && value !== null) {
      type = SchemaDataType.OBJECT;
    }

    const field: OutputSchemaField = {
      id: fieldId,
      name: key,
      type: type,
      required: required,
      parentId: parentId,
    };

    fields.push(OutputSchemaFieldSchema.parse(field));

    if (type === SchemaDataType.ARRAY) {
      const itemDef = (value as any[])[0];
      const itemFieldId = uuidv4();
      if (typeof itemDef === "string") {
        const lowerItem = itemDef.toLowerCase();
        let itemType: (typeof SchemaDataType)[keyof typeof SchemaDataType] =
          SchemaDataType.STRING;
        if (lowerItem === "integer") itemType = SchemaDataType.INTEGER;
        else if (lowerItem === "float" || lowerItem === "number")
          itemType = SchemaDataType.FLOAT;
        else if (lowerItem === "boolean") itemType = SchemaDataType.BOOLEAN;
        fields.push(
          OutputSchemaFieldSchema.parse({
            id: itemFieldId,
            name: key,
            type: itemType,
            required: true,
            parentId: fieldId,
          }),
        );
      } else if (typeof itemDef === "object" && itemDef !== null) {
        fields.push(
          OutputSchemaFieldSchema.parse({
            id: itemFieldId,
            name: key,
            type: SchemaDataType.OBJECT,
            required: true,
            parentId: fieldId,
          }),
        );
        const nestedFields = generateOutputSchema("", itemDef, itemFieldId);
        fields.push(...nestedFields);
      }
    } else if (type === SchemaDataType.OBJECT) {
      const nestedFields = generateOutputSchema("", value, fieldId);
      fields.push(...nestedFields);
    }
  }
  return fields;
}

type JsonSchemaNode = Record<string, unknown>;

const TOOL_SCHEMA_KEYWORDS = new Set([
  "type",
  "description",
  "enum",
  "properties",
  "items",
  "required",
]);

const TOOL_SCHEMA_ANNOTATIONS = new Set([
  "$schema",
  "title",
  "default",
  "examples",
]);

const UNION_KEYWORDS = ["anyOf", "oneOf", "allOf"];

const isNullSchema = (node: JsonSchemaNode): boolean =>
  Object.keys(node).length === 1 && node.type === "null";

function inlineToolSchemaRef(
  node: JsonSchemaNode,
  defs: Record<string, JsonSchemaNode>,
  location: string,
  expanding: string[],
): [JsonSchemaNode, string[]] {
  const { $ref, ...rest } = node;
  const name = String($ref).split("/").pop() ?? "";
  if (expanding.includes(name)) {
    throw new Error(
      `${location} refers to '${name}' recursively, which a tool schema can't express.`,
    );
  }
  return [{ ...defs[name], ...rest }, [...expanding, name]];
}

function unwrapToolSchemaUnion(
  node: JsonSchemaNode,
  keyword: string,
  location: string,
): JsonSchemaNode {
  const { [keyword]: branches, ...rest } = node;
  const typed = (branches as JsonSchemaNode[]).filter(
    (branch) => !isNullSchema(branch),
  );
  if (typed.length !== 1) {
    throw new Error(
      `${location} is a union of several types. A tool schema supports only an optional or nullable field of one type.`,
    );
  }
  return { ...typed[0], ...rest };
}

function resolveToolSchemaNode(
  node: JsonSchemaNode,
  defs: Record<string, JsonSchemaNode>,
  location: string,
  expanding: string[],
): [JsonSchemaNode, string[]] {
  let resolved = node;
  let refs = expanding;
  for (;;) {
    if ("$ref" in resolved) {
      [resolved, refs] = inlineToolSchemaRef(resolved, defs, location, refs);
      continue;
    }
    const keyword = UNION_KEYWORDS.find((union) => union in resolved);
    if (!keyword) {
      return [resolved, refs];
    }
    resolved = unwrapToolSchemaUnion(resolved, keyword, location);
  }
}

function dropSafeIntegerBounds(node: JsonSchemaNode): JsonSchemaNode {
  const { minimum, maximum, ...rest } = node;
  return {
    ...rest,
    ...(minimum !== undefined && minimum !== Number.MIN_SAFE_INTEGER
      ? { minimum }
      : {}),
    ...(maximum !== undefined && maximum !== Number.MAX_SAFE_INTEGER
      ? { maximum }
      : {}),
  };
}

function normalizeToolSchemaNode(
  node: JsonSchemaNode,
  defs: Record<string, JsonSchemaNode>,
  path: string[],
  expanding: string[],
): JsonSchemaNode {
  const location = ["root", ...path].join(".");
  const [resolved, refs] = resolveToolSchemaNode(
    node,
    defs,
    location,
    expanding,
  );
  const {
    const: constValue,
    additionalProperties = false,
    ...rest
  } = dropSafeIntegerBounds(resolved);

  if (additionalProperties !== false) {
    throw new Error(
      `${location} is a free-form record. Describe its keys with z.object instead.`,
    );
  }

  const normalized: JsonSchemaNode = {
    ...Object.fromEntries(
      Object.entries(rest).filter(
        ([keyword]) => !TOOL_SCHEMA_ANNOTATIONS.has(keyword),
      ),
    ),
    ...(constValue !== undefined ? { enum: [constValue] } : {}),
  };

  const unsupported = Object.keys(normalized)
    .filter((keyword) => !TOOL_SCHEMA_KEYWORDS.has(keyword))
    .sort();
  if (unsupported.length > 0) {
    throw new Error(
      `${location} uses ${unsupported.join(", ")}, which a tool schema doesn't support. Use only ${[...TOOL_SCHEMA_KEYWORDS].sort().join(", ")}.`,
    );
  }
  if (!("type" in normalized)) {
    throw new Error(`${location} has no type. Give the field a concrete type.`);
  }

  if (normalized.properties) {
    normalized.properties = Object.fromEntries(
      Object.entries(
        normalized.properties as Record<string, JsonSchemaNode>,
      ).map(([name, child]) => [
        name,
        normalizeToolSchemaNode(child, defs, [...path, name], refs),
      ]),
    );
  }
  if (normalized.items) {
    normalized.items = normalizeToolSchemaNode(
      normalized.items as JsonSchemaNode,
      defs,
      [...path, "items"],
      refs,
    );
  }
  return normalized;
}

export function constructToolInputSchema(schema: z.ZodType): JsonSchemaNode {
  const { $defs = {}, ...root } = z.toJSONSchema(schema, {
    io: "input",
  }) as JsonSchemaNode;
  return normalizeToolSchemaNode(
    root,
    $defs as Record<string, JsonSchemaNode>,
    [],
    [],
  );
}

export type { OutputSchemaField, OutputSchema };
