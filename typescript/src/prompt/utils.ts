import { v4 as uuidv4 } from "uuid";
import {
  SchemaDataType,
  OutputSchemaField,
  OutputSchemaFieldSchema,
  OutputSchema,
  OutputSchemaSchema,
  StructuredSchemaField,
} from "./types";

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

export function outputSchemaToJsonSchema(
  schema: OutputSchema | null | undefined,
): Record<string, any> {
  if (!schema || !schema.fields || schema.fields.length === 0) {
    return {
      type: "object",
      properties: {},
      additionalProperties: false,
    };
  }

  OutputSchemaSchema.parse(schema);

  const childrenMap: Record<string, OutputSchemaField[]> = {};
  const rootFields: OutputSchemaField[] = [];

  for (const field of schema.fields) {
    if (field.parentId) {
      if (!childrenMap[field.parentId]) {
        childrenMap[field.parentId] = [];
      }
      childrenMap[field.parentId].push(field);
    } else {
      rootFields.push(field);
    }
  }

  function mapType(
    dtype: (typeof SchemaDataType)[keyof typeof SchemaDataType],
  ): string {
    const mapping: Record<string, string> = {
      STRING: "string",
      INTEGER: "integer",
      FLOAT: "number",
      BOOLEAN: "boolean",
      OBJECT: "object",
      ARRAY: "array",
      NULL: "null",
    };
    return mapping[dtype] || "string";
  }

  function buildNode(fieldList: OutputSchemaField[]): Record<string, any> {
    const properties: Record<string, any> = {};
    const requiredFields: string[] = [];

    for (const field of fieldList) {
      const fieldSchema: Record<string, any> = {
        type: mapType(field.type),
      };

      if (field.description) {
        fieldSchema.description = field.description;
      }

      if (field.type === "ARRAY") {
        const children = childrenMap[field.id] || [];
        if (children.length > 0) {
          const itemField = children[0];
          const itemSchema: Record<string, any> = {
            type: mapType(itemField.type),
          };
          if (itemField.type === "OBJECT") {
            const objChildren = childrenMap[itemField.id] || [];
            if (objChildren.length > 0) {
              const nested = buildNode(objChildren);
              itemSchema.properties = nested.properties;
              if (nested.required && nested.required.length > 0) {
                itemSchema.required = nested.required;
              }
              itemSchema.additionalProperties = false;
            } else {
              itemSchema.properties = {};
              itemSchema.additionalProperties = false;
            }
          }
          fieldSchema.items = itemSchema;
        } else {
          fieldSchema.items = {};
        }
      } else if (field.type === "OBJECT") {
        const children = childrenMap[field.id] || [];
        if (children.length > 0) {
          const nested = buildNode(children);
          fieldSchema.properties = nested.properties;
          if (nested.required && nested.required.length > 0) {
            fieldSchema.required = nested.required;
          }
          fieldSchema.additionalProperties = false;
        } else {
          fieldSchema.properties = {};
          fieldSchema.additionalProperties = false;
        }
      }

      properties[field.name] = fieldSchema;
      if (field.required) {
        requiredFields.push(field.name);
      }
    }

    const result: Record<string, any> = {
      type: "object",
      properties: properties,
      additionalProperties: false,
    };

    if (requiredFields.length > 0) {
      result.required = requiredFields;
    }

    return result;
  }

  return buildNode(rootFields);
}

export function fieldsToJsonSchema(
  fields: StructuredSchemaField[],
): Record<string, any> {
  const schema: OutputSchema = {
    fields: fields,
    name: null,
    id: null,
  };
  return outputSchemaToJsonSchema(schema);
}

export type { OutputSchemaField, OutputSchema, StructuredSchemaField };
