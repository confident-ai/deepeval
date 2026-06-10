import { z } from "zod";
import { randomUUID } from "crypto";

export const SchemaDataTypeEnum = z.enum([
  "STRING",
  "INTEGER",
  "FLOAT",
  "BOOLEAN",
  "OBJECT",
  "ARRAY",
  "NULL",
]);

export type SchemaDataType = z.infer<typeof SchemaDataTypeEnum>;

export const SchemaDataType = {
  STRING: "STRING" as const,
  INTEGER: "INTEGER" as const,
  FLOAT: "FLOAT" as const,
  BOOLEAN: "BOOLEAN" as const,
  OBJECT: "OBJECT" as const,
  ARRAY: "ARRAY" as const,
  NULL: "NULL" as const,
};

export const ToolModeEnum = z.enum([
  "ALLOW_ADDITIONAL",
  "NO_ADDITIONAL",
  "STRICT",
]);

export type ToolMode = z.infer<typeof ToolModeEnum>;

export const ToolMode = {
  ALLOW_ADDITIONAL: "ALLOW_ADDITIONAL" as const,
  NO_ADDITIONAL: "NO_ADDITIONAL" as const,
  STRICT: "STRICT" as const,
};

export const OutputTypeEnum = z.enum(["TEXT", "JSON", "SCHEMA"]);

export type OutputType = z.infer<typeof OutputTypeEnum>;

export const OutputType = {
  TEXT: "TEXT" as const,
  JSON: "JSON" as const,
  SCHEMA: "SCHEMA" as const,
};

export const OutputSchemaFieldSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: SchemaDataTypeEnum,
  description: z.string().optional().nullable(),
  required: z.boolean(),
  parentId: z.string().optional().nullable(),
});

export type OutputSchemaField = z.infer<typeof OutputSchemaFieldSchema>;

export const OutputSchemaSchema = z.object({
  id: z.string().optional().nullable(),
  name: z.string().optional().nullable(),
  fields: z.array(OutputSchemaFieldSchema).optional().nullable(),
});

export type OutputSchema = z.infer<typeof OutputSchemaSchema>;

export type StructuredSchemaField = OutputSchemaField;

export const ModelSettingsSchema = z.object({
  provider: z.string().optional(),
  name: z.string().optional(),
  temperature: z.number().optional(),
  maxTokens: z.number().optional(),
  topP: z.number().optional(),
  frequencyPenalty: z.number().optional(),
  presencePenalty: z.number().optional(),
  stopSequence: z.array(z.string()).optional(),
  reasoningEffort: z.string().optional(),
  verbosity: z.string().optional(),
});

export type ModelSettings = z.infer<typeof ModelSettingsSchema>;

export const PromptMessageSchema = z.object({
  role: z.string(),
  content: z.string(),
});

export type PromptMessageType = z.infer<typeof PromptMessageSchema>;

export const SchemaDefinitionSchema = z.object({
  name: z.string(),
  fields: z.record(z.any(), z.any()),
});

export type SchemaDefinition = z.infer<typeof SchemaDefinitionSchema>;

export const ToolDataSchema = z.object({
  id: z
    .string()
    .optional()
    .default(() => randomUUID()),
  name: z.string(),
  description: z.string().optional().nullable(),
  mode: ToolModeEnum,
  structuredSchema: z.union([OutputSchemaSchema, SchemaDefinitionSchema]),
});

export type ToolData = z.infer<typeof ToolDataSchema>;

export const NormalizedToolDataSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  mode: ToolModeEnum,
  structuredSchema: OutputSchemaSchema,
});

export type NormalizedToolData = z.infer<typeof NormalizedToolDataSchema>;

export const PullOptionsSchema = z
  .object({
    version: z.string().optional(),
    label: z.string().optional(),
    hash: z.string().optional(),
    branch: z.string().optional(),
  })
  .optional();

export type PullOptions = z.infer<typeof PullOptionsSchema>;

export const CreateVersionOptionsSchema = z
  .object({
    commit: z.string().optional(),
  })
  .optional();

export type CreateVersionOptions = z.infer<typeof CreateVersionOptionsSchema>;

const PromptVersionSchema = z.object({
  id: z.string(),
  version: z.string(),
});

export const GetVersionsResponseSchema = z.object({
  data: z.object({
    textVersions: z.array(PromptVersionSchema).optional(),
    messagesVersions: z.array(PromptVersionSchema).optional(),
  }),
});

const PromptCommitSchema = z.object({
  id: z.string(),
  hash: z.string(),
  message: z.string(),
});

export const GetCommitsResponseSchema = z.object({
  data: z.object({
    commits: z.array(PromptCommitSchema),
  }),
});

export const CreateVersionResponseSchema = z.object({
  data: z.object({
    version: z.string(),
    hash: z.string(),
  }),
});

export const PushOptionsSchema = z
  .object({
    text: z.string().optional(),
    messages: z.array(PromptMessageSchema).optional(),
    interpolationType: z.string().optional(),
    version: z.string().optional(),
    label: z.string().optional(),
    modelSettings: ModelSettingsSchema.optional(),
    outputType: OutputTypeEnum.optional(),
    outputSchema: z
      .union([OutputSchemaSchema, SchemaDefinitionSchema])
      .optional(),
    tools: z.array(ToolDataSchema).optional(),
    branch: z.string().optional(),
  })
  .optional();

export type PushOptions = z.infer<typeof PushOptionsSchema>;

export const PromptResponseSchema = z.object({
  data: z.object({
    id: z.string().optional(),
    hash: z.string().optional(),
    version: z.string().optional().nullable(),
    label: z.string().optional().nullable(),
    text: z.string().optional().nullable(),
    messages: z.array(PromptMessageSchema).optional().nullable(),
    type: z.string().optional(),
    interpolationType: z.string().optional(),
    promptVersionId: z.string().optional(),
    modelSettings: ModelSettingsSchema.optional().nullable(),
    outputType: OutputTypeEnum.optional().nullable(),
    outputSchema: OutputSchemaSchema.optional().nullable(),
    tools: z.array(NormalizedToolDataSchema).optional().nullable(),
  }),
  link: z.string().optional().nullable(),
});

export type PromptResponse = z.infer<typeof PromptResponseSchema>;

export const PromptBranchSchema = z.object({
  id: z.string(),
  name: z.string(),
});

export type PromptBranch = z.infer<typeof PromptBranchSchema>;

export const GetBranchesResponseSchema = z.object({
  data: z.object({
    branches: z.array(PromptBranchSchema),
  }),
});
