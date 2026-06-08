import { Api, Endpoints, HttpMethods } from "../confident/api";
import {
  interpolateText,
  outputSchemaToJsonSchema,
  generateOutputSchema,
} from "./utils";
import {
  ToolMode,
  OutputType,
  ModelSettings,
  PromptMessageType,
  PromptMessageSchema,
  SchemaDefinition,
  SchemaDefinitionSchema,
  ToolData,
  ToolDataSchema,
  NormalizedToolData,
  NormalizedToolDataSchema,
  OutputSchema,
  OutputSchemaSchema,
  PullOptions,
  PullOptionsSchema,
  PushOptions,
  PushOptionsSchema,
  CreateVersionOptions,
  CreateVersionOptionsSchema,
  GetVersionsResponseSchema,
  GetCommitsResponseSchema,
  CreateVersionResponseSchema,
  PromptResponseSchema,
} from "./types";

export { ToolMode, OutputType, type ModelSettings, type SchemaDefinition };

export class PromptMessage {
  role: string;
  content: string;

  constructor(params: { role: string; content: string }) {
    const validated = PromptMessageSchema.parse(params);
    this.role = validated.role;
    this.content = validated.content;
  }
}

export class Tool {
  private _data: NormalizedToolData;

  constructor(data: ToolData) {
    const validatedInput = ToolDataSchema.parse(data);

    let normalizedSchema: OutputSchema;

    if (
      "fields" in validatedInput.structuredSchema &&
      Array.isArray(validatedInput.structuredSchema.fields)
    ) {
      normalizedSchema = OutputSchemaSchema.parse(
        validatedInput.structuredSchema,
      );
    } else {
      const schemaDef = SchemaDefinitionSchema.parse(
        validatedInput.structuredSchema,
      );
      normalizedSchema = {
        name: schemaDef.name,
        fields: generateOutputSchema(schemaDef.name, schemaDef.fields),
      };
    }

    this._data = NormalizedToolDataSchema.parse({
      id: validatedInput.id,
      name: validatedInput.name,
      description: validatedInput.description,
      mode: validatedInput.mode,
      structuredSchema: normalizedSchema,
    });
  }

  public get id(): string {
    return this._data.id;
  }

  public get name(): string {
    return this._data.name;
  }

  public get description(): string | null | undefined {
    return this._data.description;
  }

  public get mode(): ToolMode {
    return this._data.mode;
  }

  public get structuredSchema(): OutputSchema {
    return this._data.structuredSchema;
  }

  public get inputSchema(): Record<string, any> {
    return outputSchemaToJsonSchema(this._data.structuredSchema);
  }

  public toJSON(): NormalizedToolData {
    return this._data;
  }
}

export class Prompt {
  private alias: string;
  private api: Api;

  // Class variables to store prompt data
  public hash: string = "latest";
  public branch: string | undefined;
  private _version: string | null = null;
  private _label: string | null = null;
  private _textTemplate: string | null = null;
  private _messagesTemplate: PromptMessageType[] | null = null;
  private _promptVersionId: string | null = null;
  private _type: string | null = null;
  private _interpolationType: string | null = null;
  private _tools: Tool[] | null = null;
  private _modelSettings: ModelSettings | null = null;
  private _outputType: OutputType | null = null;
  private _outputSchema: OutputSchema | null = null;

  constructor({
    alias,
    branch,
    confidentApiKey,
  }: {
    alias: string;
    branch?: string;
    confidentApiKey?: string;
  }) {
    if (!alias || typeof alias !== "string") {
      throw new Error("Alias must be a non-empty string");
    }
    this.alias = alias;
    this.branch = branch;
    this.api = new Api(confidentApiKey);
  }

  private _normalizeTools(toolsInput?: ToolData[] | Tool[]): Tool[] | null {
    if (!toolsInput || !Array.isArray(toolsInput)) {
      return null;
    }

    return toolsInput.map((t) => {
      if (t instanceof Tool) {
        return t;
      } else {
        return new Tool(t);
      }
    });
  }

  public async pull(options?: PullOptions): Promise<any> {
    const validatedOptions = PullOptionsSchema.parse(options);

    let endpoint: string;

    if (validatedOptions?.label) {
      endpoint = Endpoints.PROMPTS_LABEL_ENDPOINT.replace(
        ":alias",
        this.alias,
      ).replace(":label", validatedOptions.label);
    } else if (validatedOptions?.version) {
      endpoint = Endpoints.PROMPTS_VERSION_ID_ENDPOINT.replace(
        ":alias",
        this.alias,
      ).replace(":version", validatedOptions?.version);
    } else {
      endpoint = Endpoints.PROMPTS_COMMIT_HASH_ENDPOINT.replace(
        ":alias",
        this.alias,
      ).replace(":hash", validatedOptions?.hash || "latest");
    }

    const queryParams = { name: validatedOptions?.branch || this.branch };

    const response = await this.api.sendRequest(
      HttpMethods.GET,
      endpoint,
      undefined,
      queryParams,
    );

    // Validate response
    const validatedResponse = PromptResponseSchema.parse(response);

    // Extract and assign values from the response
    this.hash =
      validatedResponse.data.hash || validatedOptions?.hash || "latest";
    this._version =
      validatedResponse.data.version || validatedOptions?.version || null;
    this._label =
      validatedResponse.data.label || validatedOptions?.label || null;
    this._textTemplate = validatedResponse.data.text || null;
    this._messagesTemplate = validatedResponse.data.messages || null;
    this._promptVersionId = validatedResponse.data.promptVersionId || null;
    this._type = validatedResponse.data.type || null;
    this._interpolationType = validatedResponse.data.interpolationType || null;
    this._modelSettings = validatedResponse.data.modelSettings || null;
    this._outputType = validatedResponse.data.outputType || null;
    this._outputSchema = validatedResponse.data.outputSchema || null;

    if (
      validatedResponse.data.tools &&
      Array.isArray(validatedResponse.data.tools)
    ) {
      this._tools = validatedResponse.data.tools.map(
        (t: NormalizedToolData) => {
          // Convert NormalizedToolData back to Tool
          return new Tool({
            id: t.id,
            name: t.name,
            description: t.description,
            mode: t.mode,
            structuredSchema: t.structuredSchema,
          });
        },
      );
    } else {
      this._tools = null;
    }

    return validatedResponse;
  }

  public async createVersion(options?: CreateVersionOptions): Promise<any> {
    const validatedOptions = CreateVersionOptionsSchema.parse(options);
    const commitHash = validatedOptions?.commit || this.hash || "latest";

    // Use the release endpoint
    const endpoint = Endpoints.PROMPTS_VERSIONS_ENDPOINT.replace(
      ":alias",
      this.alias,
    );

    const body = {
      hash: commitHash,
    };

    const response = await this.api.sendRequest(
      HttpMethods.POST,
      endpoint,
      body,
      undefined,
    );

    const validatedResponse = CreateVersionResponseSchema.parse(response);

    if (validatedResponse.data.version) {
      this._version = response.version;
      console.log(
        `✅ Version ${validatedResponse.data.version} created successfully!`,
      );
    }

    return validatedResponse;
  }

  public async getCommits(branch?: string): Promise<any[]> {
    const params = branch ? { branch } : undefined;
    const response = await this.api.sendRequest(
      HttpMethods.GET,
      Endpoints.PROMPTS_COMMITS_ENDPOINT,
      undefined,
      params,
      undefined,
      { alias: this.alias },
    );

    const validatedResponse = GetCommitsResponseSchema.parse(response);

    return validatedResponse.data.commits || [];
  }

  public async getVersions(): Promise<any> {
    const response = await this.api.sendRequest(
      HttpMethods.GET,
      Endpoints.PROMPTS_VERSIONS_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias: this.alias },
    );

    const validatedResponse = GetVersionsResponseSchema.parse(response);

    if (
      validatedResponse.data.messagesVersions ||
      validatedResponse.data.textVersions
    ) {
      return response.data.textVersions || response.data.messagesVersions || [];
    }

    return validatedResponse;
  }

  public async push(options?: PushOptions): Promise<any> {
    if (!this.alias || !this.alias.trim()) {
      throw new Error(
        "Prompt alias is not set or is empty. Please set an alias to continue.",
      );
    }

    const validatedOptions = PushOptionsSchema.parse(options);

    const text = validatedOptions?.text || this._textTemplate;
    const messages = validatedOptions?.messages || this._messagesTemplate;
    const interpolationType =
      validatedOptions?.interpolationType ||
      this._interpolationType ||
      "FSTRING";

    const toolsToUse = validatedOptions?.tools
      ? this._normalizeTools(validatedOptions.tools)
      : this._tools;

    if (text && messages) {
      throw new TypeError(
        "Cannot push prompt with both 'text' and 'messages'. Please provide only one.",
      );
    }

    let formattedOutputSchema: OutputSchema | undefined = undefined;
    if (validatedOptions?.outputSchema) {
      if (
        "fields" in validatedOptions.outputSchema &&
        Array.isArray(validatedOptions.outputSchema.fields)
      ) {
        formattedOutputSchema = OutputSchemaSchema.parse(
          validatedOptions.outputSchema,
        );
      } else {
        const schemaDef = SchemaDefinitionSchema.parse(
          validatedOptions.outputSchema,
        );
        formattedOutputSchema = {
          name: schemaDef.name,
          fields: generateOutputSchema(schemaDef.name, schemaDef.fields),
        };
      }
    }

    const body = {
      alias: this.alias,
      text: text || undefined,
      messages: messages || undefined,
      interpolationType: interpolationType,
      modelSettings: validatedOptions?.modelSettings,
      outputType: validatedOptions?.outputType,
      outputSchema: formattedOutputSchema,
      tools: toolsToUse?.map((t) => t.toJSON()) || undefined,
      branch: validatedOptions?.branch || this.branch,
    };

    console.log(`Pushing '${this.alias}' prompt to Confident AI...`);

    const response = await this.api.sendRequest(
      HttpMethods.POST,
      Endpoints.PROMPTS_ENDPOINT,
      body,
      undefined,
    );

    const validatedResponse = PromptResponseSchema.parse(response);

    if (validatedResponse.data) {
      this.hash = validatedResponse.data.hash || "latest";
      this._textTemplate = text;
      this._messagesTemplate = messages;
      this._interpolationType = interpolationType;
      this._type = text ? "TEXT" : "LIST";
      this._modelSettings = validatedOptions?.modelSettings || null;
      this._outputType = validatedOptions?.outputType || null;
      this._outputSchema = formattedOutputSchema || null;
      this._tools = toolsToUse;
    }

    const link = validatedResponse.link;
    if (link) {
      console.log(
        `✅ Prompt successfully pushed to Confident AI! View at: ${link}`,
      );
    }

    return validatedResponse;
  }

  // ==========================================
  // Branching Methods
  // ==========================================

  public async getBranches(): Promise<{ id: string; name: string }[]> {
    if (!this.alias) {
      throw new Error(
        "Prompt alias is not set. Please set an alias to continue.",
      );
    }

    const response = await this.api.sendRequest(
      HttpMethods.GET,
      Endpoints.PROMPTS_BRANCHES_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias: this.alias },
    );

    return response?.data?.branches || [];
  }

  public async createBranch(branch: string): Promise<void> {
    if (!this.alias) {
      throw new Error(
        "Prompt alias is not set. Please set an alias to continue.",
      );
    }

    const body = { branch };

    await this.api.sendRequest(
      HttpMethods.POST,
      Endpoints.PROMPTS_BRANCHES_ENDPOINT,
      body,
      undefined,
      undefined,
      { alias: this.alias },
    );

    this.branch = branch;

    console.log(`✅ Prompt branch '${branch}' successfully created!`);
  }

  public async updateBranch(name: string, branch?: string): Promise<void> {
    if (!this.alias) {
      throw new Error(
        "Prompt alias is not set. Please set an alias to continue.",
      );
    }

    const branchToUpdate = branch || this.branch;
    if (branchToUpdate === "main" || !branchToUpdate) {
      throw new Error(
        "Cannot update the name of the main branch or unspecified branch. Please pass a branch which is not 'main'.",
      );
    }

    const body = { name };

    await this.api.sendRequest(
      HttpMethods.PUT,
      Endpoints.PROMPTS_BRANCH_ENDPOINT,
      body,
      undefined,
      undefined,
      { alias: this.alias, name: branchToUpdate },
    );

    if (branchToUpdate === this.branch) {
      this.branch = name;
    }

    console.log(
      `✅ Successfully renamed branch '${branchToUpdate}' to '${name}'.`,
    );
  }

  public async deleteBranch(branch?: string): Promise<void> {
    if (!this.alias) {
      throw new Error(
        "Prompt alias is not set. Please set an alias to continue.",
      );
    }

    const branchToDelete = branch || this.branch;
    if (branchToDelete === "main" || !branchToDelete) {
      throw new Error(
        "Cannot delete the main branch or unspecified branch. Please pass a branch which is not 'main'.",
      );
    }

    await this.api.sendRequest(
      HttpMethods.DELETE,
      Endpoints.PROMPTS_BRANCH_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias: this.alias, name: branchToDelete },
    );

    if (branchToDelete === this.branch) {
      this.branch = "main";
    }

    console.log(`✅ Successfully deleted branch '${branchToDelete}'.`);
  }

  // Getter methods for accessing the private variables

  public get version(): string | null {
    return this._version;
  }

  public set version(version: string) {
    this._version = version;
  }

  public get textTemplate(): string | null {
    return this._textTemplate;
  }

  public get messagesTemplate(): PromptMessageType[] | null {
    return this._messagesTemplate;
  }

  public get promptVersionId(): string | null {
    return this._promptVersionId;
  }

  public get type(): string | null {
    return this._type;
  }

  public get interpolationType(): string | null {
    return this._interpolationType;
  }

  public get tools(): Tool[] | null {
    return this._tools;
  }

  public get modelSettings(): ModelSettings | null {
    return this._modelSettings;
  }

  public get outputType(): OutputType | null {
    return this._outputType;
  }

  public get outputSchema(): OutputSchema | null {
    return this._outputSchema;
  }

  public get label(): string | null {
    return this._label;
  }

  public set label(label: string) {
    this._label = label;
  }

  public get _alias(): string | null {
    return this.alias;
  }

  public interpolate(kwargs?: {
    [key: string]: any;
  }): string | PromptMessageType[] {
    kwargs = kwargs || {};
    if (this._type === "TEXT") {
      if (this._textTemplate === null) {
        throw new TypeError(
          "Unable to interpolate empty prompt template. Please pull a prompt from Confident AI or set template manually to continue.",
        );
      }
      return interpolateText(
        this._interpolationType as string,
        this._textTemplate,
        kwargs,
      );
    } else if (this._type === "LIST") {
      if (this._messagesTemplate === null) {
        throw new TypeError(
          "Unable to interpolate empty prompt template messages. Please pull a prompt from Confident AI or set template manually to continue.",
        );
      }
      const interpolatedMessages: PromptMessageType[] = [];
      for (const message of this._messagesTemplate) {
        const interpolatedContent = interpolateText(
          this._interpolationType as string,
          message.content,
          kwargs,
        );
        interpolatedMessages.push({
          role: message.role,
          content: interpolatedContent,
        });
      }
      return interpolatedMessages;
    } else {
      throw new Error(`Unsupported prompt type: ${this._type}`);
    }
  }
}
