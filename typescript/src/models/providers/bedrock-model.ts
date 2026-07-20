import type { ZodType } from "zod";
import { DeepEvalBaseLLM, type GenerationResult } from "../base-model";
import { computeCost, extractJson, importOptional } from "../utils";

const DEFAULT_BEDROCK_REGION = "us-east-1";

export interface AmazonBedrockModelOptions {
  model?: string;
  region?: string;
  awsAccessKeyId?: string;
  awsSecretAccessKey?: string;
  awsSessionToken?: string;
  temperature?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

type BedrockCredentials = {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
};

export class AmazonBedrockModel extends DeepEvalBaseLLM {
  private readonly region: string;
  private readonly credentials?: BedrockCredentials;
  private readonly temperature?: number;
  private readonly costPerInputToken?: number;
  private readonly costPerOutputToken?: number;
  private sdk?: any;
  private client?: any;

  constructor(options: AmazonBedrockModelOptions = {}) {
    super(options.model ?? process.env.AWS_BEDROCK_MODEL_NAME);
    this.region =
      options.region ??
      process.env.AWS_BEDROCK_REGION ??
      process.env.AWS_REGION ??
      DEFAULT_BEDROCK_REGION;
    this.credentials = AmazonBedrockModel.resolveCredentials(options);
    // Only sent when explicitly set — some models (e.g. reasoning models) reject `temperature`.
    this.temperature = options.temperature;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
  }

  private static resolveCredentials(
    options: AmazonBedrockModelOptions,
  ): BedrockCredentials | undefined {
    const hasExplicitCredential =
      options.awsAccessKeyId !== undefined ||
      options.awsSecretAccessKey !== undefined ||
      options.awsSessionToken !== undefined;

    if (hasExplicitCredential) {
      if (!options.awsAccessKeyId || !options.awsSecretAccessKey) {
        throw new Error(
          "Amazon Bedrock explicit credentials require both `awsAccessKeyId` and `awsSecretAccessKey` when any credential option is provided.",
        );
      }
      return {
        accessKeyId: options.awsAccessKeyId,
        secretAccessKey: options.awsSecretAccessKey,
        ...(options.awsSessionToken
          ? { sessionToken: options.awsSessionToken }
          : {}),
      };
    }

    const envAccessKeyId = process.env.AWS_ACCESS_KEY_ID;
    const envSecretAccessKey = process.env.AWS_SECRET_ACCESS_KEY;
    const envSessionToken = process.env.AWS_SESSION_TOKEN;
    const hasEnvCredential =
      envAccessKeyId !== undefined ||
      envSecretAccessKey !== undefined ||
      envSessionToken !== undefined;

    if (!hasEnvCredential) {
      return undefined;
    }

    if (!envAccessKeyId || !envSecretAccessKey) {
      throw new Error(
        "Amazon Bedrock environment credentials require both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY when any AWS credential environment variable is set.",
      );
    }

    return {
      accessKeyId: envAccessKeyId,
      secretAccessKey: envSecretAccessKey,
      ...(envSessionToken ? { sessionToken: envSessionToken } : {}),
    };
  }

  private async getSdk(): Promise<any> {
    if (!this.sdk) {
      this.sdk = await importOptional(
        "@aws-sdk/client-bedrock-runtime",
        "Amazon Bedrock",
      );
    }
    return this.sdk;
  }

  private async getClient(): Promise<any> {
    if (!this.client) {
      const { BedrockRuntimeClient } = await this.getSdk();
      this.client = new BedrockRuntimeClient({
        region: this.region,
        ...(this.credentials ? { credentials: this.credentials } : {}),
      });
    }
    return this.client;
  }

  async generate<T = string>(
    prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    if (!this.modelName) {
      throw new Error(
        "Amazon Bedrock requires a model id. Pass `model` or set AWS_BEDROCK_MODEL_NAME.",
      );
    }
    const client = await this.getClient();
    const { ConverseCommand } = await this.getSdk();

    const response = await client.send(
      new ConverseCommand({
        modelId: this.modelName,
        messages: [{ role: "user", content: [{ text: prompt }] }],
        ...(this.temperature !== undefined && {
          inferenceConfig: { temperature: this.temperature },
        }),
      }),
    );

    const blocks: any[] = response.output?.message?.content ?? [];
    const text: string = blocks.map((block) => block.text ?? "").join("");
    const cost = computeCost(
      response.usage?.inputTokens,
      response.usage?.outputTokens,
      this.costPerInputToken,
      this.costPerOutputToken,
    );

    if (schema) {
      return { output: schema.parse(extractJson(text)), cost };
    }
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? "amazon-bedrock";
  }

  supportsMultimodal(): boolean {
    return false;
  }
}
