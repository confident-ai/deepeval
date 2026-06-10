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

export class AmazonBedrockModel extends DeepEvalBaseLLM {
  private readonly region: string;
  private readonly awsAccessKeyId?: string;
  private readonly awsSecretAccessKey?: string;
  private readonly awsSessionToken?: string;
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
    this.awsAccessKeyId =
      options.awsAccessKeyId ?? process.env.AWS_ACCESS_KEY_ID;
    this.awsSecretAccessKey =
      options.awsSecretAccessKey ?? process.env.AWS_SECRET_ACCESS_KEY;
    this.awsSessionToken =
      options.awsSessionToken ?? process.env.AWS_SESSION_TOKEN;
    // Only sent when explicitly set — some models (e.g. reasoning models) reject `temperature`.
    this.temperature = options.temperature;
    this.costPerInputToken = options.costPerInputToken;
    this.costPerOutputToken = options.costPerOutputToken;
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
      const credentials =
        this.awsAccessKeyId && this.awsSecretAccessKey
          ? {
              accessKeyId: this.awsAccessKeyId,
              secretAccessKey: this.awsSecretAccessKey,
              sessionToken: this.awsSessionToken,
            }
          : undefined;
      this.client = new BedrockRuntimeClient({
        region: this.region,
        ...(credentials ? { credentials } : {}),
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
    return true;
  }
}
