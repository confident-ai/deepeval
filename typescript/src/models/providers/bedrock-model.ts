import type { ZodType } from "zod";
import {
  DeepEvalBaseLLM,
  type ExtraGenerationParams,
  type GenerationResult,
} from "@/models/base-model";
import { extractJson, importOptional } from "@/models/utils";
import { bedrockContent } from "@/models/multimodal";
import type { ModelNamespace } from "@/models/registry";

const DEFAULT_BEDROCK_REGION = "us-east-1";

/** Any other key is merged into the Converse `inferenceConfig` (e.g. `topP`, `maxTokens`). */
export interface AmazonBedrockModelOptions extends ExtraGenerationParams {
  model?: string;
  region?: string;
  awsAccessKeyId?: string;
  awsSecretAccessKey?: string;
  awsSessionToken?: string;
  /** Defaults to `0`. Pass `null` to omit it from the request entirely. */
  temperature?: number | null;
  costPerInputToken?: number;
  costPerOutputToken?: number;
}

export class AmazonBedrockModel extends DeepEvalBaseLLM {
  private readonly region: string;
  private readonly awsAccessKeyId?: string;
  private readonly awsSecretAccessKey?: string;
  private readonly awsSessionToken?: string;
  private readonly extraParams: ExtraGenerationParams;
  private sdk?: any;
  protected registryNamespace: ModelNamespace = "bedrock";
  private client?: any;

  constructor(options: AmazonBedrockModelOptions = {}) {
    const {
      model,
      region,
      awsAccessKeyId,
      awsSecretAccessKey,
      awsSessionToken,
      temperature,
      costPerInputToken,
      costPerOutputToken,
      ...extraParams
    } = options;

    super(model ?? process.env.AWS_BEDROCK_MODEL_NAME);
    this.region =
      region ??
      process.env.AWS_BEDROCK_REGION ??
      process.env.AWS_REGION ??
      DEFAULT_BEDROCK_REGION;
    this.awsAccessKeyId = awsAccessKeyId ?? process.env.AWS_ACCESS_KEY_ID;
    this.awsSecretAccessKey =
      awsSecretAccessKey ?? process.env.AWS_SECRET_ACCESS_KEY;
    this.awsSessionToken = awsSessionToken ?? process.env.AWS_SESSION_TOKEN;
    this.temperature = temperature;
    this.costPerInputToken = costPerInputToken;
    this.costPerOutputToken = costPerOutputToken;
    this.extraParams = extraParams;
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

    const temperature = this.resolveTemperature();
    const inferenceConfig: Record<string, unknown> = {
      ...(temperature !== undefined && { temperature }),
      ...this.extraParams,
    };

    const response = await client.send(
      new ConverseCommand({
        modelId: this.modelName,
        messages: [{ role: "user", content: await bedrockContent(prompt) }],
        ...(Object.keys(inferenceConfig).length > 0 && { inferenceConfig }),
      }),
    );

    const blocks: any[] = response.output?.message?.content ?? [];
    const text: string = blocks.map((block) => block.text ?? "").join("");
    const cost = this.resolveCost(
      response.usage?.inputTokens,
      response.usage?.outputTokens,
    );

    if (schema) {
      return { output: schema.parse(extractJson(text)), cost };
    }
    return { output: text as T, cost };
  }

  getModelName(): string {
    return this.modelName ?? "amazon-bedrock";
  }
}
