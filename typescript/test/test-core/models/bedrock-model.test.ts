import { AmazonBedrockModel } from "../../../src/models/providers/bedrock-model";

const AWS_CREDENTIAL_ENV_KEYS = [
  "AWS_ACCESS_KEY_ID",
  "AWS_SECRET_ACCESS_KEY",
  "AWS_SESSION_TOKEN",
] as const;

type BedrockCredentials = {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
};

type BedrockClientConfig = {
  region?: string;
  credentials?: BedrockCredentials;
};

class FakeConverseCommand {
  constructor(readonly input: unknown) {}
}

class FakeBedrockRuntimeClient {
  static lastConfig?: BedrockClientConfig;
  static lastCommand?: FakeConverseCommand;

  constructor(readonly config: BedrockClientConfig) {}

  async send(command: FakeConverseCommand) {
    FakeBedrockRuntimeClient.lastConfig = this.config;
    FakeBedrockRuntimeClient.lastCommand = command;
    return {
      output: { message: { content: [{ text: "ok" }] } },
      usage: {},
    };
  }
}

function installFakeSdk(model: AmazonBedrockModel) {
  (model as unknown as { sdk: unknown }).sdk = {
    BedrockRuntimeClient: FakeBedrockRuntimeClient,
    ConverseCommand: FakeConverseCommand,
  };
}

describe("AmazonBedrockModel", () => {
  const originalCredentialEnv = new Map<string, string | undefined>();

  beforeAll(() => {
    for (const key of AWS_CREDENTIAL_ENV_KEYS) {
      originalCredentialEnv.set(key, process.env[key]);
    }
  });

  beforeEach(() => {
    for (const key of AWS_CREDENTIAL_ENV_KEYS) {
      delete process.env[key];
    }
    FakeBedrockRuntimeClient.lastConfig = undefined;
    FakeBedrockRuntimeClient.lastCommand = undefined;
  });

  afterAll(() => {
    for (const key of AWS_CREDENTIAL_ENV_KEYS) {
      const value = originalCredentialEnv.get(key);
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  });

  it("does not advertise multimodal support", () => {
    const model = new AmazonBedrockModel({
      model: "anthropic.claude-3-haiku-20240307-v1:0",
    });

    expect(model.supportsMultimodal()).toBe(false);
  });

  it("sends prompts as text-only Converse content", async () => {
    const model = new AmazonBedrockModel({
      model: "anthropic.claude-3-haiku-20240307-v1:0",
      region: "us-west-2",
    });
    installFakeSdk(model);

    await model.generate("Describe this image: [DEEPEVAL:IMAGE:test-image]");

    expect(FakeBedrockRuntimeClient.lastConfig?.region).toBe("us-west-2");
    expect(FakeBedrockRuntimeClient.lastCommand?.input).toMatchObject({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      messages: [
        {
          role: "user",
          content: [
            { text: "Describe this image: [DEEPEVAL:IMAGE:test-image]" },
          ],
        },
      ],
    });
  });

  it("omits explicit credentials when no credential options or env vars are set", async () => {
    const model = new AmazonBedrockModel({
      model: "anthropic.claude-3-haiku-20240307-v1:0",
    });
    installFakeSdk(model);

    await model.generate("hello");

    expect(FakeBedrockRuntimeClient.lastConfig?.credentials).toBeUndefined();
  });

  it("uses a complete explicit credential bundle without environment mixing", async () => {
    process.env.AWS_ACCESS_KEY_ID = "env-key";
    process.env.AWS_SECRET_ACCESS_KEY = "env-secret";
    process.env.AWS_SESSION_TOKEN = "env-token";
    const model = new AmazonBedrockModel({
      model: "anthropic.claude-3-haiku-20240307-v1:0",
      awsAccessKeyId: "explicit-key",
      awsSecretAccessKey: "explicit-secret",
      awsSessionToken: "explicit-token",
    });
    installFakeSdk(model);

    await model.generate("hello");

    expect(FakeBedrockRuntimeClient.lastConfig?.credentials).toEqual({
      accessKeyId: "explicit-key",
      secretAccessKey: "explicit-secret",
      sessionToken: "explicit-token",
    });
  });

  it("uses a complete environment credential bundle", async () => {
    process.env.AWS_ACCESS_KEY_ID = "env-key";
    process.env.AWS_SECRET_ACCESS_KEY = "env-secret";
    process.env.AWS_SESSION_TOKEN = "env-token";
    const model = new AmazonBedrockModel({
      model: "anthropic.claude-3-haiku-20240307-v1:0",
    });
    installFakeSdk(model);

    await model.generate("hello");

    expect(FakeBedrockRuntimeClient.lastConfig?.credentials).toEqual({
      accessKeyId: "env-key",
      secretAccessKey: "env-secret",
      sessionToken: "env-token",
    });
  });

  it("rejects partial explicit credentials instead of mixing them with the environment", () => {
    process.env.AWS_ACCESS_KEY_ID = "env-key";
    process.env.AWS_SECRET_ACCESS_KEY = "env-secret";

    expect(
      () =>
        new AmazonBedrockModel({
          model: "anthropic.claude-3-haiku-20240307-v1:0",
          awsAccessKeyId: "explicit-key",
        }),
    ).toThrow(/explicit credentials require both/);
  });

  it("rejects partial environment credentials instead of falling through to another AWS principal", () => {
    process.env.AWS_ACCESS_KEY_ID = "env-key";

    expect(
      () =>
        new AmazonBedrockModel({
          model: "anthropic.claude-3-haiku-20240307-v1:0",
        }),
    ).toThrow(/environment credentials require both/);
  });
});
