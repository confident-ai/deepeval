import { DeepEvalError } from "@/errors";
import { getSettings } from "@/config/settings";
import { importOptional } from "@/models/utils";
import {
  DeepEvalBaseSystemOneModel,
  type SystemOneDecision,
} from "@/models/system-one/base-system-one-model";
import {
  DEFAULT_TYPESAFE_MODEL,
  typesafeModelData,
  type SystemOneModelData,
} from "@/models/system-one/constants";
import { checkContextBudget } from "@/models/system-one/limits";
import {
  ChoiceAnswer,
  NoulAnswer,
  ScoreAnswer,
  SystemOneAnswers,
  type SystemOneQuestion,
} from "@/models/system-one/schema";

const TYPESAFE_SDK = "@typesafe-ai/sdk";

export interface TypeSafeModelOptions {
  /** Defaults to `TYPESAFE_MODEL_NAME`, then `jev-latest`. */
  model?: string;
  /** Defaults to `TYPESAFE_API_KEY`. */
  apiKey?: string;
  /** USD per input token; overrides `TYPESAFE_COST_PER_INPUT_TOKEN`. */
  costPerInputToken?: number;
  /** Passed through to the SDK's `TypeSafeClient` (e.g. `baseURL`). */
  [key: string]: unknown;
}

type SdkQuestion = {
  type: "noul" | "choice" | "score";
  instructions: unknown;
  criteria?: unknown;
};

export class TypeSafeModel extends DeepEvalBaseSystemOneModel {
  private apiKey?: string;
  private modelData: SystemOneModelData;
  private clientOptions: Record<string, unknown>;
  private client?: any;

  constructor(options: TypeSafeModelOptions = {}) {
    const settings = getSettings();
    const { model, apiKey, costPerInputToken, ...clientOptions } = options;
    const name =
      model || settings.TYPESAFE_MODEL_NAME || DEFAULT_TYPESAFE_MODEL;
    super(name);

    this.apiKey = apiKey ?? settings.TYPESAFE_API_KEY;
    // Checked here rather than on first call, so a metric fails at
    // construction instead of mid-evaluation.
    if (!this.apiKey) {
      throw new DeepEvalError(
        "TypeSafe AI API key is not configured. Set TYPESAFE_API_KEY in your " +
          "environment or pass `apiKey` to new TypeSafeModel(...).",
      );
    }
    const cost = costPerInputToken ?? settings.TYPESAFE_COST_PER_INPUT_TOKEN;
    if (cost !== undefined && cost < 0) {
      throw new DeepEvalError("TYPESAFE_COST_PER_INPUT_TOKEN must be >= 0.");
    }
    this.modelData = typesafeModelData(name);
    if (cost !== undefined) this.modelData.inputPrice = cost;
    this.modelData.outputPrice = 0;
    this.clientOptions = clientOptions;
  }

  async decide(
    state: unknown,
    questions: Record<string, SystemOneQuestion>,
  ): Promise<SystemOneDecision> {
    const sdkQuestions = this.toSdkQuestions(questions);
    checkContextBudget(state, sdkQuestions);
    const client = await this.loadModel();
    const response = await client.systemOne({
      model: this.modelName,
      state,
      questions: sdkQuestions,
    });
    return this.fromSdkResponse(response);
  }

  getModelName(): string {
    return `${this.modelName} (TypeSafe AI)`;
  }

  private toSdkQuestions(
    questions: Record<string, SystemOneQuestion>,
  ): Record<string, SdkQuestion> {
    const keys = Object.keys(questions);
    if (keys.length === 0) {
      throw new DeepEvalError(
        "TypeSafeModel.decide requires at least one question.",
      );
    }
    const sdk: Record<string, SdkQuestion> = {};
    for (const key of keys) {
      const question = questions[key];
      let entry: SdkQuestion;
      if (question.type === "noul") {
        const hasCriteria =
          (question.true !== undefined && question.true !== null) ||
          (question.false !== undefined && question.false !== null);
        entry = { type: "noul", instructions: question.instructions };
        if (hasCriteria) {
          entry.criteria = {
            true: question.true ?? null,
            false: question.false ?? null,
          };
        }
      } else if (question.type === "choice") {
        entry = {
          type: "choice",
          instructions: question.instructions,
          criteria: { ...question.options },
        };
      } else if (question.type === "score") {
        entry = {
          type: "score",
          instructions: question.instructions,
          criteria: [...question.levels],
        };
      } else {
        throw new DeepEvalError(
          `Unsupported System One question type: ${(question as { type?: unknown }).type}`,
        );
      }
      sdk[key] = entry;
    }
    return sdk;
  }

  private fromSdkResponse(response: any): SystemOneDecision {
    const answers = new SystemOneAnswers();
    for (const [key, raw] of Object.entries<any>(response?.answers ?? {})) {
      if (raw?.type === "noul") {
        answers.nouls[key] = new NoulAnswer(Number(raw.noul));
      } else if (raw?.type === "choice") {
        answers.choices[key] = new ChoiceAnswer(
          String(raw.choice),
          { ...(raw.probabilities ?? {}) },
          Number(raw.confidence),
        );
      } else if (raw?.type === "score") {
        const probabilities: Record<number, number> = {};
        for (const [level, p] of Object.entries<any>(raw.probabilities ?? {})) {
          probabilities[Number(level)] = Number(p);
        }
        answers.scores[key] = new ScoreAnswer(
          Number(raw.score),
          probabilities,
          Number(raw.confidence),
        );
      }
    }
    const usage = response?.usage ?? {};
    const inputTokens = Number(usage.input_tokens ?? usage.inputTokens ?? 0);
    const outputTokens = Number(usage.output_tokens ?? usage.outputTokens ?? 0);
    return { answers, cost: this.calculateCost(inputTokens, outputTokens) };
  }

  private calculateCost(
    inputTokens: number,
    outputTokens: number,
  ): number | null {
    if (this.modelData.inputPrice === undefined) return null;
    return (
      inputTokens * this.modelData.inputPrice +
      outputTokens * this.modelData.outputPrice
    );
  }

  private async loadModel(): Promise<any> {
    if (this.client) return this.client;
    const sdk = await importOptional(TYPESAFE_SDK, "TypeSafeModel");
    const TypeSafeClient = sdk.TypeSafeClient ?? sdk.default?.TypeSafeClient;
    this.client = new TypeSafeClient({
      apiKey: this.apiKey,
      ...this.clientOptions,
    });
    return this.client;
  }
}
