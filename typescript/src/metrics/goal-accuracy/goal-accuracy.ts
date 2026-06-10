import { BaseConversationalMetric } from "../base-conversational-metric";
import {
  ConversationalTestCase,
  MultiTurnParams,
  Turn,
} from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  constructVerboseLogs,
  printToolsCalled,
} from "../utils";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
} from "../conversational-utils";
import {
  GoalScoreSchema,
  PlanScoreSchema,
  type GoalScore,
  type PlanScore,
  type GoalSteps,
} from "./schema";

const TEMPLATE_CLASS = "GoalAccuracyMetric";

export interface GoalAccuracyMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
}

/**
 * Goal Accuracy — for each interaction, how well did the assistant's steps
 * achieve the user's goal (goal score) and how sound was the plan (plan score)?
 * Final score = mean of the two averages. **Higher is better**.
 */
export class GoalAccuracyMetric extends BaseConversationalMetric {
  goalScores: GoalScore[] = [];
  planScores: PlanScore[] = [];

  constructor(options: GoalAccuracyMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT];
    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

      const tasks = this.goalAndStepsTaken(
        getUnitInteractions(testCase.turns),
      );
      [this.goalScores, this.planScores] = await Promise.all([
        Promise.all(
          tasks.map((t) =>
            this.scoreWith("get_accuracy_score", t, GoalScoreSchema),
          ),
        ),
        Promise.all(
          tasks.map((t) =>
            this.scoreWith("get_plan_evaluation_score", t, PlanScoreSchema),
          ),
        ),
      ]);
      this.score = this.calculateScore();
      this.success = this.score >= this.threshold;
      this.reason = await this.generateFinalReason();

      this.verboseLogs = constructVerboseLogs(this, [
        `Tasks: ${tasks.length}`,
        `Goal scores: ${this.goalScores.map((g) => g.score).join(", ")}`,
        `Plan scores: ${this.planScores.map((p) => p.score).join(", ")}`,
        `Final Score: ${this.score}\nFinal Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private goalAndStepsTaken(unitInteractions: Turn[][]): GoalSteps[] {
    const result: GoalSteps[] = [];
    for (const interaction of unitInteractions) {
      let userMessages = "User messages: \n";
      for (const turn of interaction) {
        if (turn.role === "user") userMessages += turn.content + "\n";
      }
      const goalSteps: GoalSteps = { user_goal: userMessages, steps_taken: [] };
      let assistantMessages = "Assistant messages: \n";
      for (const turn of interaction.slice(1)) {
        if (turn.role === "assistant") {
          assistantMessages += `${turn.content} \n`;
          if (turn.toolsCalled && turn.toolsCalled.length > 0) {
            assistantMessages += `Tools called: \n${printToolsCalled(turn.toolsCalled)} \n`;
          }
          goalSteps.steps_taken.push(assistantMessages);
        }
      }
      result.push(goalSteps);
    }
    return result;
  }

  private async scoreWith<T extends GoalScore | PlanScore>(
    method: "get_accuracy_score" | "get_plan_evaluation_score",
    task: GoalSteps,
    schema: typeof GoalScoreSchema | typeof PlanScoreSchema,
  ): Promise<T> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, method, {
      task: task.user_goal,
      steps_taken: task.steps_taken.join("\n"),
    });
    return generateWithSchema(this, prompt, schema) as Promise<T>;
  }

  private calculateScore(): number {
    const goalAvg =
      this.goalScores.reduce((s, g) => s + g.score, 0) /
      Math.max(this.goalScores.length, 1);
    const planAvg =
      this.planScores.reduce((s, p) => s + p.score, 0) /
      Math.max(this.planScores.length, 1);
    const score = (goalAvg + planAvg) / 2;
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  private async generateFinalReason(): Promise<string> {
    const goalEvaluations = this.goalScores
      .map((g) => `Score: ${g.score}, Reason: ${g.reason}`)
      .join("");
    const planEvaluations = this.planScores
      .map((p) => `Score: ${p.score}, Reason: ${p.reason} \n`)
      .join("");
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "get_final_reason", {
      final_score: this.score,
      threshold: this.threshold,
      goal_evaluations: goalEvaluations,
      // NOTE: matches Python's misspelled template variable.
      plan_evalautions: planEvaluations,
    });
    // Free-text reason (no schema), mirroring Python's raw `model.generate`.
    const { output, cost } = await this.model!.generate(prompt);
    this.accrueCost(cost);
    return output;
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Goal Accuracy";
  }
}
