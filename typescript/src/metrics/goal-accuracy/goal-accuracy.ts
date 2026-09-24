import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { resolveThreshold } from "@/metrics/base-metrics";
import { ConversationalTestCase, MultiTurnParams, Turn } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  constructVerboseLogs,
  printToolsCalled,
} from "@/metrics/utils";
import {
  formatDecisionReason,
  parseQuestions,
  runSystemOneEval,
  systemOneScore,
  type SystemOneEvalSpec,
  type SystemOneScoreSpec,
} from "@/metrics/system-one";
import {
  checkConversationalTestCaseParams,
  getUnitInteractions,
} from "@/metrics/conversational-utils";
import {
  GoalScoreSchema,
  PlanScoreSchema,
  type GoalScore,
  type PlanScore,
  type GoalSteps,
} from "@/metrics/goal-accuracy/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "GoalAccuracyMetric";

const GOAL_ACCURACY_LEVELS = [
  "Not achieved",
  "Weak attempt",
  "Partially achieved",
  "Mostly achieved",
  "Fully achieved",
];

const PLAN_QUALITY_LEVELS = [
  "No plan",
  "Weak or fragmented plan, rarely followed",
  "Partial plan, partly followed",
  "Clear plan, mostly followed",
  "Complete plan, fully followed",
];

export type GoalAccuracyTemplateOverride =
  MetricTemplateOverride<"GoalAccuracyMetric">;

export interface GoalAccuracyMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  /** The System One model (Jev) used under `hybrid` / `system_one`. */
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  /** Who decides; defaults to `DEEPEVAL_EVAL_MODE`, then `llm`. */
  evalMode?: EvalModeName;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: GoalAccuracyTemplateOverride;
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
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.multimodalAware = true;
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [MultiTurnParams.ROLE, MultiTurnParams.CONTENT];
    initializeMetricModels(this, options);
  }

  async measure(testCase: ConversationalTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkConversationalTestCaseParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      const tasks = this.goalAndStepsTaken(getUnitInteractions(testCase.turns));
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
      this.success = this.isSuccessful();
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
    const kind = method === "get_accuracy_score" ? "goal" : "plan";
    const value = await systemOneScore(
      this,
      this.systemOneScoreSpec(kind, task),
    );
    if (value !== undefined) {
      return {
        score: value,
        reason: formatDecisionReason(
          this,
          kind === "goal" ? "goal accuracy" : "plan quality",
          value,
        ),
      } as T;
    }
    const prompt = this.getPrompt(method, {
      task: task.user_goal,
      steps_taken: task.steps_taken.join("\n"),
    });
    return generateWithSchema(this, prompt, schema) as Promise<T>;
  }

  private systemOneScoreSpec(
    kind: "goal" | "plan",
    task: GoalSteps,
  ): SystemOneScoreSpec | undefined {
    if (this.multimodal) return undefined;
    return {
      instructions: this.getPrompt(`_experimental_system_one_${kind}_score`),
      levels: kind === "goal" ? GOAL_ACCURACY_LEVELS : PLAN_QUALITY_LEVELS,
      state: { task: task.user_goal, steps_taken: [...task.steps_taken] },
    };
  }

  systemOneEvalSpec(
    testCase: ConversationalTestCase,
  ): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: [...this.requiredParams, MultiTurnParams.TOOLS_CALLED],
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
    };
  }

  private calculateScore(): number {
    const goalAvg =
      this.goalScores.reduce((s, g) => s + g.score, 0) /
      Math.max(this.goalScores.length, 1);
    const planAvg =
      this.planScores.reduce((s, p) => s + p.score, 0) /
      Math.max(this.planScores.length, 1);
    const score = (goalAvg + planAvg) / 2;
    return this.applyStrictMode(score);
  }

  private async generateFinalReason(): Promise<string> {
    const goalEvaluations = this.goalScores
      .map((g) => `Score: ${g.score}, Reason: ${g.reason}`)
      .join("");
    const planEvaluations = this.planScores
      .map((p) => `Score: ${p.score}, Reason: ${p.reason} \n`)
      .join("");
    const prompt = this.getPrompt("get_final_reason", {
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

  get name(): string {
    return "Goal Accuracy";
  }
}
