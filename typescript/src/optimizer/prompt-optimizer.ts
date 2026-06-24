import { DeepEvalBaseLLM, OpenAIModel } from "../models";
import {
  BaseMetric,
  BaseConversationalMetric,
} from "../metrics";
import { Golden, ConversationalGolden } from "../dataset";
import { Prompt } from "../prompt";
import {
  ModelCallback,
  OptimizationConfig,
  IterationLog,
  AcceptedIteration,
  OptimizationReport,
} from "./types";

/**
 * PromptOptimizer — the TS counterpart of Python's `PromptOptimizer`.
 *
 * Uses a simple evolutionary algorithm to iteratively improve a prompt based on
 * evaluation metrics applied to a set of goldens.
 *
 * @example
 * ```ts
 * const optimizer = new PromptOptimizer({
 *   modelCallback: async (prompt, golden) => {
 *     const template = prompt.interpolate({ input: golden.input }) as string;
 *     const { output } = await myModel.generate(template);
 *     return output;
 *   },
 *   metrics: [new FaithfulnessMetric()],
 * });
 *
 * const best = await optimizer.optimize(prompt, goldens);
 * ```
 */
export class PromptOptimizer {
  private modelCallback: ModelCallback;
  private metrics: (BaseMetric | BaseConversationalMetric)[];
  private optimizerModel: DeepEvalBaseLLM;
  private config: Required<OptimizationConfig>;

  constructor(options: {
    modelCallback: ModelCallback;
    metrics: (BaseMetric | BaseConversationalMetric)[];
    optimizerModel?: DeepEvalBaseLLM | string;
    config?: OptimizationConfig;
  }) {
    this.modelCallback = options.modelCallback;
    this.metrics = options.metrics;

    if (typeof options.optimizerModel === "string") {
      this.optimizerModel = new OpenAIModel({ model: options.optimizerModel });
    } else if (options.optimizerModel instanceof DeepEvalBaseLLM) {
      this.optimizerModel = options.optimizerModel;
    } else {
      this.optimizerModel = new OpenAIModel();
    }

    this.config = {
      iterations: options.config?.iterations ?? 5,
      minibatchSize: options.config?.minibatchSize ?? 8,
      paretoSize: options.config?.paretoSize ?? 3,
      patience: options.config?.patience ?? 3,
      randomSeed: options.config?.randomSeed ?? Date.now(),
    };
  }

  /**
   * Run prompt optimization.
   *
   * @param prompt - The base prompt to optimize.
   * @param goldens - Evaluation goldens to optimize against.
   * @returns The optimized prompt.
   */
  async optimize(
    prompt: Prompt,
    goldens: (Golden | ConversationalGolden)[],
  ): Promise<OptimizationReport> {
    const logs: IterationLog[] = [];
    const acceptedIterations: AcceptedIteration[] = [];

    // Split goldens into feedback set and pareto set
    const paretoSet = goldens.slice(0, this.config.paretoSize);
    const feedbackSet = goldens.slice(this.config.paretoSize);

    // Score the original prompt on the pareto set as baseline
    const baseScore = await this.scorePrompt(prompt, paretoSet);
    let bestPrompt = prompt;
    let bestScore = baseScore;
    let consecutiveRejections = 0;

    console.log(
      `\n🚀 Starting prompt optimization (${this.config.iterations} max iterations)...`,
    );
    console.log(`   Base prompt score: ${(baseScore * 100).toFixed(1)}%\n`);

    for (let i = 0; i < this.config.iterations; i++) {
      const iterationStart = Date.now();

      // Select a minibatch from the feedback set
      const minibatch = this.selectMinibatch(feedbackSet);

      // Evaluate parent on minibatch
      const parentScore = await this.scorePrompt(bestPrompt, minibatch);

      // Generate feedback from the scorer on failure cases
      const feedback = await this.generateFeedback(bestPrompt, minibatch);

      // Rewrite the prompt using the feedback
      const candidatePrompt = await this.rewritePrompt(bestPrompt, feedback);
      const candidateScore = await this.scorePrompt(candidatePrompt, minibatch);

      const elapsed = (Date.now() - iterationStart) / 1000;

      if (candidateScore > parentScore) {
        // Full evaluation on pareto set
        const fullCandidateScore = await this.scorePrompt(
          candidatePrompt,
          paretoSet,
        );

        if (fullCandidateScore > bestScore) {
          bestPrompt = candidatePrompt;
          bestScore = fullCandidateScore;

          acceptedIterations.push({
            parent: `iteration_${i}`,
            child: `iteration_${i + 1}`,
            module: "__module__",
            before: parentScore,
            after: fullCandidateScore,
          });

          logs.push({
            iteration: i,
            outcome: "accepted",
            reason: `Score improved from ${(parentScore * 100).toFixed(1)}% to ${(fullCandidateScore * 100).toFixed(1)}%`,
            before: parentScore,
            after: fullCandidateScore,
          });

          consecutiveRejections = 0;
          console.log(
            `   ✓ Iteration ${i + 1}: ${(fullCandidateScore * 100).toFixed(1)}% (improved, ${elapsed.toFixed(1)}s)`,
          );
        } else {
          logs.push({
            iteration: i,
            outcome: "rejected",
            reason: `Candidate score ${(fullCandidateScore * 100).toFixed(1)}% did not beat best ${(bestScore * 100).toFixed(1)}%`,
            before: parentScore,
            after: fullCandidateScore,
          });
          consecutiveRejections++;
          console.log(
            `   ✗ Iteration ${i + 1}: ${(fullCandidateScore * 100).toFixed(1)}% (no improvement, ${elapsed.toFixed(1)}s)`,
          );
        }
      } else {
        logs.push({
          iteration: i,
          outcome: "rejected",
          reason: `Candidate score ${(candidateScore * 100).toFixed(1)}% did not beat parent ${(parentScore * 100).toFixed(1)}% on minibatch`,
          before: parentScore,
          after: candidateScore,
        });
        consecutiveRejections++;
        console.log(
          `   ✗ Iteration ${i + 1}: ${(candidateScore * 100).toFixed(1)}% (no improvement, ${elapsed.toFixed(1)}s)`,
        );
      }

      if (consecutiveRejections >= this.config.patience) {
        console.log(
          `\n   Early stopping after ${consecutiveRejections} consecutive rejections.`,
        );
        break;
      }
    }

    console.log(
      `\n✅ Optimization complete. Best score: ${(bestScore * 100).toFixed(1)}%\n`,
    );

    return {
      optimizationId: `opt_${Date.now()}`,
      bestPrompt,
      acceptedIterations,
      logs,
    };
  }

  private selectMinibatch(
    goldens: (Golden | ConversationalGolden)[],
  ): (Golden | ConversationalGolden)[] {
    const size = Math.min(this.config.minibatchSize, goldens.length);
    const shuffled = [...goldens].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, size);
  }

  private async scorePrompt(
    prompt: Prompt,
    goldens: (Golden | ConversationalGolden)[],
  ): Promise<number> {
    if (goldens.length === 0) return 0;

    let totalScore = 0;
    let count = 0;

    for (const golden of goldens) {
      try {
        const actualOutput = await this.modelCallback(prompt, golden);
        const testCase = this.goldenToTestCase(golden, actualOutput);

        for (const metric of this.metrics) {
          metric.score = undefined;
          metric.success = undefined;
          metric.reason = undefined;
          metric.error = undefined;
          metric.skipped = false;
          await (metric.measure as (tc: any) => Promise<number>)(testCase);

          if (metric.score != null) {
            totalScore += metric.score;
            count++;
          }
        }
      } catch {
        continue;
      }
    }

    return count > 0 ? totalScore / count : 0;
  }

  private async generateFeedback(
    prompt: Prompt,
    goldens: (Golden | ConversationalGolden)[],
  ): Promise<string> {
    const failures: string[] = [];

    for (const golden of goldens.slice(0, 3)) {
      try {
        const actualOutput = await this.modelCallback(prompt, golden);
        const testCase = this.goldenToTestCase(golden, actualOutput);

        for (const metric of this.metrics) {
          metric.score = undefined;
          metric.success = undefined;
          metric.reason = undefined;
          metric.error = undefined;
          metric.skipped = false;
          await (metric.measure as (tc: any) => Promise<number>)(testCase);

          if (!metric.success && metric.reason) {
            failures.push(
              `Input: ${(golden as Golden).input}\nMetric: ${metric.name}\nReason: ${metric.reason}\nScore: ${metric.score}`,
            );
          }
        }
      } catch {
        continue;
      }
    }

    if (failures.length === 0) return "";

    const { output } = await this.optimizerModel.generate(
      `Analyze the following evaluation failures and suggest concrete improvements to the prompt:\n\n${failures.join("\n---\n")}\n\nProvide specific, actionable feedback for improving the prompt. Focus on patterns in the failures.`,
    );
    return output;
  }

  private async rewritePrompt(
    basePrompt: Prompt,
    feedback: string,
  ): Promise<Prompt> {
    const currentText = basePrompt.textTemplate ?? "";

    if (!feedback) {
      return basePrompt;
    }

    const { output } = await this.optimizerModel.generate(
      `Given the current prompt:\n\n${currentText}\n\nAnd the following feedback on its failures:\n\n${feedback}\n\nGenerate an improved version of the prompt that addresses the feedback. Keep the same variable placeholders (e.g. {{input}}). Return only the improved prompt text, no explanation.`,
    );

    // Create a local Prompt with the improved text
    const newPrompt = new Prompt({ alias: `${(basePrompt as any)._alias}_optimized` });
    Object.assign(newPrompt, {
      _textTemplate: output,
      _type: "TEXT",
      _interpolationType: "FSTRING",
    });

    return newPrompt;
  }

  private goldenToTestCase(
    golden: Golden | ConversationalGolden,
    actualOutput: string,
  ): any {
    const { LLMTestCase } = require("../test-case");
    const g = golden as Golden;
    return new LLMTestCase({
      input: g.input,
      actualOutput,
      expectedOutput: g.expectedOutput,
      context: g.context,
      retrievalContext: g.retrievalContext,
    });
  }
}
