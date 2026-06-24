import { DeepEvalBaseLLM } from "../models";
import { Golden, ConversationalGolden } from "../dataset";
import {
  SynthesizerOptions,
  FiltrationConfig,
  EvolutionConfig,
  StylingConfig,
  ConversationalStylingConfig,
  ContextConstructionConfig,
  Evolution,
} from "./types";

const DEFAULT_FILTRATION: Required<FiltrationConfig> = {
  syntheticInputQualityThreshold: 0.5,
  maxQualityRetries: 3,
};

const DEFAULT_EVOLUTION: Required<EvolutionConfig> = {
  numEvolutions: 1,
  evolutions: {
    [Evolution.Reasoning]: 1 / 7,
    [Evolution.MultiContext]: 1 / 7,
    [Evolution.Concretizing]: 1 / 7,
    [Evolution.Constrained]: 1 / 7,
    [Evolution.Comparative]: 1 / 7,
    [Evolution.Hypothetical]: 1 / 7,
    [Evolution.InBreadth]: 1 / 7,
  },
};

/**
 * Synthesizer for generating synthetic evaluation data.
 *
 * Supports generating goldens from documents, contexts, scratch, or existing
 * goldens. Uses either the Confident AI API or a local LLM for generation.
 *
 * Mirrors Python's `deepeval.synthesizer.Synthesizer`.
 */
export class Synthesizer {
  private model?: DeepEvalBaseLLM;
  private filtrationConfig: Required<FiltrationConfig>;
  private evolutionConfig: Required<EvolutionConfig>;
  private stylingConfig?: StylingConfig;
  private conversationalStylingConfig?: ConversationalStylingConfig;
  private _goldens: Golden[] = [];
  private _conversationalGoldens: ConversationalGolden[] = [];

  constructor(
    model?: DeepEvalBaseLLM,
    options: SynthesizerOptions = {},
  ) {
    this.model = model;
    this.filtrationConfig = {
      ...DEFAULT_FILTRATION,
      ...options.filtrationConfig,
    };
    this.evolutionConfig = {
      ...DEFAULT_EVOLUTION,
      ...options.evolutionConfig,
      evolutions: {
        ...DEFAULT_EVOLUTION.evolutions,
        ...options.evolutionConfig?.evolutions,
      },
    };
    this.stylingConfig = options.stylingConfig;
    this.conversationalStylingConfig = options.conversationalStylingConfig;
  }

  get goldens(): Golden[] {
    return this._goldens;
  }

  get conversationalGoldens(): ConversationalGolden[] {
    return this._conversationalGoldens;
  }

  /**
   * Generate goldens from document files.
   */
  async generateGoldensFromDocs(
    documentPaths: string[],
    options?: {
      includeExpectedOutput?: boolean;
      maxGoldensPerContext?: number;
      contextConstructionConfig?: ContextConstructionConfig;
    },
  ): Promise<Golden[]> {
    const { includeExpectedOutput = true, maxGoldensPerContext = 2 } = options ?? {};

    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const fs = await import("node:fs");
    const contexts: string[][] = [];

    for (const docPath of documentPaths) {
      const content = fs.readFileSync(docPath, "utf-8");
      contexts.push([content]);
    }

    return this.generateGoldensFromContexts(contexts, {
      includeExpectedOutput,
      maxGoldensPerContext,
    });
  }

  /**
   * Generate goldens from provided contexts.
   */
  async generateGoldensFromContexts(
    contexts: string[][],
    options?: {
      includeExpectedOutput?: boolean;
      maxGoldensPerContext?: number;
    },
  ): Promise<Golden[]> {
    const { includeExpectedOutput = true, maxGoldensPerContext = 2 } = options ?? {};

    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const results: Golden[] = [];

    for (const contextGroup of contexts) {
      for (let i = 0; i < maxGoldensPerContext; i++) {
        const contextText = contextGroup.join("\n\n");
        const input = await this.generateInput(contextText);
        const quality = await this.assessQuality(input);

        let finalInput = input;
        if (quality < this.filtrationConfig.syntheticInputQualityThreshold) {
          finalInput = await this.rewriteInput(input);
        }

        const evolvedInput = await this.evolveInput(finalInput);
        const expectedOutput = includeExpectedOutput
          ? await this.generateExpectedOutputInline(evolvedInput, contextText)
          : undefined;

        const golden = new Golden({
          input: evolvedInput,
          expectedOutput,
          context: contextGroup,
          retrievalContext: contextGroup,
        });
        results.push(golden);
      }
    }

    this._goldens = results;
    return results;
  }

  /**
   * Generate goldens from scratch (requires styling config).
   */
  async generateGoldensFromScratch(
    numGoldens: number,
  ): Promise<Golden[]> {
    if (!this.stylingConfig) {
      throw new Error(
        "A styling config is required for generating goldens from scratch. " +
        "Pass `stylingConfig` in the constructor options.",
      );
    }
    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const results: Golden[] = [];

    for (let i = 0; i < numGoldens; i++) {
      const prompt = `Generate a synthetic user input for the following scenario:
Scenario: ${this.stylingConfig.scenario ?? "N/A"}
Task: ${this.stylingConfig.task ?? "N/A"}
Input Format: ${this.stylingConfig.inputFormat ?? "N/A"}

Return only the generated input text.`;

      const { output: input } = await this.model.generate(prompt);
      const evolvedInput = await this.evolveInput(input);

      const expectedOutput = this.stylingConfig.expectedOutputFormat
        ? await this.model.generate(
            `Generate an expected output for the following input:\n${evolvedInput}\n\nOutput format: ${this.stylingConfig.expectedOutputFormat}\n\nReturn only the expected output text.`,
          )
        : undefined;

      const golden = new Golden({
        input: evolvedInput,
        expectedOutput: expectedOutput?.output,
      });
      results.push(golden);
    }

    this._goldens = results;
    return results;
  }

  /**
   * Evolve existing goldens to create new variations.
   */
  async generateGoldensFromGoldens(
    goldens: Golden[],
    options?: {
      maxGoldensPerGolden?: number;
      includeExpectedOutput?: boolean;
    },
  ): Promise<Golden[]> {
    const { maxGoldensPerGolden = 2, includeExpectedOutput = true } = options ?? {};

    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const results: Golden[] = [];

    for (const golden of goldens) {
      for (let i = 0; i < maxGoldensPerGolden; i++) {
        const evolvedInput = await this.evolveInput(golden.input);

        let expectedOutput: string | undefined;
        if (includeExpectedOutput && golden.expectedOutput) {
          const result = await this.model.generate(
            `Given the original input "${golden.input}" with expected output "${golden.expectedOutput}", generate the expected output for the evolved input "${evolvedInput}". Return only the expected output text.`,
          );
          expectedOutput = result.output;
        }

        const newGolden = new Golden({
          input: evolvedInput,
          expectedOutput,
          context: golden.context,
          retrievalContext: golden.retrievalContext,
          additionalMetadata: golden.additionalMetadata,
        });
        results.push(newGolden);
      }
    }

    this._goldens = results;
    return results;
  }

  /**
   * Generate conversational goldens from documents.
   */
  async generateConversationalGoldensFromDocs(
    documentPaths: string[],
    options?: {
      includeExpectedOutcome?: boolean;
      maxGoldensPerContext?: number;
    },
  ): Promise<ConversationalGolden[]> {
    const { includeExpectedOutcome = true, maxGoldensPerContext = 2 } = options ?? {};

    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const fs = await import("node:fs");
    const contexts: string[][] = [];

    for (const docPath of documentPaths) {
      const content = fs.readFileSync(docPath, "utf-8");
      contexts.push([content]);
    }

    return this.generateConversationalGoldensFromContexts(contexts, {
      includeExpectedOutcome,
      maxGoldensPerContext,
    });
  }

  /**
   * Generate conversational goldens from provided contexts.
   */
  async generateConversationalGoldensFromContexts(
    contexts: string[][],
    options?: {
      includeExpectedOutcome?: boolean;
      maxGoldensPerContext?: number;
    },
  ): Promise<ConversationalGolden[]> {
    const { includeExpectedOutcome = true, maxGoldensPerContext = 2 } = options ?? {};

    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const results: ConversationalGolden[] = [];

    for (const contextGroup of contexts) {
      for (let i = 0; i < maxGoldensPerContext; i++) {
        const contextText = contextGroup.join("\n\n");

        const scenario = await this.model.generate(
          `Based on the following context, generate a conversational scenario:\n${contextText}\n\nReturn only the scenario description.`,
        );

        const expectedOutcome = includeExpectedOutcome
          ? await this.model.generate(
              `For the following scenario, describe the expected successful outcome:\n${scenario.output}\n\nReturn only the expected outcome.`,
            )
          : undefined;

        const golden = new ConversationalGolden({
          scenario: scenario.output,
          expectedOutcome: expectedOutcome?.output,
          context: contextGroup,
        });
        results.push(golden);
      }
    }

    this._conversationalGoldens = results;
    return results;
  }

  /**
   * Generate conversational goldens from scratch.
   */
  async generateConversationalGoldensFromScratch(
    numGoldens: number,
  ): Promise<ConversationalGolden[]> {
    if (!this.conversationalStylingConfig) {
      throw new Error(
        "A conversational styling config is required for generating conversational goldens from scratch.",
      );
    }
    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const results: ConversationalGolden[] = [];

    for (let i = 0; i < numGoldens; i++) {
      const prompt = `Generate a conversational scenario:
Scenario Context: ${this.conversationalStylingConfig.scenarioContext ?? "N/A"}
Task: ${this.conversationalStylingConfig.conversationalTask ?? "N/A"}
Participant Roles: ${this.conversationalStylingConfig.participantRoles ?? "N/A"}

Return only the scenario description.`;

      const { output: scenario } = await this.model.generate(prompt);

      const result = this.conversationalStylingConfig.expectedOutcomeFormat
        ? await this.model.generate(
            `For the scenario "${scenario}", generate the expected outcome using the format: ${this.conversationalStylingConfig.expectedOutcomeFormat}. Return only the expected outcome.`,
          )
        : undefined;

      const golden = new ConversationalGolden({
        scenario,
        expectedOutcome: result?.output,
      });
      results.push(golden);
    }

    this._conversationalGoldens = results;
    return results;
  }

  /**
   * Generate conversational goldens from existing goldens.
   */
  async generateConversationalGoldensFromGoldens(
    goldens: ConversationalGolden[],
    options?: {
      maxGoldensPerGolden?: number;
      includeExpectedOutcome?: boolean;
    },
  ): Promise<ConversationalGolden[]> {
    const { maxGoldensPerGolden = 2, includeExpectedOutcome = true } = options ?? {};

    if (!this.model) {
      throw new Error(
        "A model is required for synthetic data generation. Pass a model to the constructor.",
      );
    }

    const results: ConversationalGolden[] = [];

    for (const golden of goldens) {
      for (let i = 0; i < maxGoldensPerGolden; i++) {
        const evolvedScenario = await this.model.generate(
          `Evolve the following conversational scenario into a new variant:\n${golden.scenario}\n\nApply a transformation that changes the context while preserving the core intent. Return only the new scenario.`,
        );

        const expectedOutcome = includeExpectedOutcome && golden.expectedOutcome
          ? await this.model.generate(
              `For the evolved scenario "${evolvedScenario.output}", generate the expected outcome based on the original "${golden.expectedOutcome}". Return only the expected outcome.`,
            )
          : undefined;

        const newGolden = new ConversationalGolden({
          scenario: evolvedScenario.output,
          expectedOutcome: expectedOutcome?.output,
          context: golden.context,
          additionalMetadata: golden.additionalMetadata,
        });
        results.push(newGolden);
      }
    }

    this._conversationalGoldens = results;
    return results;
  }

  private async generateExpectedOutputInline(
    input: string,
    context: string,
  ): Promise<string> {
    const { output } = await this.model!.generate(
      `Given the input "${input}" and the following context, generate the expected correct output:\n${context}\n\nReturn only the expected output, no explanation.`,
    );
    return output;
  }

  private async generateInput(context: string): Promise<string> {
    const { output } = await this.model!.generate(
      `Based on the following context, generate a realistic user input that a user might ask about this content:\n${context}\n\nReturn only the input text, no explanation.`,
    );
    return output;
  }

  private async assessQuality(input: string): Promise<number> {
    const { output } = await this.model!.generate(
      `Rate the quality of this synthetic input on a scale of 0 to 1 (where 1 is highest quality):\n"${input}"\n\nReturn only a number between 0 and 1.`,
    );
    const parsed = parseFloat(output);
    return isNaN(parsed) ? 0.5 : Math.max(0, Math.min(1, parsed));
  }

  private async rewriteInput(input: string): Promise<string> {
    const { output } = await this.model!.generate(
      `Rewrite the following input to be more realistic, specific, and useful for testing an LLM:\n"${input}"\n\nReturn only the rewritten input, no explanation.`,
    );
    return output;
  }

  private async evolveInput(input: string): Promise<string> {
    const { numEvolutions } = this.evolutionConfig;
    let current = input;

    for (let i = 0; i < numEvolutions; i++) {
      const evolution = this.sampleEvolution();
      const instruction = this.getEvolutionInstruction(evolution);
      const { output } = await this.model!.generate(
        `${instruction}"${current}"\n\nReturn only the evolved input, no explanation.`,
      );
      current = output;
    }

    return current;
  }

  private sampleEvolution(): Evolution {
    const { evolutions } = this.evolutionConfig;
    const entries = Object.entries(evolutions) as [Evolution, number][];
    const r = Math.random();
    let cumulative = 0;
    for (const [evolution, weight] of entries) {
      cumulative += weight;
      if (r <= cumulative) return evolution;
    }
    return entries[entries.length - 1][0];
  }

  private getEvolutionInstruction(evolution: Evolution): string {
    switch (evolution) {
      case Evolution.Reasoning:
        return "Add a reasoning requirement to the following input, requiring multi-step logical thinking. Evolve the input to make it more complex:";
      case Evolution.MultiContext:
        return "Add multiple contextual dimensions to the following input, making it require cross-reference of different topics. Evolve the input:";
      case Evolution.Concretizing:
        return "Make the following input more concrete and specific, adding real-world details and constraints. Evolve the input:";
      case Evolution.Constrained:
        return "Add specific constraints (format, length, style, etc.) to the following input. Evolve the input:";
      case Evolution.Comparative:
        return "Add a comparative or contrastive element to the following input, requiring comparison between options. Evolve the input:";
      case Evolution.Hypothetical:
        return "Add a hypothetical or 'what if' scenario twist to the following input. Evolve the input:";
      case Evolution.InBreadth:
        return "Broaden the scope of the following input, adding related but distinct sub-questions. Evolve the input:";
    }
  }
}
