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
  PromptEvolution,
} from "./types";
import {
  EvolutionTemplate,
  PromptEvolutionTemplate,
  SynthesizerPromptTemplate,
} from "./templates";

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

const evolutionMap: Record<string, (input: string, context: string | null) => string> = {
  [Evolution.Reasoning]: EvolutionTemplate.reasoning_evolution,
  [Evolution.MultiContext]: (input, context) => EvolutionTemplate.multi_context_evolution(input, context!),
  [Evolution.Concretizing]: EvolutionTemplate.concretizing_evolution,
  [Evolution.Constrained]: EvolutionTemplate.constrained_evolution,
  [Evolution.Comparative]: EvolutionTemplate.comparative_question_evolution,
  [Evolution.Hypothetical]: EvolutionTemplate.hypothetical_scenario_evolution,
  [Evolution.InBreadth]: EvolutionTemplate.in_breadth_evolution,
};

const promptEvolutionMap: Record<string, (input: string) => string> = {
  [PromptEvolution.Reasoning]: PromptEvolutionTemplate.reasoning_evolution,
  [PromptEvolution.Concretizing]: PromptEvolutionTemplate.concretizing_evolution,
  [PromptEvolution.Constrained]: PromptEvolutionTemplate.constrained_evolution,
  [PromptEvolution.Comparative]: PromptEvolutionTemplate.comparative_question_evolution,
  [PromptEvolution.Hypothetical]: PromptEvolutionTemplate.hypothetical_scenario_evolution,
  [PromptEvolution.InBreadth]: PromptEvolutionTemplate.in_breadth_evolution,
};

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
        const qualifiedInput = await this.qualifyInput(input, contextText);
        const evolvedInput = await this.evolveInput(qualifiedInput, contextText);

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
    const { scenario, task, inputFormat } = this.stylingConfig;

    for (let i = 0; i < numGoldens; i++) {
      const { output: input } = await this.model.generate(
        `Generate a synthetic user input for the following scenario:\nScenario: ${scenario ?? "N/A"}\nTask: ${task ?? "N/A"}\nInput Format: ${inputFormat ?? "N/A"}\n\nReturn only the generated input text.`,
      );

      const evolvedInput = await this.evolveInputNoContext(input);

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
        const contextText = golden.context?.join("\n\n") ?? null;
        const evolvedInput = contextText
          ? await this.evolveInput(golden.input, contextText)
          : await this.evolveInputNoContext(golden.input);

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
      const prompt = `Generate a conversational scenario:\nScenario Context: ${this.conversationalStylingConfig.scenarioContext ?? "N/A"}\nTask: ${this.conversationalStylingConfig.conversationalTask ?? "N/A"}\nParticipant Roles: ${this.conversationalStylingConfig.participantRoles ?? "N/A"}\n\nReturn only the scenario description.`;

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
    const prompt = SynthesizerPromptTemplate.generate_expected_output(input, context);
    const { output } = await this.model!.generate(prompt);
    return output;
  }

  private async generateInput(context: string): Promise<string> {
    const prompt = SynthesizerPromptTemplate.generate_inputs(context, 1);
    const { output } = await this.model!.generate(prompt);
    return output;
  }

  private async evaluateInputQuality(input: string): Promise<{ score: number; feedback: string }> {
    const prompt = SynthesizerPromptTemplate.evaluate_input_quality(input);
    const { output } = await this.model!.generate(prompt);
    try {
      const parsed = JSON.parse(output);
      return {
        score: typeof parsed.score === "number" ? Math.max(0, Math.min(1, parsed.score)) : 0.5,
        feedback: typeof parsed.feedback === "string" ? parsed.feedback : "",
      };
    } catch {
      return { score: 0.5, feedback: "" };
    }
  }

  private async qualifyInput(
    input: string,
    _context: string,
  ): Promise<string> {
    let current = input;
    for (let attempt = 0; attempt < this.filtrationConfig.maxQualityRetries; attempt++) {
      const { score, feedback } = await this.evaluateInputQuality(current);
      if (score >= this.filtrationConfig.syntheticInputQualityThreshold) {
        return current;
      }
      const prompt = SynthesizerPromptTemplate.rewrite_input(current, feedback);
      const { output } = await this.model!.generate(prompt);
      current = output;
    }
    return current;
  }

  private get evolutionWeights(): Record<Evolution, number> {
    return this.evolutionConfig.evolutions as Record<Evolution, number>;
  }

  private async evolveInput(input: string, context: string): Promise<string> {
    const { numEvolutions } = this.evolutionConfig;
    const evolutions = this.evolutionWeights;
    let current = input;

    for (let i = 0; i < numEvolutions; i++) {
      const evolution = this.sampleEvolution(evolutions);
      const method = evolutionMap[evolution];
      if (!method) continue;
      const prompt = method(current, context);
      const { output } = await this.model!.generate(prompt);
      current = output;
    }

    return current;
  }

  private async evolveInputNoContext(input: string): Promise<string> {
    const { numEvolutions } = this.evolutionConfig;
    const promptEvolutions = this.toPromptEvolutions(this.evolutionWeights);
    let current = input;

    for (let i = 0; i < numEvolutions; i++) {
      const evolution = this.sampleEvolution(promptEvolutions);
      const method = promptEvolutionMap[evolution];
      if (!method) continue;
      const prompt = method(current);
      const { output } = await this.model!.generate(prompt);
      current = output;
    }

    return current;
  }

  private sampleEvolution(
    evolutions: Record<string, number>,
  ): string {
    const entries = Object.entries(evolutions);
    const r = Math.random();
    let cumulative = 0;
    for (const [evolution, weight] of entries) {
      cumulative += weight;
      if (r <= cumulative) return evolution;
    }
    return entries[entries.length - 1][0];
  }

  private toPromptEvolutions(
    evolutions: Record<Evolution, number>,
  ): Record<PromptEvolution, number> {
    const result: Record<string, number> = {};
    for (const [key, weight] of Object.entries(evolutions)) {
      if (key === Evolution.MultiContext) continue;
      result[key] = weight;
    }
    // Renormalize weights after removing MultiContext
    const total = Object.values(result).reduce((s, w) => s + w, 0);
    if (total > 0) {
      for (const key of Object.keys(result)) {
        result[key] /= total;
      }
    }
    return result as unknown as Record<PromptEvolution, number>;
  }
}
