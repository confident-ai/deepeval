export abstract class DeepEvalBaseLLM {
  modelName?: string;
  model: any;

  constructor(modelName?: string, ...args: any[]) {
    this.modelName = this.parseModelName(modelName);
    this.model = this.loadModel(...args);
  }

  /**
   * Parses the input model name, can be overridden by subclass.
   */
  protected parseModelName(modelName?: string): string | undefined {
    return modelName;
  }

  /**
   * Loads a model, that will be responsible for scoring.
   * @param args Model loading arguments
   * @returns The loaded model
   */
  abstract loadModel(...args: any[]): DeepEvalBaseLLM;

  /**
   * Runs the model to output an LLM response.
   * @param args Generation arguments
   * @returns The generated string
   */
  abstract generate(...args: any[]): string | Promise<string>;

  /**
   * Runs the model to output LLM responses in batch mode.
   * @param args Generation arguments
   * @returns The list of generated strings
   * @throws Error if not implemented
   */
  batchGenerate(..._args: any[]): string[] {
    throw new Error("batchGenerate is not implemented.");
  }

  /**
   * Returns the name of the model.
   * @param args Arguments as needed
   * @returns The model name string
   */
  abstract getModelName(...args: any[]): string;
}
