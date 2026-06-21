import fs from "node:fs";
import Papa from "papaparse";

import {
  convertGoldensToTestCases,
  convertConvoGoldensToConvoTestCases,
  stripPrivateFields,
  parseDelimited,
  safeJsonParse,
} from "./utils";
import { isConfident } from "../utils";

import { Api, Endpoints, HttpMethods } from "../confident/api";
import {
  CreateDatasetVersionResponse,
  DatasetHttpResponse,
  DatasetVersion,
  GetDatasetVersionsResponse,
} from "./api";
import { ConversationalGolden, Golden } from "./golden";
import { ConversationalTestCase, LLMTestCase } from "../test-case";
import type { MultiBar, SingleBar } from "cli-progress";
import { traceManager, Trace, BaseSpan } from "../tracing/tracing";
import { evaluateTrace, countTraceMetrics } from "../evaluate/trace-eval";
import { buildTestResult } from "../evaluate/evaluate";
import { postTestRun } from "../evaluate/confident";
import {
  printResultsTable,
  printCompletionSummary,
  printHyperparametersWarning,
  newProgressMultiBar,
} from "../evaluate/console-report";
import type { TestResult, EvaluatedCase } from "../evaluate/types";
import type { ErrorConfig, DisplayConfig } from "../evaluate/configs";
import type { BaseMetric } from "../metrics/base-metrics";

export type GoldenUnion = Golden | ConversationalGolden;
export type GoldenUnionArray = Golden[] | ConversationalGolden[];
export type TestCaseUnion = LLMTestCase | ConversationalTestCase;
export type TestCaseUnionArray = LLMTestCase[] | ConversationalTestCase[];

export class EvaluationDataset {
  private _multiTurn: boolean | null = null;
  private _alias: string | null = null;
  private _id: string | null = null;
  private _version: string | null = null;

  private _goldens: Golden[] = [];
  private _conversationalGoldens: ConversationalGolden[] = [];

  private _llmTestCases: LLMTestCase[] = [];
  private _conversationalTestCases: ConversationalTestCase[] = [];
  private _evalResults: TestResult[] = [];

  constructor(params: { goldens?: GoldenUnionArray } = {}) {
    this._alias = null;
    this._id = null;
    this._version = null;
    const goldens = params.goldens ?? [];
    if (goldens.length > 0) {
      this._multiTurn = goldens[0] instanceof ConversationalGolden;
    }
    this._goldens = [];
    this._conversationalGoldens = [];
    for (const golden of goldens) {
      golden._datasetRank = goldens.length;
      if (this._multiTurn) {
        this._addConversationalGolden(golden);
      } else {
        this._addGolden(golden);
      }
    }
    this._llmTestCases = [];
    this._conversationalTestCases = [];
  }

  toString(): string {
    return `${this.constructor.name}(test_cases=${JSON.stringify(
      this.testCases,
    )}, goldens=${JSON.stringify(this.goldens)}, _alias=${this._alias}, _id=${
      this._id
    }, _multi_turn=${this._multiTurn})`;
  }

  ////////////////////////////////////////////////////////
  // Golden Properties
  ////////////////////////////////////////////////////////

  get goldens(): GoldenUnionArray {
    return this._multiTurn ? this._conversationalGoldens : this._goldens;
  }

  set goldens(goldens: GoldenUnionArray) {
    const prevGoldens = this._goldens;
    const prevConvGoldens = this._conversationalGoldens;
    this._goldens = [];
    this._conversationalGoldens = [];
    try {
      for (const golden of goldens) {
        if (
          !(golden instanceof Golden) &&
          !(golden instanceof ConversationalGolden)
        ) {
          throw new TypeError(
            "Your goldens must be instances of either ConversationalGolden or Golden",
          );
        }
        golden._datasetAlias = this._alias ?? undefined;
        golden._datasetId = this._id ?? undefined;
        golden._datasetRank = goldens.length;
        if (this._multiTurn) {
          this._addConversationalGolden(golden);
        } else {
          this.addGolden(golden);
        }
      }
    } catch (e) {
      this._goldens = prevGoldens;
      this._conversationalGoldens = prevConvGoldens;
      throw e;
    }
  }

  addGolden(golden: GoldenUnion): void {
    if (this._multiTurn === null) {
      this._multiTurn = golden instanceof ConversationalGolden;
    }
    if (this._multiTurn) {
      this._addConversationalGolden(golden);
    } else {
      this._addGolden(golden);
    }
  }

  private _addGolden(golden: GoldenUnion): void {
    if (golden instanceof Golden) {
      this._goldens.push(golden);
    } else {
      throw new TypeError(
        "You cannot add a multi-turn ConversationalGolden to a single-turn dataset. You can only add a Golden.",
      );
    }
  }

  private _addConversationalGolden(golden: GoldenUnion): void {
    if (golden instanceof ConversationalGolden) {
      this._conversationalGoldens.push(golden);
    } else {
      throw new TypeError(
        "You cannot add a single-turn Golden to a multi-turn dataset. You can only add a ConversationalGolden.",
      );
    }
  }

  ////////////////////////////////////////////////////////
  // Test Case Properties
  ////////////////////////////////////////////////////////

  get testCases(): TestCaseUnionArray {
    return this._multiTurn ? this._conversationalTestCases : this._llmTestCases;
  }

  set testCases(testCases: TestCaseUnionArray) {
    const llmTestCases: LLMTestCase[] = [];
    const conversationalTestCases: ConversationalTestCase[] = [];

    for (const testCase of testCases) {
      if (
        !(testCase instanceof LLMTestCase) &&
        !(testCase instanceof ConversationalTestCase)
      ) {
        continue;
      }

      testCase._datasetAlias = this._alias ?? undefined;
      testCase._datasetId = this._id ?? undefined;

      if (testCase instanceof LLMTestCase) {
        testCase._datasetRank = llmTestCases.length;
        llmTestCases.push(testCase);
      } else if (testCase instanceof ConversationalTestCase) {
        testCase._datasetRank = conversationalTestCases.length;
        conversationalTestCases.push(testCase);
      }
    }

    this._llmTestCases = llmTestCases;
    this._conversationalTestCases = conversationalTestCases;
  }

  addTestCase(testCase: TestCaseUnion): void {
    testCase._datasetAlias = this._alias ?? undefined;
    testCase._datasetId = this._id ?? undefined;
    if (testCase instanceof LLMTestCase) {
      testCase._datasetRank = this._llmTestCases.length;
      this._llmTestCases.push(testCase);
    } else if (testCase instanceof ConversationalTestCase) {
      testCase._datasetRank = this._conversationalTestCases.length;
      this._conversationalTestCases.push(testCase);
    }
  }

  ////////////////////////////////////////////////////////
  // Push and Pull Methods
  ////////////////////////////////////////////////////////

  async pull(params: {
    alias: string;
    finalized?: boolean;
    autoConvertGoldensToTestCases?: boolean;
    version?: string;
  }): Promise<void> {
    const {
      alias,
      finalized = true,
      autoConvertGoldensToTestCases = false,
      version,
    } = params;
    if (!isConfident()) {
      throw new Error("Set CONFIDENT_API_KEY to pull dataset.");
    }
    console.log(`Pulling '${alias}' from Confident AI...`);

    const api = new Api();
    const startTime = performance.now();
    const queryParams: Record<string, string> = {
      finalized: finalized.toString().toLowerCase(),
    };
    if (version !== undefined) {
      queryParams.version = version;
    }
    const result = await api.sendRequest(
      HttpMethods.GET,
      Endpoints.DATASET_ALIAS_ENDPOINT,
      undefined,
      queryParams,
      undefined,
      { alias },
    );

    const datasetData = result.data || result;

    const response: DatasetHttpResponse = {
      goldens: datasetData.goldens
        ? datasetData.goldens.map(
            (goldenData: any) =>
              new Golden({
                input: goldenData.input,
                actualOutput: goldenData.actualOutput,
                expectedOutput: goldenData.expectedOutput,
                context: goldenData.context,
                retrievalContext: goldenData.retrievalContext,
                toolsCalled: goldenData.toolsCalled,
                expectedTools: goldenData.expectedTools,
                additionalMetadata: goldenData.additionalMetadata,
                sourceFile: goldenData.sourceFile,
                comments: goldenData.comments,
              }),
          )
        : undefined,
      conversationalGoldens: datasetData.conversationalGoldens
        ? datasetData.conversationalGoldens.map(
            (goldenData: any) =>
              new ConversationalGolden({
                scenario: goldenData.scenario,
                expectedOutcome: goldenData.expectedOutcome,
                userDescription: goldenData.userDescription,
                context: goldenData.context,
                additionalMetadata: goldenData.additionalMetadata,
                comments: goldenData.comments,
                name: goldenData.name,
                customColumnKeyValues: goldenData.customColumnKeyValues,
                turns: goldenData.turns,
                _datasetRank: goldenData._datasetRank,
                _datasetAlias: goldenData._datasetAlias,
                _datasetId: goldenData._datasetId,
              }),
          )
        : undefined,
      id: datasetData.id,
      version: datasetData.version ?? null,
    };

    this._alias = alias;
    this._id = response.id;
    this._version = response.version ?? null;
    this._multiTurn = datasetData.goldens === undefined;
    this.goldens = [];
    this.testCases = [];

    if (autoConvertGoldensToTestCases) {
      if (!this._multiTurn) {
        const llmTestCases = convertGoldensToTestCases(
          response.goldens,
          alias,
          response.id,
        );
        this._llmTestCases.push(...llmTestCases);
      } else {
        const conversationalTestCases = convertConvoGoldensToConvoTestCases(
          response.conversationalGoldens,
          alias,
          response.id,
        );
        this._conversationalTestCases.push(...conversationalTestCases);
      }
    } else {
      if (!this._multiTurn) {
        this.goldens = response.goldens;
      } else {
        this.goldens = response.conversationalGoldens;
      }
      for (const golden of this.goldens) {
        golden._datasetAlias = alias;
        golden._datasetId = response.id;
      }
    }

    const endTime = performance.now();
    const timeTaken = ((endTime - startTime) / 1000).toFixed(2);
    console.log(`Done! (${timeTaken}s)`);
  }

  async push(params: {
    alias: string;
    finalized?: boolean;
    version?: string;
  }): Promise<void> {
    const { alias, finalized = true, version } = params;
    if (this.goldens.length === 0) {
      throw new Error(
        "Unable to push empty dataset to Confident AI, there must be at least one golden in dataset.",
      );
    }
    const api = new Api();
    const apiDataset: Record<string, unknown> = {
      finalized: finalized,
      goldens: !this._multiTurn ? this.goldens : undefined,
      conversationalGoldens: this._multiTurn ? this.goldens : undefined,
    };
    if (version !== undefined) {
      apiDataset.version = version;
    }
    const body = stripPrivateFields(JSON.parse(JSON.stringify(apiDataset)));
    console.log(`Pushing '${alias}' to Confident AI...`);
    const result = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.DATASET_ALIAS_ENDPOINT,
      body,
      undefined,
      undefined,
      { alias },
    );
    const link = result?.link;
    if (link) {
      console.log(
        `✅ Dataset successfully pushed to Confident AI! View at: ${link}`,
      );
    }
  }

  ////////////////////////////////////////////////////////
  // Version Methods
  ////////////////////////////////////////////////////////

  async createVersion(params: {
    alias: string;
  }): Promise<CreateDatasetVersionResponse> {
    const { alias } = params;
    const api = new Api();
    const result = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.DATASET_ALIAS_VERSIONS_ENDPOINT,
      {},
      undefined,
      undefined,
      { alias },
    );
    const data = (result?.data ?? result) as CreateDatasetVersionResponse;
    this._alias = alias;
    this._id = data.id;
    this._version = data.version;
    console.log(`✅ New Dataset version successfully created: ${data.version}`);
    return data;
  }

  async getVersions(params: { alias: string }): Promise<DatasetVersion[]> {
    const { alias } = params;
    const api = new Api();
    const result = await api.sendRequest(
      HttpMethods.GET,
      Endpoints.DATASET_ALIAS_VERSIONS_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias },
    );
    const data = (result?.data ?? result) as GetDatasetVersionsResponse;
    return data.versions ?? [];
  }

  ////////////////////////////////////////////////////////
  // Queue Methods
  ////////////////////////////////////////////////////////

  async queue(params: {
    alias: string;
    goldens: Array<Golden | ConversationalGolden>;
    printResponse?: boolean;
  }): Promise<void> {
    const { alias, goldens, printResponse = true } = params;
    if (!goldens || goldens.length === 0) {
      throw new Error(
        `Can't queue empty list of goldens to dataset with alias: ${alias} on Confident AI.`,
      );
    }
    const api = new Api();
    const isMultiTurn = goldens[0] instanceof ConversationalGolden;

    const apiDataset = {
      goldens: !isMultiTurn ? goldens : undefined,
      conversationalGoldens: isMultiTurn ? goldens : undefined,
    };
    const body = stripPrivateFields(apiDataset);

    console.log(
      `Queueing ${goldens.length} golden(s) to '${alias}' on Confident AI...`,
    );

    const result = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.DATASET_ALIAS_QUEUE_ENDPOINT,
      body,
      undefined,
      undefined,
      { alias },
    );

    const link = result?.link;
    if (link && printResponse) {
      console.log(
        `✅ Goldens successfully queued to Confident AI! Annotate & finalize at: ${link}`,
      );
    }
  }

  async delete(alias: string): Promise<void> {
    const api = new Api();
    await api.sendRequest(
      HttpMethods.DELETE,
      Endpoints.DATASET_ALIAS_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias },
    );
    console.log("✅ Dataset successfully deleted from Confident AI!");
  }

  async addTestCasesFromCSV({
    filePath,
    inputCol,
    actualOutputCol,
    expectedOutputCol,
    contextCol,
    contextDelimiter = ";",
    retrievalContextCol,
    retrievalContextDelimiter = ";",
    toolsCalledCol,
    expectedToolsCol,
    additionalMetadataCol,
  }: {
    filePath: string;
    inputCol: string;
    actualOutputCol: string;
    expectedOutputCol?: string;
    contextCol?: string;
    contextDelimiter?: string;
    retrievalContextCol?: string;
    retrievalContextDelimiter?: string;
    toolsCalledCol?: string;
    expectedToolsCol?: string;
    additionalMetadataCol?: string;
  }) {
    const csvData = fs.readFileSync(filePath, "utf8");
    const { data, errors } = Papa.parse<Record<string, string>>(csvData, {
      header: true,
      skipEmptyLines: true,
    });
    if (errors.length) {
      throw new Error(`CSV parse error: ${errors[0].message}`);
    }

    return data.map(
      (row) =>
        new LLMTestCase({
          input: row[inputCol],
          actualOutput: row[actualOutputCol],
          expectedOutput: expectedOutputCol
            ? row[expectedOutputCol]
            : undefined,
          context: parseDelimited(row[contextCol!], contextDelimiter),
          retrievalContext: parseDelimited(
            row[retrievalContextCol!],
            retrievalContextDelimiter,
          ),
          toolsCalled: safeJsonParse(row[toolsCalledCol!], []),
          expectedTools: safeJsonParse(row[expectedToolsCol!], []),
          additionalMetadata: safeJsonParse(
            row[additionalMetadataCol!],
            undefined,
          ),
        }),
    );
  }

  get evalResults(): TestResult[] {
    return this._evalResults;
  }

  /**
   * Traced/agentic evaluation loop (TS port of Python's `evals_iterator`). Yields
   * each golden; you run your `observe`-wrapped agent in the loop body, and the
   * trace it produces is evaluated with `metrics` (trace-level) plus any metrics
   * attached to spans via `observe`/`updateCurrentSpan`. Results are printed at
   * the end and available via `dataset.evalResults`.
   *
   * @example
   * for await (const golden of dataset.evalsIterator({ metrics: [taskCompletion] })) {
   *   await myAgent(golden.input);
   * }
   */
  async *evalsIterator(
    options: {
      metrics?: BaseMetric[];
      errorConfig?: ErrorConfig;
      displayConfig?: DisplayConfig;
    } = {},
  ): AsyncGenerator<GoldenUnion> {
    const goldens = this.goldens;
    const metrics = options.metrics ?? [];
    const showIndicator = options.displayConfig?.showIndicator ?? true;

    let multibar: MultiBar | null = null;
    let mainBar: SingleBar | null = null;
    let callbackBar: SingleBar | null = null;
    if (showIndicator && goldens.length > 0) {
      multibar = newProgressMultiBar();
      mainBar = multibar.create(goldens.length, 0, {
        label: "Running Component-Level Evals",
      });
      callbackBar = multibar.create(goldens.length, 0, {
        label: `\t⚡ Calling LLM app (with ${goldens.length} goldens)`,
      });
    }

    // Suppress per-metric spinners (the bars are the progress UI).
    const suppressSpinners = (spans: BaseSpan[]) => {
      for (const s of spans) {
        (s.metrics ?? []).forEach((m) => (m.showIndicator = false));
        suppressSpinners(s.children ?? []);
      }
    };

    const captured: Trace[] = [];
    traceManager.setTraceCaptureSink((t) => captured.push(t));
    const allCases: EvaluatedCase[] = [];
    const startTime = Date.now();
    let count = 0;
    try {
      for (const golden of goldens) {
        const start = captured.length;
        yield golden;
        // Resumed: the agent ran in the loop body — evaluate the traces it produced.
        callbackBar?.increment();
        count += 1;
        const newTraces = captured.slice(start);
        for (const trace of newTraces) {
          if (metrics.length > 0) {
            metrics.forEach((m) => (m.showIndicator = false));
            trace.metrics = [...(trace.metrics ?? []), ...metrics];
          }
          suppressSpinners(trace.rootSpans);
        }
        const total = newTraces.reduce((s, t) => s + countTraceMetrics(t), 0);
        const evalBar = multibar?.create(Math.max(total, 1), 0, {
          label: `     🎯 Evaluating component(s) (#${count})`,
        });
        for (const trace of newTraces) {
          // Run all metrics (span + trace); attaches metricsData to each scope.
          await evaluateTrace(trace, {
            errorConfig: options.errorConfig,
            onMetric: () => evalBar?.increment(),
          });
          // One golden-based test case per trace, with the trace embedded — so
          // the platform shows the I/O and links to the trace (mirrors Python).
          const g = golden as Golden;
          const rootOutput = trace.output ?? trace.rootSpans?.[0]?.output;
          const testCase = new LLMTestCase({
            input: g.input,
            actualOutput:
              rootOutput != null
                ? String(rootOutput)
                : (g.actualOutput ?? "None"),
            expectedOutput: trace.expectedOutput,
            context: trace.context,
            retrievalContext: trace.retrievalContext,
            toolsCalled: trace.toolsCalled,
            expectedTools: trace.expectedTools,
          });
          const { confidentApiKey: _omit, ...traceApi } =
            traceManager.createTraceApi(trace);
          allCases.push({
            testCase,
            metricsData: trace.metricsData ?? [],
            runDuration: 0,
            trace: traceApi,
          });
        }
        evalBar?.update(Math.max(total, 1));
        mainBar?.increment();
      }
    } finally {
      traceManager.setTraceCaptureSink(undefined);
      multibar?.stop();
    }
    const runDuration = (Date.now() - startTime) / 1000;
    const results: TestResult[] = allCases.map((c, i) =>
      buildTestResult(i, c.testCase, c.metricsData),
    );
    this._evalResults = results;

    const printResults = options.displayConfig?.printResults ?? true;
    if (printResults && results.length > 0) {
      printResultsTable(results, {
        truncatePassing: options.displayConfig?.truncatePassingCases ?? true,
      });
      printHyperparametersWarning();
    }

    // Post a TestRun to Confident AI (mirrors Python's evals_iterator); silent so
    // we control the wrap-up message below.
    const { link } = await postTestRun(allCases, runDuration, false, true);

    if (printResults && results.length > 0) {
      if (link) {
        console.log(`\n✓ Done 🎉! View results on ${link}`);
      } else {
        const tokenCost = results
          .flatMap((r) => r.metricsData ?? [])
          .reduce((s, m) => s + (m.evaluationCost ?? 0), 0);
        const passed = results.filter((r) => r.success).length;
        printCompletionSummary({
          runDuration,
          tokenCost,
          passed,
          failed: results.length - passed,
        });
      }
    }
  }
}
