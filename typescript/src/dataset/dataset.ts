import fs from "node:fs";
import path from "node:path";
import Papa from "papaparse";

import {
  convertGoldensToTestCases,
  convertConvoGoldensToConvoTestCases,
  stripPrivateFields,
  parseDelimited,
  safeJsonParse,
  convertConvoTestCasesToConvoGoldens,
  convertTestCasesToGoldens,
  formatTurns,
  goldenFromRecord,
  joinRetrievalContext,
  parseToolCalls,
  parseTurns,
  pickKey,
  reconstructRetrievalContext,
  serializeModels,
  serializeRetrievalContext,
  trimAndLoadJson,
  DEFAULT_GOLDEN_KEY_NAMES,
  type GoldenKeyNames,
} from "@/dataset/utils";
import { isConfident } from "@/utils";

import { Api, Endpoints, HttpMethods } from "@/confident/api";
import {
  CreateDatasetVersionResponse,
  DatasetHttpResponse,
  DatasetVersion,
  GetDatasetVersionsResponse,
} from "@/dataset/api";
import { ConversationalGolden, Golden } from "@/dataset/golden";
import { ConversationalTestCase, LLMTestCase } from "@/test-case";
import { asTestCaseString, asToolCalls } from "@/test-case/utils";
import type { MultiBar, SingleBar } from "cli-progress";
import { traceManager, Trace, BaseSpan } from "@/tracing/tracing";
import {
  evaluateTrace,
  countTraceMetrics,
  isDuplicateOfCase,
  primaryTraceFor,
} from "@/evaluate/trace-eval";
import { buildTestResult } from "@/evaluate/evaluate";
import { postTestRun } from "@/evaluate/confident";
import {
  processHyperparameters,
  type Hyperparameters,
} from "@/evaluate/hyperparameters";
import {
  printResultsTable,
  printCompletionSummary,
  printHyperparametersWarning,
  newProgressMultiBar,
} from "@/evaluate/console-report";
import type { TestResult, EvaluatedCase } from "@/evaluate/types";
import type { ErrorConfig, DisplayConfig } from "@/evaluate/configs";
import type { BaseMetric } from "@/metrics/base-metrics";

export type GoldenUnion = Golden | ConversationalGolden;
export type GoldenUnionArray = Golden[] | ConversationalGolden[];
export type TestCaseUnion = LLMTestCase | ConversationalTestCase;
export type TestCaseUnionArray = LLMTestCase[] | ConversationalTestCase[];

export interface LoadGoldensOptions {
  filePath: string;
  /** Anything left out keeps its {@link DEFAULT_GOLDEN_KEY_NAMES} default. */
  keys?: Partial<GoldenKeyNames>;
  /** Splits a `context` written as one delimited string rather than an array. */
  contextDelimiter?: string;
  retrievalContextDelimiter?: string;
  encoding?: BufferEncoding;
}

export const VALID_FILE_TYPES = ["csv", "json", "jsonl"] as const;
export type DatasetFileType = (typeof VALID_FILE_TYPES)[number];

const SINGLE_TURN_COLUMNS = [
  "input",
  "actual_output",
  "expected_output",
  "retrieval_context",
  "context",
  "name",
  "comments",
  "source_file",
  "tools_called",
  "expected_tools",
  "additional_metadata",
  "custom_column_key_values",
];

const MULTI_TURN_COLUMNS = [
  "scenario",
  "turns",
  "expected_outcome",
  "user_description",
  "context",
  "name",
  "comments",
  "additional_metadata",
  "custom_column_key_values",
];

/** Local-time `YYYYMMDD_HHMMSS`, matching Python's default file name. */
function fileTimestamp(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  );
}

const asJsonCell = (value: unknown): string | null =>
  value == null ? null : JSON.stringify(value);

function singleTurnRecord(
  golden: Golden,
  fileType: DatasetFileType,
): Record<string, unknown> {
  return {
    input: golden.input ?? null,
    actual_output: golden.actualOutput ?? null,
    expected_output: golden.expectedOutput ?? null,
    // jsonl flattens the list fields to delimited strings, as Python's does.
    retrieval_context:
      (fileType === "jsonl"
        ? joinRetrievalContext(golden.retrievalContext)
        : serializeRetrievalContext(golden.retrievalContext)) ?? null,
    context:
      (fileType === "jsonl" ? golden.context?.join("|") : golden.context) ??
      null,
    name: golden.name ?? null,
    comments: golden.comments ?? null,
    source_file: golden.sourceFile ?? null,
    tools_called: serializeModels(golden.toolsCalled) ?? null,
    expected_tools: serializeModels(golden.expectedTools) ?? null,
    additional_metadata: golden.additionalMetadata ?? null,
    custom_column_key_values: golden.customColumnKeyValues ?? null,
  };
}

function multiTurnRecord(
  golden: ConversationalGolden,
): Record<string, unknown> {
  return {
    scenario: golden.scenario ?? null,
    turns: golden.turns?.length ? JSON.parse(formatTurns(golden.turns)) : null,
    expected_outcome: golden.expectedOutcome ?? null,
    user_description: golden.userDescription ?? null,
    context: golden.context ?? null,
    name: golden.name ?? null,
    comments: golden.comments ?? null,
    additional_metadata: golden.additionalMetadata ?? null,
    custom_column_key_values: golden.customColumnKeyValues ?? null,
  };
}

function singleTurnCsvRow(golden: Golden): (string | null)[] {
  return [
    golden.input ?? null,
    golden.actualOutput ?? null,
    golden.expectedOutput ?? null,
    joinRetrievalContext(golden.retrievalContext) ?? null,
    golden.context?.join("|") ?? null,
    golden.name ?? null,
    golden.comments ?? null,
    golden.sourceFile ?? null,
    asJsonCell(serializeModels(golden.toolsCalled)),
    asJsonCell(serializeModels(golden.expectedTools)),
    asJsonCell(golden.additionalMetadata),
    asJsonCell(golden.customColumnKeyValues),
  ];
}

function multiTurnCsvRow(golden: ConversationalGolden): (string | null)[] {
  return [
    golden.scenario ?? null,
    golden.turns ? formatTurns(golden.turns) : null,
    golden.expectedOutcome ?? null,
    golden.userDescription ?? null,
    golden.context?.join("|") ?? null,
    golden.name ?? null,
    golden.comments ?? null,
    asJsonCell(golden.additionalMetadata),
    asJsonCell(golden.customColumnKeyValues),
  ];
}

async function readJsonArray(
  filePath: string,
  encoding: BufferEncoding,
): Promise<Record<string, unknown>[]> {
  let contents: string;
  try {
    contents = await fs.promises.readFile(filePath, encoding);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      throw new Error(`The file ${filePath} was not found.`);
    }
    throw error;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(contents);
  } catch {
    throw new Error(`The file ${filePath} is not a valid JSON file.`);
  }
  if (!Array.isArray(parsed)) {
    throw new Error(`The file ${filePath} must contain an array of objects.`);
  }
  return parsed as Record<string, unknown>[];
}

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
    if (golden instanceof Golden) {
      if (
        this._conversationalGoldens.length > 0 ||
        this._conversationalTestCases.length > 0
      ) {
        throw new TypeError("You cannot add 'Golden' to a multi-turn dataset.");
      }
      this._multiTurn = false;
      this._addGolden(golden);
    } else {
      if (this._goldens.length > 0 || this._llmTestCases.length > 0) {
        throw new TypeError(
          "You cannot add 'ConversationalGolden' to a single-turn dataset.",
        );
      }
      this._multiTurn = true;
      this._addConversationalGolden(golden);
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
      if (
        this._conversationalGoldens.length > 0 ||
        this._conversationalTestCases.length > 0
      ) {
        throw new TypeError(
          "You cannot add 'LLMTestCase' to a multi-turn dataset.",
        );
      }
      testCase._datasetRank = this._llmTestCases.length;
      this._llmTestCases.push(testCase);
    } else if (testCase instanceof ConversationalTestCase) {
      if (this._goldens.length > 0 || this._llmTestCases.length > 0) {
        throw new TypeError(
          "You cannot add 'ConversationalTestCase' to a single-turn dataset.",
        );
      }
      this._multiTurn = true;
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
    projectId?: string;
  }): Promise<void> {
    const {
      alias,
      finalized = true,
      autoConvertGoldensToTestCases = false,
      version,
      projectId,
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
      projectId,
    );

    const datasetData = result.data || result;

    const response: DatasetHttpResponse = {
      goldens: datasetData.goldens
        ? datasetData.goldens.map(
            (goldenData: any) =>
              new Golden({
                id: goldenData.id,
                input: goldenData.input,
                actualOutput: goldenData.actualOutput,
                expectedOutput: goldenData.expectedOutput,
                context: goldenData.context,
                retrievalContext: reconstructRetrievalContext(
                  goldenData.retrievalContext,
                ),
                toolsCalled: parseToolCalls(goldenData.toolsCalled),
                expectedTools: parseToolCalls(goldenData.expectedTools),
                additionalMetadata: goldenData.additionalMetadata,
                sourceFile: goldenData.sourceFile,
                comments: goldenData.comments,
                name: goldenData.name,
                customColumnKeyValues: goldenData.customColumnKeyValues,
              }),
          )
        : undefined,
      conversationalGoldens: datasetData.conversationalGoldens
        ? datasetData.conversationalGoldens.map(
            (goldenData: any) =>
              new ConversationalGolden({
                id: goldenData.id,
                scenario: goldenData.scenario,
                expectedOutcome: goldenData.expectedOutcome,
                userDescription: goldenData.userDescription,
                context: goldenData.context,
                additionalMetadata: goldenData.additionalMetadata,
                comments: goldenData.comments,
                name: goldenData.name,
                customColumnKeyValues: goldenData.customColumnKeyValues,
                turns: goldenData.turns
                  ? parseTurns(goldenData.turns)
                  : undefined,
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
    projectId?: string;
  }): Promise<void> {
    const { alias, finalized = true, version, projectId } = params;
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
    this.stripGoldenIds(body);
    console.log(`Pushing '${alias}' to Confident AI...`);
    const result = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.DATASET_ALIAS_ENDPOINT,
      body,
      undefined,
      undefined,
      { alias },
      projectId,
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
    projectId?: string;
  }): Promise<CreateDatasetVersionResponse> {
    const { alias, projectId } = params;
    const api = new Api();
    const result = await api.sendRequest(
      HttpMethods.POST,
      Endpoints.DATASET_ALIAS_VERSIONS_ENDPOINT,
      {},
      undefined,
      undefined,
      { alias },
      projectId,
    );
    const data = (result?.data ?? result) as CreateDatasetVersionResponse;
    this._alias = alias;
    this._id = data.id;
    this._version = data.version;
    console.log(`✅ New Dataset version successfully created: ${data.version}`);
    return data;
  }

  async getVersions(params: {
    alias: string;
    projectId?: string;
  }): Promise<DatasetVersion[]> {
    const { alias, projectId } = params;
    const api = new Api();
    const result = await api.sendRequest(
      HttpMethods.GET,
      Endpoints.DATASET_ALIAS_VERSIONS_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias },
      projectId,
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
    projectId?: string;
  }): Promise<void> {
    const { alias, goldens, printResponse = true, projectId } = params;
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
    this.stripGoldenIds(body);

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
      projectId,
    );

    const link = result?.link;
    if (link && printResponse) {
      console.log(
        `✅ Goldens successfully queued to Confident AI! Annotate & finalize at: ${link}`,
      );
    }
  }

  async delete(alias: string, projectId?: string): Promise<void> {
    const api = new Api();
    await api.sendRequest(
      HttpMethods.DELETE,
      Endpoints.DATASET_ALIAS_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias },
      projectId,
    );
    console.log("✅ Dataset successfully deleted from Confident AI!");
  }

  ////////////////////////////////////////////////////////
  // Golden Mutation Methods
  ////////////////////////////////////////////////////////

  private stripGoldenIds(body: any): void {
    for (const key of ["goldens", "conversationalGoldens"]) {
      const goldens = body?.[key];
      if (Array.isArray(goldens)) {
        for (const golden of goldens) {
          if (golden && typeof golden === "object") delete golden.id;
        }
      }
    }
  }

  /** A column the file lacks stays unset, so a metric still reports it missing. */
  async addTestCasesFromCSV({
    filePath,
    inputCol = "input",
    actualOutputCol = "actual_output",
    expectedOutputCol = "expected_output",
    contextCol = "context",
    contextDelimiter = ";",
    retrievalContextCol = "retrieval_context",
    retrievalContextDelimiter = ";",
    toolsCalledCol = "tools_called",
    expectedToolsCol = "expected_tools",
    additionalMetadataCol = "additional_metadata",
    encoding = "utf-8",
  }: {
    filePath: string;
    inputCol?: string;
    actualOutputCol?: string;
    expectedOutputCol?: string;
    contextCol?: string;
    contextDelimiter?: string;
    retrievalContextCol?: string;
    retrievalContextDelimiter?: string;
    toolsCalledCol?: string;
    expectedToolsCol?: string;
    additionalMetadataCol?: string;
    encoding?: BufferEncoding;
  }): Promise<LLMTestCase[]> {
    const csvData = await fs.promises.readFile(filePath, encoding);
    const { data, errors, meta } = Papa.parse<Record<string, string>>(csvData, {
      header: true,
      skipEmptyLines: true,
    });
    if (errors.length) {
      throw new Error(`CSV parse error: ${errors[0].message}`);
    }

    const columns = new Set(meta.fields ?? []);
    /** Undefined when the file has no such column at all. */
    const cell = (row: Record<string, string>, col: string) =>
      columns.has(col) ? row[col] : undefined;

    const testCases = data.map((row) => {
      const context = cell(row, contextCol);
      const retrievalContext = cell(row, retrievalContextCol);
      return new LLMTestCase({
        input: row[inputCol],
        actualOutput: row[actualOutputCol],
        expectedOutput: cell(row, expectedOutputCol),
        context:
          context === undefined
            ? undefined
            : parseDelimited(context, contextDelimiter),
        retrievalContext: reconstructRetrievalContext(
          retrievalContext === undefined
            ? undefined
            : parseDelimited(retrievalContext, retrievalContextDelimiter),
        ),
        toolsCalled: parseToolCalls(cell(row, toolsCalledCol) || undefined),
        expectedTools: parseToolCalls(cell(row, expectedToolsCol) || undefined),
        additionalMetadata: safeJsonParse(
          cell(row, additionalMetadataCol),
          undefined,
        ),
      });
    });

    for (const testCase of testCases) {
      this.addTestCase(testCase);
    }
    return testCases;
  }

  async addGoldensFromJSON(
    options: LoadGoldensOptions,
  ): Promise<GoldenUnionArray> {
    const records = await readJsonArray(
      options.filePath,
      options.encoding ?? "utf-8",
    );
    return this._addGoldensFromRecords(records, options);
  }

  /** Blank lines are skipped; any other bad line fails with its number. */
  async addGoldensFromJSONL(
    options: LoadGoldensOptions,
  ): Promise<GoldenUnionArray> {
    const contents = await fs.promises.readFile(
      options.filePath,
      options.encoding ?? "utf-8",
    );
    const records = contents
      .split("\n")
      .map((line, index) => ({ line: line.trim(), lineNumber: index + 1 }))
      .filter(({ line }) => line.length > 0)
      .map(({ line, lineNumber }) => {
        try {
          return JSON.parse(line) as Record<string, unknown>;
        } catch {
          throw new Error(
            `The file ${options.filePath} contains invalid JSON on line ${lineNumber}.`,
          );
        }
      });
    return this._addGoldensFromRecords(records, options);
  }

  private _addGoldensFromRecords(
    records: Record<string, unknown>[],
    options: LoadGoldensOptions,
  ): GoldenUnionArray {
    const keys = { ...DEFAULT_GOLDEN_KEY_NAMES, ...options.keys };
    const delimiters = {
      context: options.contextDelimiter ?? "|",
      retrievalContext: options.retrievalContextDelimiter ?? "|",
    };
    const goldens = records.map((record) =>
      goldenFromRecord(record, keys, delimiters),
    );
    for (const golden of goldens) {
      this.addGolden(golden);
    }
    return goldens as GoldenUnionArray;
  }

  /** Cells holding a list are split on their delimiter, defaulting to `|`. */
  async addGoldensFromCSV(
    options: LoadGoldensOptions,
  ): Promise<GoldenUnionArray> {
    const keys = { ...DEFAULT_GOLDEN_KEY_NAMES, ...options.keys };
    const csvData = await fs.promises.readFile(
      options.filePath,
      options.encoding ?? "utf-8",
    );
    const { data, errors, meta } = Papa.parse<Record<string, string>>(csvData, {
      header: true,
      skipEmptyLines: true,
    });
    if (errors.length) {
      throw new Error(`CSV parse error: ${errors[0].message}`);
    }

    // An empty cell reads as unset, except in a list column, where it is [].
    const listColumns = new Set([keys.context, keys.retrievalContext]);
    const jsonColumns = new Set([
      keys.additionalMetadata,
      keys.customColumnKeyValues,
    ]);
    const records = data.map((row) => {
      const record: Record<string, unknown> = {};
      for (const column of meta.fields ?? []) {
        const value = row[column];
        if (value === "" && !listColumns.has(column)) continue;
        record[column] =
          jsonColumns.has(column) && value ? trimAndLoadJson(value) : value;
      }
      return record;
    });

    return this._addGoldensFromRecords(records, options);
  }

  /** Every object needs an input and an actual output, unlike a golden. */
  async addTestCasesFromJSON(
    options: LoadGoldensOptions,
  ): Promise<LLMTestCase[]> {
    const keys = { ...DEFAULT_GOLDEN_KEY_NAMES, ...options.keys };
    const delimiters = {
      context: options.contextDelimiter ?? "|",
      retrievalContext: options.retrievalContextDelimiter ?? "|",
    };
    const records = await readJsonArray(
      options.filePath,
      options.encoding ?? "utf-8",
    );

    const goldens = records.map((record) => {
      if (
        pickKey(record, keys.input) === undefined ||
        pickKey(record, keys.actualOutput) === undefined
      ) {
        throw new Error(
          `Required keys '${keys.input}' and '${keys.actualOutput}' are missing in one or more JSON objects.`,
        );
      }
      return goldenFromRecord(record, keys, delimiters) as Golden;
    });

    const testCases = convertGoldensToTestCases(goldens);
    for (const testCase of testCases) {
      this.addTestCase(testCase);
    }
    return testCases;
  }

  /** Returns the path written. The layout matches Python's `save_as`. */
  async saveAs(options: {
    fileType: DatasetFileType;
    directory: string;
    fileName?: string;
    /** Also write the dataset's test cases, converted to goldens. */
    includeTestCases?: boolean;
  }): Promise<string> {
    const { fileType, directory, fileName, includeTestCases = false } = options;
    if (!VALID_FILE_TYPES.includes(fileType)) {
      throw new Error(
        `Invalid file type. Available file types to save as: ${VALID_FILE_TYPES.join(", ")}`,
      );
    }

    const goldens: GoldenUnion[] = [...this.goldens];
    if (includeTestCases) {
      goldens.push(
        ...(this._multiTurn
          ? convertConvoTestCasesToConvoGoldens(this._conversationalTestCases)
          : convertTestCasesToGoldens(this._llmTestCases)),
      );
    }
    if (goldens.length === 0) {
      throw new Error(
        `No goldens found. Please generate goldens before attempting to save data as ${fileType}`,
      );
    }

    await fs.promises.mkdir(directory, { recursive: true });
    const fullFilePath = path.join(
      directory,
      `${fileName ?? fileTimestamp()}.${fileType}`,
    );

    let contents: string;
    if (fileType === "csv") {
      contents = Papa.unparse({
        fields: this._multiTurn ? MULTI_TURN_COLUMNS : SINGLE_TURN_COLUMNS,
        data: goldens.map((golden) =>
          this._multiTurn
            ? multiTurnCsvRow(golden as ConversationalGolden)
            : singleTurnCsvRow(golden as Golden),
        ),
      });
    } else {
      const records = goldens.map((golden) =>
        this._multiTurn
          ? multiTurnRecord(golden as ConversationalGolden)
          : singleTurnRecord(golden as Golden, fileType),
      );
      contents =
        fileType === "jsonl"
          ? records.map((record) => JSON.stringify(record)).join("\n") + "\n"
          : JSON.stringify(records, null, 4);
    }

    await fs.promises.writeFile(fullFilePath, contents, "utf-8");
    console.log(`Evaluation dataset saved at ${fullFilePath}!`);
    return fullFilePath;
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
      identifier?: string;
      hyperparameters?: Hyperparameters;
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
    const unsubscribe = traceManager.addTraceCaptureSink((t) =>
      captured.push(t),
    );
    // Tells integrations that spans must be materialised in-process for this run
    // instead of being exported straight to Confident AI.
    const endEvaluation = traceManager.beginEvaluation();
    const allCases: EvaluatedCase[] = [];
    /** Span-level cases for the local report, paired with their golden's case. */
    const componentCases: Array<{ case: EvaluatedCase; parentIndex: number }> =
      [];
    const startTime = Date.now();
    let count = 0;
    try {
      for (const golden of goldens) {
        const start = captured.length;
        yield golden;
        // Resumed: the agent ran in the loop body — evaluate the traces it produced.
        await traceManager.awaitSettled();
        callbackBar?.increment();
        count += 1;
        const newTraces = captured.slice(start);
        const traceGolden = golden as Golden;

        const primary = primaryTraceFor(newTraces);

        for (const trace of newTraces) {
          // Trace-level metrics judge the turn, so they belong to the reported
          // trace only — attaching them to every trace would score fragments and
          // multiply the metric runs.
          if (metrics.length > 0 && trace === primary) {
            metrics.forEach((m) => (m.showIndicator = false));
            trace.metrics = [...(trace.metrics ?? []), ...metrics];
          }
          suppressSpinners(trace.rootSpans);
        }
        const total = newTraces.reduce(
          (s, t) =>
            s + countTraceMetrics(t, t === primary ? traceGolden : undefined),
          0,
        );
        const evalBar = multibar?.create(Math.max(total, 1), 0, {
          label: `     🎯 Evaluating component(s) (#${count})`,
        });
        const parentIndex = allCases.length;
        for (const trace of newTraces) {
          // Run all metrics (span + trace); attaches metricsData to each scope.
          const evaluated = await evaluateTrace(trace, {
            errorConfig: options.errorConfig,
            onMetric: () => evalBar?.increment(),
            golden: trace === primary ? traceGolden : undefined,
          });
          // Span-level results are reported locally only — the posted test run
          // keeps one case per golden, with the per-span scores riding along
          // inside the embedded trace (mirrors Python).
          for (const _case of evaluated) {
            if (_case.isTraceScope || _case.metricsData.length === 0) continue;
            componentCases.push({ case: _case, parentIndex });
          }
        }

        if (primary) {
          const rootOutput = primary.output ?? primary.rootSpans?.[0]?.output;
          const testCase = new LLMTestCase({
            input: traceGolden.input,
            actualOutput:
              rootOutput != null
                ? asTestCaseString(rootOutput)
                : (traceGolden.actualOutput ?? "None"),
            expectedOutput: primary.expectedOutput,
            context: primary.context,
            retrievalContext: primary.retrievalContext,
            toolsCalled: asToolCalls(primary.toolsCalled),
            expectedTools: asToolCalls(primary.expectedTools),
            // Links the posted run back to the dataset these goldens came from.
            _datasetAlias: traceGolden._datasetAlias,
            _datasetId: traceGolden._datasetId,
            _datasetRank: traceGolden._datasetRank,
          });
          const { confidentApiKey: _omit, ...traceApi } =
            traceManager.createTraceApi(primary);
          allCases.push({
            testCase,
            metricsData: primary.metricsData ?? [],
            runDuration: 0,
            trace: traceApi,
          });
        }
        evalBar?.update(Math.max(total, 1));
        mainBar?.increment();
      }
    } finally {
      unsubscribe();
      endEvaluation();
      multibar?.stop();
    }
    const runDuration = (Date.now() - startTime) / 1000;
    const goldenResults: TestResult[] = allCases.map((c, i) =>
      buildTestResult(i, c.testCase, c.metricsData),
    );
    const componentResults: TestResult[] = [];
    for (const { case: c, parentIndex } of componentCases) {
      const result = buildTestResult(
        goldenResults.length + componentResults.length,
        c.testCase,
        c.metricsData,
      );
      if (!isDuplicateOfCase(result, goldenResults[parentIndex])) {
        componentResults.push(result);
      }
    }
    const results: TestResult[] = [...goldenResults, ...componentResults];
    this._evalResults = results;

    const hyperparameters = processHyperparameters(options.hyperparameters);
    const printResults = options.displayConfig?.printResults ?? true;
    if (printResults && results.length > 0) {
      printResultsTable(results, {
        truncatePassing: options.displayConfig?.truncatePassingCases ?? true,
      });
      printHyperparametersWarning(hyperparameters);
    }

    // Post a TestRun to Confident AI (mirrors Python's evals_iterator); silent so
    // we control the wrap-up message below.
    const { link } = await postTestRun(allCases, runDuration, {
      silent: true,
      identifier: options.identifier,
      hyperparameters,
    });

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
