import * as fs from "fs";
import * as os from "os";
import * as path from "path";

import { EvaluationDataset, Golden, ConversationalGolden } from "@/dataset";
import { LLMTestCase, RetrievedContextData, ToolCall, Turn } from "@/test-case";

describe("Dataset file loading", () => {
  let tempDir: string;

  beforeAll(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-dataset-"));
  });

  afterAll(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  const write = (fileName: string, contents: string): string => {
    const filePath = path.join(tempDir, fileName);
    fs.writeFileSync(filePath, contents, "utf-8");
    return filePath;
  };

  const writeJson = (fileName: string, value: unknown): string =>
    write(fileName, JSON.stringify(value));

  test("loads single-turn goldens from a JSON file", async () => {
    const filePath = writeJson("goldens.json", [
      {
        input: "What is the capital of Germany?",
        actual_output: "Berlin.",
        expected_output: "Berlin",
        context: ["Geography"],
        retrieval_context: ["Germany is in Central Europe."],
        source_file: "atlas.pdf",
        name: "capital",
        comments: "seeded",
        additional_metadata: { topic: "geography" },
      },
      { input: "What is 2 + 2?", expected_output: "4" },
    ]);

    const dataset = new EvaluationDataset();
    await dataset.addGoldensFromJSON({ filePath });

    expect(dataset.goldens.length).toBe(2);
    const golden = dataset.goldens[0] as Golden;
    expect(golden).toBeInstanceOf(Golden);
    expect(golden.input).toBe("What is the capital of Germany?");
    expect(golden.actualOutput).toBe("Berlin.");
    expect(golden.context).toEqual(["Geography"]);
    expect(golden.retrievalContext).toEqual(["Germany is in Central Europe."]);
    expect(golden.sourceFile).toBe("atlas.pdf");
    expect(golden.name).toBe("capital");
    expect(golden.comments).toBe("seeded");
    expect(golden.additionalMetadata).toEqual({ topic: "geography" });
  });

  test("accepts camelCase keys and custom key overrides", async () => {
    const camelPath = writeJson("camel.json", [
      { input: "hi", actualOutput: "hello", expectedOutput: "hello there" },
    ]);
    const camelDataset = new EvaluationDataset();
    await camelDataset.addGoldensFromJSON({ filePath: camelPath });
    expect((camelDataset.goldens[0] as Golden).actualOutput).toBe("hello");

    const customPath = writeJson("custom.json", [
      { query: "hi", response: "hello" },
    ]);
    const customDataset = new EvaluationDataset();
    await customDataset.addGoldensFromJSON({
      filePath: customPath,
      keys: { input: "query", actualOutput: "response" },
    });
    expect((customDataset.goldens[0] as Golden).input).toBe("hi");
    expect((customDataset.goldens[0] as Golden).actualOutput).toBe("hello");
  });

  test("parses tool calls into ToolCall instances", async () => {
    const filePath = writeJson("tools.json", [
      {
        input: "Book a flight",
        tools_called: [
          {
            name: "search_flights",
            input_parameters: { to: "SFO" },
            output: "3 results",
          },
        ],
        expected_tools: [{ name: "book_flight" }],
      },
    ]);

    const dataset = new EvaluationDataset();
    await dataset.addGoldensFromJSON({ filePath });

    const golden = dataset.goldens[0] as Golden;
    expect(golden.toolsCalled?.[0]).toBeInstanceOf(ToolCall);
    expect(golden.toolsCalled?.[0].name).toBe("search_flights");
    expect(golden.toolsCalled?.[0].inputParameters).toEqual({ to: "SFO" });
    expect(golden.expectedTools?.[0].name).toBe("book_flight");
  });

  test("rebuilds RetrievedContextData from its serialized marker", async () => {
    const filePath = writeJson("retrieval.json", [
      {
        input: "What is DeepEval?",
        retrieval_context: [
          "deepeval_source=docs.md,deepeval_context=An LLM eval framework.",
          "A plain string.",
        ],
      },
    ]);

    const dataset = new EvaluationDataset();
    await dataset.addGoldensFromJSON({ filePath });

    const [retrieved, plain] = (dataset.goldens[0] as Golden).retrievalContext!;
    expect(retrieved).toBeInstanceOf(RetrievedContextData);
    expect((retrieved as RetrievedContextData).source).toBe("docs.md");
    expect((retrieved as RetrievedContextData).context).toBe(
      "An LLM eval framework.",
    );
    expect(plain).toBe("A plain string.");
  });

  test("loads multi-turn goldens when a record carries a scenario", async () => {
    const filePath = writeJson("conversational.json", [
      {
        scenario: "A user asks for help evaluating an LLM app.",
        expected_outcome: "The user understands evaluation datasets.",
        user_description: "A first-time user.",
        context: ["DeepEval supports evaluation datasets."],
        turns: [
          { role: "user", content: "How do I evaluate my app?" },
          {
            role: "assistant",
            content: "Start with a dataset.",
            retrieval_context: ["docs"],
            metadata: { source: "docs" },
          },
        ],
      },
    ]);

    const dataset = new EvaluationDataset();
    await dataset.addGoldensFromJSON({ filePath });

    const golden = dataset.goldens[0] as ConversationalGolden;
    expect(golden).toBeInstanceOf(ConversationalGolden);
    expect(golden.turns?.length).toBe(2);
    expect(golden.turns?.[1].role).toBe("assistant");
    expect(golden.turns?.[1].additionalMetadata).toEqual({ source: "docs" });
    expect(golden.userDescription).toBe("A first-time user.");
  });

  test("refuses to mix single-turn and multi-turn goldens", async () => {
    const filePath = writeJson("mixed.json", [
      { input: "What is DeepEval?" },
      { scenario: "A user asks about DeepEval." },
    ]);

    const dataset = new EvaluationDataset();
    await expect(dataset.addGoldensFromJSON({ filePath })).rejects.toThrow(
      "You cannot add 'ConversationalGolden' to a single-turn dataset.",
    );
  });

  test("loads goldens from a JSONL file, skipping blank lines", async () => {
    const filePath = write(
      "goldens.jsonl",
      [
        '{"input": "What is DeepEval?", "expected_output": "A framework."}',
        "",
        '{"input": "What is a golden?", "context": "a|b"}',
        "",
      ].join("\n"),
    );

    const dataset = new EvaluationDataset();
    await dataset.addGoldensFromJSONL({ filePath });

    expect(dataset.goldens.length).toBe(2);
    expect((dataset.goldens[1] as Golden).context).toEqual(["a", "b"]);
  });

  test("reports the line number of malformed JSONL", async () => {
    const filePath = write(
      "broken.jsonl",
      ['{"input": "ok"}', "{not json}"].join("\n"),
    );

    const dataset = new EvaluationDataset();
    await expect(dataset.addGoldensFromJSONL({ filePath })).rejects.toThrow(
      "invalid JSON on line 2",
    );
  });

  test("loads test cases from a JSON file", async () => {
    const filePath = writeJson("test-cases.json", [
      {
        query: "What is the capital of Germany?",
        response: "Berlin.",
        expected_output: "Berlin",
        retrieval_context: ["Germany is in Central Europe."],
      },
    ]);

    const dataset = new EvaluationDataset();
    const testCases = await dataset.addTestCasesFromJSON({
      filePath,
      keys: { input: "query", actualOutput: "response" },
    });

    expect(testCases.length).toBe(1);
    expect(dataset.testCases.length).toBe(1);
    const testCase = dataset.testCases[0] as LLMTestCase;
    expect(testCase).toBeInstanceOf(LLMTestCase);
    expect(testCase.actualOutput).toBe("Berlin.");
    expect(testCase.expectedOutput).toBe("Berlin");
  });

  test("fails when a test case record has no actual output", async () => {
    const filePath = writeJson("incomplete.json", [
      { input: "What is DeepEval?" },
    ]);

    const dataset = new EvaluationDataset();
    await expect(dataset.addTestCasesFromJSON({ filePath })).rejects.toThrow(
      "Required keys 'input' and 'actual_output' are missing",
    );
  });

  test("loads test cases from a CSV file, adding them to the dataset", async () => {
    const filePath = write(
      "test-cases.csv",
      [
        "input,actual_output,expected_output,context,tools_called",
        `"What is the capital of Germany?","Berlin.","Berlin","Geography|Europe","[{""name"": ""search""}]"`,
      ].join("\n"),
    );

    const dataset = new EvaluationDataset();
    const testCases = await dataset.addTestCasesFromCSV({ filePath });

    expect(testCases.length).toBe(1);
    expect(dataset.testCases.length).toBe(1);
    const testCase = testCases[0];
    expect(testCase.input).toBe("What is the capital of Germany?");
    expect(testCase.expectedOutput).toBe("Berlin");
    expect(testCase.context).toEqual(["Geography", "Europe"]);
    expect(testCase.toolsCalled?.[0]).toBeInstanceOf(ToolCall);
    expect(testCase.toolsCalled?.[0].name).toBe("search");
  });

  test("leaves a CSV field unset when the file has no such column", async () => {
    const filePath = write(
      "minimal.csv",
      ["input,actual_output", '"What is DeepEval?","A framework."'].join("\n"),
    );

    const dataset = new EvaluationDataset();
    const [testCase] = await dataset.addTestCasesFromCSV({ filePath });

    // Not `[]`, which a metric's required-parameter check reads as present.
    expect(testCase.context).toBeUndefined();
    expect(testCase.retrievalContext).toBeUndefined();
    expect(testCase.toolsCalled).toBeUndefined();
    expect(testCase.expectedTools).toBeUndefined();
  });

  test("loads goldens from a CSV file", async () => {
    const filePath = write(
      "goldens.csv",
      [
        "input,expected_output,context,retrieval_context,expected_tools,name,custom_column_key_values",
        '"What is DeepEval?","A framework.","Docs|OSS","deepeval_source=docs.md,deepeval_context=DeepEval evaluates LLMs.","[{""name"": ""search""}]","intro","{""owner"": ""platform""}"',
      ].join("\n"),
    );

    const dataset = new EvaluationDataset();
    const [golden] = (await dataset.addGoldensFromCSV({
      filePath,
    })) as Golden[];

    expect(dataset.goldens.length).toBe(1);
    expect(golden.input).toBe("What is DeepEval?");
    expect(golden.expectedOutput).toBe("A framework.");
    expect(golden.context).toEqual(["Docs", "OSS"]);
    const retrieved = golden.retrievalContext?.[0] as RetrievedContextData;
    expect(retrieved).toBeInstanceOf(RetrievedContextData);
    expect(retrieved.source).toBe("docs.md");
    expect(golden.expectedTools?.[0]).toBeInstanceOf(ToolCall);
    expect(golden.name).toBe("intro");
    expect(golden.customColumnKeyValues).toEqual({ owner: "platform" });
    // Absent columns stay unset rather than becoming empty.
    expect(golden.actualOutput).toBeUndefined();
    expect(golden.toolsCalled).toBeUndefined();
  });

  test.each(["json", "jsonl", "csv"] as const)(
    "round trips single-turn goldens through %s",
    async (fileType) => {
      const dataset = new EvaluationDataset({
        goldens: [
          new Golden({
            input: "What is DeepEval?",
            actualOutput: "A framework.",
            expectedOutput: "An evaluation framework.",
            context: ["Docs", "OSS"],
            retrievalContext: [
              new RetrievedContextData({
                source: "docs.md",
                context: "DeepEval evaluates LLMs.",
              }),
              "A plain chunk.",
            ],
            toolsCalled: [
              new ToolCall({ name: "search", inputParameters: { q: "docs" } }),
            ],
            name: "intro",
            comments: "seeded",
            sourceFile: "docs.md",
            additionalMetadata: { topic: "overview" },
            customColumnKeyValues: { owner: "platform" },
          }),
        ],
      });

      const filePath = await dataset.saveAs({
        fileType,
        directory: path.join(tempDir, "saved"),
        fileName: `single-turn-${fileType}`,
      });
      expect(fs.existsSync(filePath)).toBe(true);

      const reloaded = new EvaluationDataset();
      if (fileType === "json") await reloaded.addGoldensFromJSON({ filePath });
      else if (fileType === "jsonl")
        await reloaded.addGoldensFromJSONL({ filePath });
      else await reloaded.addGoldensFromCSV({ filePath });

      const golden = reloaded.goldens[0] as Golden;
      expect(golden.input).toBe("What is DeepEval?");
      expect(golden.actualOutput).toBe("A framework.");
      expect(golden.expectedOutput).toBe("An evaluation framework.");
      expect(golden.context).toEqual(["Docs", "OSS"]);
      expect(golden.name).toBe("intro");
      expect(golden.comments).toBe("seeded");
      expect(golden.sourceFile).toBe("docs.md");
      expect(golden.additionalMetadata).toEqual({ topic: "overview" });
      expect(golden.customColumnKeyValues).toEqual({ owner: "platform" });
      expect(golden.toolsCalled?.[0]).toBeInstanceOf(ToolCall);
      expect(golden.toolsCalled?.[0].inputParameters).toEqual({ q: "docs" });

      // The source of a structured chunk survives the trip to disk.
      const retrieved = golden.retrievalContext?.[0] as RetrievedContextData;
      expect(retrieved).toBeInstanceOf(RetrievedContextData);
      expect(retrieved.source).toBe("docs.md");
      expect(retrieved.context).toBe("DeepEval evaluates LLMs.");
      expect(golden.retrievalContext?.[1]).toBe("A plain chunk.");
    },
  );

  test.each(["json", "jsonl", "csv"] as const)(
    "round trips multi-turn goldens through %s",
    async (fileType) => {
      const dataset = new EvaluationDataset({
        goldens: [
          new ConversationalGolden({
            scenario: "A user asks about pricing.",
            turns: [
              new Turn({ role: "user", content: "How much is it?" }),
              new Turn({
                role: "assistant",
                content: "It is free.",
                toolsCalled: [new ToolCall({ name: "lookup_price" })],
              }),
            ],
            expectedOutcome: "The user learns the price.",
            userDescription: "A prospective customer.",
            context: ["Pricing page"],
            name: "pricing",
            additionalMetadata: { tier: "free" },
          }),
        ],
      });

      const filePath = await dataset.saveAs({
        fileType,
        directory: path.join(tempDir, "saved"),
        fileName: `multi-turn-${fileType}`,
      });

      const reloaded = new EvaluationDataset();
      if (fileType === "json") await reloaded.addGoldensFromJSON({ filePath });
      else if (fileType === "jsonl")
        await reloaded.addGoldensFromJSONL({ filePath });
      else await reloaded.addGoldensFromCSV({ filePath });

      const golden = reloaded.goldens[0] as ConversationalGolden;
      expect(golden).toBeInstanceOf(ConversationalGolden);
      expect(golden.scenario).toBe("A user asks about pricing.");
      expect(golden.expectedOutcome).toBe("The user learns the price.");
      expect(golden.userDescription).toBe("A prospective customer.");
      expect(golden.context).toEqual(["Pricing page"]);
      expect(golden.name).toBe("pricing");
      expect(golden.additionalMetadata).toEqual({ tier: "free" });
      expect(golden.turns?.length).toBe(2);
      expect(golden.turns?.[1].content).toBe("It is free.");
      expect(golden.turns?.[1].toolsCalled?.[0]).toBeInstanceOf(ToolCall);
    },
  );

  test("saves test cases as goldens only when asked", async () => {
    const dataset = new EvaluationDataset();
    dataset.addTestCase(
      new LLMTestCase({ input: "Ping?", actualOutput: "Pong." }),
    );

    const directory = path.join(tempDir, "saved");
    await expect(
      dataset.saveAs({ fileType: "json", directory, fileName: "empty" }),
    ).rejects.toThrow("No goldens found");

    const filePath = await dataset.saveAs({
      fileType: "json",
      directory,
      fileName: "with-test-cases",
      includeTestCases: true,
    });

    const reloaded = new EvaluationDataset();
    await reloaded.addGoldensFromJSON({ filePath });
    expect((reloaded.goldens[0] as Golden).actualOutput).toBe("Pong.");
  });

  test("rejects an unsupported file type", async () => {
    const dataset = new EvaluationDataset({
      goldens: [new Golden({ input: "Ping?" })],
    });
    await expect(
      dataset.saveAs({
        fileType: "txt" as never,
        directory: path.join(tempDir, "saved"),
      }),
    ).rejects.toThrow("Invalid file type");
  });

  test("surfaces a readable error for a missing or invalid file", async () => {
    const dataset = new EvaluationDataset();
    await expect(
      dataset.addGoldensFromJSON({ filePath: path.join(tempDir, "nope.json") }),
    ).rejects.toThrow("was not found");

    const filePath = write("not-json.json", "{ nope");
    await expect(dataset.addGoldensFromJSON({ filePath })).rejects.toThrow(
      "is not a valid JSON file",
    );
  });
});
