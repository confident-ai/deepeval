import axios from "axios";
import { postExperiment } from "@/evaluate/confident";
import { ArenaTestCase, Contestant, LLMTestCase } from "@/test-case";
import { type ArenaCaseResult } from "@/evaluate/types";
import { Prompt } from "@/prompt";

jest.mock("axios");

const mockedAxios = axios as unknown as jest.Mock;

const contestant = (
  name: string,
  actualOutput: string,
  hyperparameters?: Contestant["hyperparameters"],
) =>
  new Contestant({
    name,
    hyperparameters,
    testCase: new LLMTestCase({ input: "Capital of France?", actualOutput }),
  });

const result = (testCase: ArenaTestCase, winner: string): ArenaCaseResult => ({
  testCase,
  index: 0,
  winner,
  runDuration: 0.1,
});

/** The `{ testRuns, name }` body the experiment endpoint was called with. */
const postedTestRuns = () => mockedAxios.mock.calls[0][0].data.testRuns;

const runFor = (identifier: string) =>
  postedTestRuns().find(
    (r: { identifier: string }) => r.identifier === identifier,
  );

describe("postExperiment hyperparameters", () => {
  beforeEach(() => {
    process.env.CONFIDENT_API_KEY = "confident_us_test_key";
    mockedAxios.mockReset();
    mockedAxios.mockResolvedValue({ status: 200, data: { link: null } });
  });

  afterEach(() => {
    delete process.env.CONFIDENT_API_KEY;
  });

  test("sends each contestant's hyperparameters on its own test run", async () => {
    const testCase = new ArenaTestCase({
      contestants: [
        contestant("Version 1", "Paris", { model: "gpt-3.5-turbo" }),
        contestant("Version 2", "Paris is the capital.", { model: "gpt-4o" }),
      ],
    });

    await postExperiment([result(testCase, "Version 2")], "Friendly", "run");

    expect(runFor("Version 1").hyperparameters).toEqual({
      model: "gpt-3.5-turbo",
    });
    expect(runFor("Version 2").hyperparameters).toEqual({ model: "gpt-4o" });
  });

  test("normalizes non-string values the way logHyperparameters does", async () => {
    const testCase = new ArenaTestCase({
      contestants: [
        contestant("Version 1", "Paris", { temperature: 0, stream: false }),
      ],
    });

    await postExperiment([result(testCase, "Version 1")], "Friendly", "run");

    expect(runFor("Version 1").hyperparameters).toEqual({
      temperature: "0",
      stream: "false",
    });
  });

  test("sends a pulled prompt by reference", async () => {
    const prompt = new Prompt({ alias: "First Prompt" });
    // Stand in for a pulled prompt, which carries the version and type the
    // platform references it by.
    prompt.hash = "hash_123";
    (prompt as unknown as { _type: string })._type = "TEXT";

    const testCase = new ArenaTestCase({
      contestants: [contestant("Version 1", "Paris", { prompt })],
    });

    await postExperiment([result(testCase, "Version 1")], "Friendly", "run");

    expect(runFor("Version 1").hyperparameters).toEqual({
      prompt: { id: "hash_123", type: "TEXT" },
    });
  });

  test("omits hyperparameters entirely when no contestant sets any", async () => {
    const testCase = new ArenaTestCase({
      contestants: [
        contestant("Version 1", "Paris"),
        contestant("Version 2", "Paris is the capital."),
      ],
    });

    await postExperiment([result(testCase, "Version 1")], "Friendly", "run");

    expect(runFor("Version 1").hyperparameters).toBeUndefined();
    expect(runFor("Version 2").hyperparameters).toBeUndefined();
  });
});
