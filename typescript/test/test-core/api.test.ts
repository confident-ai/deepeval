import axios from "axios";
import { Api, Endpoints, HttpMethods } from "../../src/confident/api";

jest.mock("axios");

const mockedAxios = axios as unknown as jest.Mock;

describe("Api project targeting", () => {
  beforeEach(() => {
    mockedAxios.mockReset();
    mockedAxios.mockResolvedValue({ status: 200, data: {} });
  });

  const lastRequestHeaders = () => mockedAxios.mock.calls[0][0].headers;

  test("adds CONFIDENT_PROJECT_ID header when a projectId is provided", async () => {
    const api = new Api("confident_us_global_key", "https://api.example.com");

    await api.sendRequest(
      HttpMethods.GET,
      Endpoints.DATASET_ALIAS_ENDPOINT,
      undefined,
      undefined,
      undefined,
      { alias: "my-dataset" },
      "proj_123",
    );

    expect(lastRequestHeaders().CONFIDENT_PROJECT_ID).toBe("proj_123");
    expect(lastRequestHeaders().CONFIDENT_API_KEY).toBe(
      "confident_us_global_key",
    );
  });

  test("omits CONFIDENT_PROJECT_ID header when no projectId is provided", async () => {
    const api = new Api("confident_project_key", "https://api.example.com");

    await api.sendRequest(HttpMethods.GET, Endpoints.PROMPTS_ENDPOINT);

    expect(lastRequestHeaders().CONFIDENT_PROJECT_ID).toBeUndefined();
    expect(lastRequestHeaders().CONFIDENT_API_KEY).toBe(
      "confident_project_key",
    );
  });
});
