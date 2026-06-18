import axios from "axios";
import { wait } from "../utils";
import { createInterface } from "readline";

enum Regions {
  EU = "EU",
  US = "US",
  AU = "AU",
}

const _LOCAL_API_BASE_URL = "http://localhost:3001";
const _LOCAL_DEEPEVAL_BASE_URL = "http://0.0.0.0:8000";
const PROD_DEEPEVAL_BASE_URL = "https://deepeval.confident-ai.com";
const CONFIDENT_BASE_URL = "https://api.confident-ai.com";
const CONFIDENT_BASE_URL_EU = "https://eu.api.confident-ai.com";
const CONFIDENT_BASE_URL_AU = "https://au.api.confident-ai.com";

const region = process.env.CONFIDENT_REGION || Regions.US;

export const DEEPEVAL_BASE_URL = PROD_DEEPEVAL_BASE_URL;
export const API_BASE_URL =
  region === Regions.EU
    ? CONFIDENT_BASE_URL_EU
    : region === Regions.AU
      ? CONFIDENT_BASE_URL_AU
      : CONFIDENT_BASE_URL;

const inferBaseUrlFromApiKey = (apiKey: string): string => {
  if (apiKey.startsWith("confident_eu_")) {
    return CONFIDENT_BASE_URL_EU;
  } else if (apiKey.startsWith("confident_au_")) {
    return CONFIDENT_BASE_URL_AU;
  }
  return CONFIDENT_BASE_URL;
};

const RETRYABLE_ERROR_CODES = [
  "ECONNRESET",
  "ETIMEDOUT",
  "ECONNREFUSED",
  "ENOTFOUND",
  "ENETUNREACH",
  "ESOCKETTIMEDOUT",
  "CERT_HAS_EXPIRED",
];

function logRetryError(error: any, attempt: number): void {
  console.error(
    `Confident AI Error: ${error}. Retrying: ${attempt} time(s)...`,
  );
}

export enum HttpMethods {
  GET = "GET",
  POST = "POST",
  DELETE = "DELETE",
  PUT = "PUT",
}

export enum Endpoints {
  DATASET_ALIAS_ENDPOINT = "/v1/datasets/:alias",
  DATASET_ALIAS_QUEUE_ENDPOINT = "/v1/datasets/:alias/queue",
  DATASET_ALIAS_VERSIONS_ENDPOINT = "/v1/datasets/:alias/versions",
  TEST_RUN_ENDPOINT = "/v1/test-run",
  TRACING_ENDPOINT = "/v1/tracing",
  TRACES_ENDPOINT = "/v1/traces",
  EVENT_ENDPOINT = "/v1/event",
  FEEDBACK_ENDPOINT = "/v1/feedback",
  PROMPTS_VERSION_ID_ENDPOINT = "/v1/prompts/:alias/versions/:version",
  PROMPTS_LABEL_ENDPOINT = "/v1/prompts/:alias/labels/:label",
  PROMPTS_COMMITS_ENDPOINT = "/v1/prompts/:alias/commits",
  PROMPTS_COMMIT_HASH_ENDPOINT = "/v1/prompts/:alias/commits/:hash",
  PROMPTS_VERSIONS_ENDPOINT = "/v1/prompts/:alias/versions",
  PROMPTS_BRANCHES_ENDPOINT = "/v1/prompts/:alias/branches",
  PROMPTS_BRANCH_ENDPOINT = "/v1/prompts/:alias/branches/:name",
  PROMPTS_ENDPOINT = "/v1/prompts",
  RECOMMEND_ENDPOINT = "/v1/recommend-metrics",
  EVALUATE_ENDPOINT = "/v1/evaluate",
  EVALUATE_THREAD_ENDPOINT = "/v1/evaluate/threads/:threadId",
  EVALUATE_TRACE_ENDPOINT = "/v1/evaluate/traces/:traceUuid",
  EVALUATE_SPAN_ENDPOINT = "/v1/evaluate/spans/:spanUuid",
  GUARD_ENDPOINT = "/guard",
  GUARDRAILS_ENDPOINT = "/guardrails",
  BASELINE_ATTACKS_ENDPOINT = "/generate-baseline-attacks",
  SIMULATE_ENDPOINT = "/v1/simulate",
  ANNOTATION_ENDPOINT = "/v1/annotations",
  EXPERIMENT_ENDPOINT = "/v1/experiment",
  GOVERNANCE_ASSESS_ENDPOINT = "/v1/governance/assess",
}

interface RetryOptions {
  maxAttempts: number;
  initialDelay: number;
  maxDelay: number;
  factor: number;
  jitter: boolean;
}

const defaultRetryOptions: RetryOptions = {
  maxAttempts: 5,
  initialDelay: 1000,
  maxDelay: 10000,
  factor: 2,
  jitter: true,
};

export class Api {
  private apiKey: string;
  private headers: Record<string, string>;
  private baseApiUrl: string;

  constructor(apiKey?: string, baseUrl?: string) {
    if (!apiKey) {
      apiKey = process.env.CONFIDENT_API_KEY;
    }

    if (!apiKey) {
      throw new Error("Please provide a valid Confident AI API Key.");
    }

    // if region is set or url is provided, respect that
    if (!process.env.CONFIDENT_REGION && !baseUrl) {
      this.baseApiUrl = inferBaseUrlFromApiKey(apiKey);
    } else {
      this.baseApiUrl = baseUrl || API_BASE_URL;
    }

    this.apiKey = apiKey;
    this.headers = {
      "Content-Type": "application/json",
      CONFIDENT_API_KEY: apiKey,
    };
  }

  private static async httpRequest(
    method: string,
    url: string,
    headers?: Record<string, string>,
    data?: any,
    params?: Record<string, any>,
    options: RetryOptions = defaultRetryOptions,
  ): Promise<any> {
    let attempt = 0;
    let delay = options.initialDelay;

    while (attempt < options.maxAttempts) {
      try {
        const response = await axios({
          method,
          url,
          headers,
          data,
          params,
        });
        return response;
      } catch (error: any) {
        attempt++;

        // Check if error is retryable
        const isRetryable =
          error.code &&
          RETRYABLE_ERROR_CODES.some((code) => error.code.includes(code));

        if (!isRetryable || attempt >= options.maxAttempts) {
          throw error;
        }

        logRetryError(error, attempt);

        // Calculate delay with exponential backoff and jitter
        if (options.jitter) {
          const jitterFactor = Math.random() + 0.5; // Random between 0.5 and 1.5
          delay = Math.min(
            delay * options.factor * jitterFactor,
            options.maxDelay,
          );
        } else {
          delay = Math.min(delay * options.factor, options.maxDelay);
        }

        await wait(delay);
      }
    }

    throw new Error(`Request failed after ${options.maxAttempts} attempts`);
  }

  public async sendRequest(
    method: HttpMethods,
    endpoint: Endpoints | string,
    body?: any,
    params?: Record<string, any>,
    endpointString?: string,
    urlParams?: Record<string, string>,
  ): Promise<any> {
    let endpointPath = endpointString || endpoint;

    if (urlParams) {
      for (const [key, value] of Object.entries(urlParams)) {
        endpointPath = endpointPath.replace(`:${key}`, value);
      }
    }

    const url = `${this.baseApiUrl}${endpointPath}`;
    try {
      const res = await Api.httpRequest(
        method,
        url,
        this.headers,
        body,
        params,
      );

      if (res.status === 200) {
        return res.data;
      } else if (res.status === 409 && body) {
        const message = res.data?.message || "Conflict occurred.";

        // In Node.js environment
        if (typeof process !== "undefined" && process.stdin && process.stdout) {
          const readline = createInterface({
            input: process.stdin,
            output: process.stdout,
          });

          return new Promise((resolve) => {
            readline.question(
              `${message} Would you like to overwrite it? [y/N] or change the alias [c]: `,
              (answer: string) => {
                readline.close();
                const userInput = answer.trim().toLowerCase();

                if (userInput === "y") {
                  body.overwrite = true;
                  resolve(this.sendRequest(method, endpoint, body));
                } else if (userInput === "c") {
                  readline.question(
                    "Enter a new alias: ",
                    (newAlias: string) => {
                      readline.close();
                      body.alias = newAlias.trim();
                      resolve(this.sendRequest(method, endpoint, body));
                    },
                  );
                } else {
                  console.log("Aborted.");
                  resolve(null);
                }
              },
            );
          });
        } else {
          console.error(
            "Conflict occurred. Please implement appropriate UI handling for this environment.",
          );
          return null;
        }
      } else {
        throw new Error(res.data?.error || res.statusText);
      }
    } catch (error: any) {
      throw new Error(error.response?.data?.error || error.message);
    }
  }
}
