import { isConfident } from "../utils";
import { Api, Endpoints, HttpMethods } from "./api";
import {
  ConfidentEvaluateRequestData,
  ConfidentEvaluateResponseData,
} from "./types";

export async function evaluate(
  data: ConfidentEvaluateRequestData,
): Promise<ConfidentEvaluateResponseData | undefined> {
  if (!isConfident()) {
    console.error("Confident AI API key not found.");
    return undefined;
  }

  const apiKey = process.env.CONFIDENT_API_KEY!;
  const api = new Api(apiKey);
  const response = await api.sendRequest(
    HttpMethods.POST,
    Endpoints.EVALUATE_ENDPOINT,
    data,
  );
  return response as ConfidentEvaluateResponseData;
}
