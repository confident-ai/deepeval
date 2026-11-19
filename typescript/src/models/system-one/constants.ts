export const DEFAULT_TYPESAFE_MODEL = "jev-latest";

// Jev is priced on input tokens only; output tokens are free.
export const JEV_INPUT_PRICE = 0.042 / 1e6;

// Jev 1.13 context budget (https://docs.typesafe.ai/models): the whole request
// (state plus every question) must fit in 64k tokens, and the state plus the
// single longest question must fit in 32k. Every current Jev alias points at
// 1.13, so one pair of limits covers the registry.
export const JEV_MAX_REQUEST_TOKENS = 64_000;
export const JEV_MAX_STATE_TOKENS = 32_000;

export interface SystemOneModelData {
  inputPrice: number | undefined;
  outputPrice: number;
}

const JEV_MODEL_DATA: SystemOneModelData = {
  inputPrice: JEV_INPUT_PRICE,
  outputPrice: 0,
};

export const TYPESAFE_MODELS_DATA: Record<string, SystemOneModelData> = {
  "jev-1.13.0": JEV_MODEL_DATA,
  "jev-latest": JEV_MODEL_DATA,
  "jev-preview": JEV_MODEL_DATA,
};

/** Unknown model names are unpriced unless a cost override is given. */
export function typesafeModelData(model: string): SystemOneModelData {
  const known = TYPESAFE_MODELS_DATA[model];
  return known ? { ...known } : { inputPrice: undefined, outputPrice: 0 };
}
