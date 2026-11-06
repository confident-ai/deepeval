import type { Adapter } from "@/tracing/otel/adapters/base";
import { openInferenceAdapter } from "@/tracing/otel/adapters/openinference";

export type { Adapter } from "@/tracing/otel/adapters/base";

export const ADAPTERS: readonly Adapter[] = [openInferenceAdapter];
