import type { Benchmark } from "../types";
import { bigBenchAudio } from "./big-bench-audio";
import { humanitysLastExam } from "./humanitys-last-exam";
import { openAsr } from "./open-asr";
import { sweBenchVerified } from "./swe-bench-verified";
import { tauVoice } from "./tau-voice";

// Placeholder benchmarks for building the pages, in display order.
export const MOCK_BENCHMARKS: Benchmark[] = [
  sweBenchVerified,
  tauVoice,
  bigBenchAudio,
  openAsr,
  humanitysLastExam,
];
