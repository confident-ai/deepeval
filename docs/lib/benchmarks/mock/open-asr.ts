import type { Benchmark } from "../types";

export const openAsr: Benchmark = {
  slug: "open-asr",
  name: "Open ASR",
  category: "speech-to-text",
  tagline: "How accurately does a model transcribe real speech?",
  summary:
    "Word error rate across public English test sets that span read speech, meetings, earnings calls and talks. Accuracy is reported next to speed, since voice agents need both.",
  version: "v2.3",
  updatedAt: "2026-11-05",
  grading: {
    method: "programmatic",
    summary:
      "Transcripts are normalized, then scored by word error rate against human reference transcripts. No model takes part in grading.",
    runsPerCase: 1,
  },
  dataset: {
    size: 39200,
    unit: "audio clips",
    source: "LibriSpeech, AMI, Earnings22, GigaSpeech, SPGISpeech, TED-LIUM and VoxPopuli test splits",
    license: "CC BY 4.0",
    host: "Hugging Face",
    url: "https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
    publicShare: 1,
  },
  metrics: [
    {
      id: "wer",
      label: "Avg. WER",
      pickLabel: "Most accurate",
      unit: "percent",
      higherIsBetter: false,
      description: "Mean word error rate across all test sets.",
      decimals: 2,
    },
    {
      id: "noisyWer",
      label: "Noisy WER",
      pickLabel: "Best in noise",
      unit: "percent",
      higherIsBetter: false,
      description: "Word error rate on the meeting and call subsets.",
      decimals: 2,
    },
    {
      id: "rtfx",
      label: "Speed",
      pickLabel: "Fastest",
      unit: "multiplier",
      higherIsBetter: true,
      description: "Seconds of audio transcribed per second of compute.",
      decimals: 0,
    },
  ],
  primaryMetricId: "wer",
  tradeoffMetricId: "rtfx",
  segmentLabel: "Test set",
  segments: [
    { id: "read", label: "Read speech", caseCount: 5559 },
    { id: "meetings", label: "Meetings", caseCount: 12643 },
    { id: "earnings", label: "Earnings calls", caseCount: 2741 },
    { id: "talks", label: "Talks", caseCount: 18257 },
  ],
  skills: [
    {
      id: "accents",
      label: "Accents",
      description: "Non-native and regional English accents.",
    },
    {
      id: "noise",
      label: "Background noise",
      description: "Speech over room noise, music and line artefacts.",
    },
    {
      id: "entities",
      label: "Numbers and entities",
      description: "Figures, tickers, names and product terms.",
    },
    {
      id: "long-form",
      label: "Long-form audio",
      description: "Staying accurate across recordings longer than ten minutes.",
    },
    {
      id: "overlap",
      label: "Speaker overlap",
      description: "Two or more people speaking at once.",
    },
  ],
  entries: [
    {
      model: "Rumble-2",
      provider: "Reson8",
      scores: { wer: 4.86, noisyWer: 8.92, rtfx: 164 },
      segmentScores: {
        wer: { read: 1.74, meetings: 9.57, earnings: 7.90, talks: 3.30 },
        noisyWer: { read: 4.84, meetings: 13.00, earnings: 11.81, talks: 7.29 },
        rtfx: { read: 315, meetings: 111, earnings: 124, talks: 208 },
      },
      skillScores: {
        accents: 88,
        noise: 86,
        entities: 87,
        "long-form": 88,
        overlap: 78,
      },
      confidenceInterval: 0.21,
      evaluatedAt: "2026-11-02",
    },
    {
      model: "Nemotron 3.5 Transcribe",
      provider: "NVIDIA",
      scores: { wer: 4.97, noisyWer: 9.14, rtfx: 2310 },
      segmentScores: {
        wer: { read: 1.78, meetings: 8.48, earnings: 8.28, talks: 3.66 },
        noisyWer: { read: 4.79, meetings: 12.13, earnings: 11.94, talks: 7.86 },
        rtfx: { read: 4346, meetings: 1739, earnings: 1635, talks: 2806 },
      },
      skillScores: {
        accents: 79,
        noise: 80,
        entities: 79,
        "long-form": 88,
        overlap: 75,
      },
      confidenceInterval: 0.22,
      openWeights: true,
      evaluatedAt: "2026-11-02",
    },
    {
      model: "Ink 2",
      provider: "Cartesia",
      scores: { wer: 5.08, noisyWer: 9.31, rtfx: 212 },
      segmentScores: {
        wer: { read: 1.84, meetings: 9.02, earnings: 7.61, talks: 3.51 },
        noisyWer: { read: 5.05, meetings: 13.38, earnings: 11.69, talks: 7.76 },
        rtfx: { read: 377, meetings: 151, earnings: 170, talks: 273 },
      },
      skillScores: {
        accents: 75,
        noise: 68,
        entities: 74,
        "long-form": 75,
        overlap: 71,
      },
      confidenceInterval: 0.22,
      evaluatedAt: "2026-10-29",
    },
    {
      model: "Scribe v3",
      provider: "ElevenLabs",
      scores: { wer: 5.22, noisyWer: 9.06, rtfx: 98 },
      segmentScores: {
        wer: { read: 1.94, meetings: 9.59, earnings: 8.20, talks: 4.00 },
        noisyWer: { read: 5.10, meetings: 13.26, earnings: 12.16, talks: 7.98 },
        rtfx: { read: 175, meetings: 69, earnings: 77, talks: 118 },
      },
      skillScores: {
        accents: 67,
        noise: 69,
        entities: 69,
        "long-form": 74,
        overlap: 61,
      },
      confidenceInterval: 0.23,
      evaluatedAt: "2026-10-29",
    },
    {
      model: "Universal 3.6 Pro",
      provider: "AssemblyAI",
      scores: { wer: 5.41, noisyWer: 9.77, rtfx: 143 },
      segmentScores: {
        wer: { read: 1.90, meetings: 9.91, earnings: 7.96, talks: 3.59 },
        noisyWer: { read: 5.18, meetings: 14.37, earnings: 12.68, talks: 7.68 },
        rtfx: { read: 271, meetings: 99, earnings: 114, talks: 184 },
      },
      skillScores: {
        accents: 66,
        noise: 57,
        entities: 64,
        "long-form": 70,
        overlap: 56,
      },
      confidenceInterval: 0.23,
      evaluatedAt: "2026-10-18",
    },
    {
      model: "Nova-3",
      provider: "Deepgram",
      scores: { wer: 5.93, noisyWer: 10.84, rtfx: 418 },
      segmentScores: {
        wer: { read: 2.19, meetings: 11.00, earnings: 9.27, talks: 4.46 },
        noisyWer: { read: 5.76, meetings: 15.88, earnings: 14.73, talks: 9.41 },
        rtfx: { read: 774, meetings: 286, earnings: 326, talks: 499 },
      },
      skillScores: {
        accents: 53,
        noise: 55,
        entities: 56,
        "long-form": 60,
        overlap: 53,
      },
      confidenceInterval: 0.25,
      evaluatedAt: "2026-10-18",
    },
    {
      model: "gpt-4o-transcribe",
      provider: "OpenAI",
      scores: { wer: 6.12, noisyWer: 11.02, rtfx: 86 },
      segmentScores: {
        wer: { read: 2.33, meetings: 11.43, earnings: 9.88, talks: 4.70 },
        noisyWer: { read: 6.14, meetings: 16.46, earnings: 14.20, talks: 9.59 },
        rtfx: { read: 148, meetings: 60, earnings: 64, talks: 99 },
      },
      skillScores: {
        accents: 51,
        noise: 50,
        entities: 53,
        "long-form": 57,
        overlap: 47,
      },
      confidenceInterval: 0.25,
      evaluatedAt: "2026-10-07",
    },
    {
      model: "Whisper Large v3",
      provider: "OpenAI",
      scores: { wer: 7.44, noisyWer: 13.65, rtfx: 146 },
      segmentScores: {
        wer: { read: 2.81, meetings: 13.84, earnings: 11.49, talks: 5.74 },
        noisyWer: { read: 7.73, meetings: 19.81, earnings: 17.88, talks: 12.08 },
        rtfx: { read: 257, meetings: 97, earnings: 111, talks: 173 },
      },
      skillScores: {
        accents: 48,
        noise: 47,
        entities: 52,
        "long-form": 49,
        overlap: 40,
      },
      confidenceInterval: 0.28,
      openWeights: true,
      evaluatedAt: "2026-10-07",
    },
  ],
  methodology: [
    {
      title: "Public test sets only",
      body: "Every clip comes from a published test split, so anyone can download the audio and reproduce a row.",
    },
    {
      title: "One normalizer",
      body: "Model output and references pass through the same text normalizer: casing, punctuation, numbers, spelling variants and filler words.",
    },
    {
      title: "Averaged across sets",
      body: "The headline number is the mean word error rate over the test sets, so no single domain dominates.",
    },
    {
      title: "Speed on fixed hardware",
      body: "Open-weight models are timed on one GPU type. API models are timed end to end from the same region.",
    },
  ],
  runSnippet: `from deepeval.benchmarks import OpenASR

benchmark = OpenASR()
benchmark.evaluate(model=your_stt_model)

print(benchmark.overall_score)`,
  faqs: [
    {
      question: "Lower is better here?",
      answer:
        "Yes. Word error rate counts substitutions, deletions and insertions, so the best model has the lowest number.",
    },
    {
      question: "Could models have trained on these test sets?",
      answer:
        "It is possible, since the sets are public. Treat small gaps at the top of the table with care, and check the noisy subset, which is harder to overfit.",
    },
    {
      question: "Is streaming accuracy measured?",
      answer:
        "This table scores final transcripts. Streaming accuracy and partial-result latency are planned as a separate track.",
    },
  ],
};
