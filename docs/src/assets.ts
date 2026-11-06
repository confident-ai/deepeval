const BUCKETS = {
  deepevalDocs: "https://deepeval-docs.s3.amazonaws.com",
  deepevalDocsRegion: "https://deepeval-docs.s3.us-east-1.amazonaws.com",
  confidentDocs: "https://confident-docs.s3.us-east-1.amazonaws.com",
  confidentBucket: "https://confident-bucket.s3.us-east-1.amazonaws.com",
};

export const ASSETS = {
  // ---- Platform Videos (Confident AI) ----
  tracingTraces: `${BUCKETS.confidentDocs}/llm-tracing:traces.mp4`,
  tracingSpans: `${BUCKETS.confidentDocs}/llm-tracing:spans.mp4`,
  tracingThreads: `${BUCKETS.confidentDocs}/llm-tracing:threads.mp4`,
  evaluationOverview: `${BUCKETS.confidentDocs}/evaluation:overview.mp4`,
  evaluationSingleTurnE2eReport: `${BUCKETS.confidentDocs}/evaluation:single-turn-e2e-report.mp4`,
  evaluationSingleTurnE2eReportTracing: `${BUCKETS.confidentDocs}/evaluation:single-turn-e2e-report-tracing.mp4`,
  evaluationMultiTurnE2eReport: `${BUCKETS.confidentDocs}/evaluation:multi-turn-e2e-report.mp4`,
  evaluationParameterInsights: `${BUCKETS.confidentDocs}/evaluation:parameter-insights.mp4`,
  metricsCreateCollection: `${BUCKETS.confidentDocs}/metrics:create-collection-4k.mp4`,
  datasetsCreate: `${BUCKETS.confidentDocs}/datasets:create-4k.mp4`,
  // ---- Getting Started Videos ----
  conversationTestReport: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aconversation-test-report.mp4`,
  gettingStartedRag: `${BUCKETS.deepevalDocsRegion}/getting-started%3Arag.mp4`,
  gettingStartedRagEvalsComponent: `${BUCKETS.deepevalDocsRegion}/getting-started%3Arag-evals%3Acomponent.mp4`,
  gettingStartedRagEvalsConversation: `${BUCKETS.deepevalDocsRegion}/getting-started%3Arag-evals%3Aconversation.mp4`,
  gettingStartedAgentEvalsEndToEnd: `${BUCKETS.deepevalDocsRegion}/getting-started:ai-agent-evals:end-to-end.mp4`,
  gettingStartedAgentEvalsEndToEndEncoded: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aai-agent-evals%3Aend-to-end.mp4`,
  gettingStartedAgentEvalsLanggraph: `${BUCKETS.deepevalDocsRegion}/getting-started:ai-agent-evals:langgraph.mp4`,
  gettingStartedAgentEvalsLangchain: `${BUCKETS.deepevalDocsRegion}/getting-started:ai-agent-evals:langchain.mp4`,
  gettingStartedAgentEvalsCrewAi: `${BUCKETS.deepevalDocsRegion}/getting-started:ai-agent-evals:crew-ai.mp4`,
  gettingStartedChatbotEvalsMultiturnDataset: `${BUCKETS.deepevalDocsRegion}/getting-started%3Achatbot-evals%3Amultiturn-dataset.mp4`,
  gettingStartedMcpSingleTurn: `${BUCKETS.deepevalDocsRegion}/docs:getting-started-mcp-single-turn.mp4`,
  gettingStartedMcpMultiTurn: `${BUCKETS.deepevalDocsRegion}/docs:getting-started-mcp-multi-turn.mp4`,
  // ---- Arena Evals Videos ----
  arenaEvalsExperiment: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aarena-evals%3Aexperiment.mp4`,
  arenaEvalsQuickRun: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aarena-evals%3Aquick-run.mp4`,
  arenaEvalsRunExperiment: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aarena-evals%3Arun-experiment.mp4`,
  arenaEvalsTracedComparisons: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aarena-evals%3Atraced-comparisons.mp4`,
  arenaEvalsMetricComparisons: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aarena-evals%3Ametric-comparisons.mp4`,
  arenaEvalsLogPrompts: `${BUCKETS.deepevalDocsRegion}/getting-started%3Aarena-evals%3Alog-prompts.mp4`,
  // ---- Evaluation Videos ----
  testCaseToolsCalled: `${BUCKETS.deepevalDocsRegion}/test-case-tools-called.mp4`,
  // ---- Tutorial: Summarization Agent ----
  tutorialSummarizationHyperparameters: `${BUCKETS.deepevalDocsRegion}/tutorial-legal-document-summarizer-hyperparameters.mp4`,
  // ---- Integration: Framework Videos ----
  integrationOpenai: `${BUCKETS.deepevalDocsRegion}/integrations:frameworks:openai.mp4`,
  integrationPydantic: `${BUCKETS.confidentBucket}/end-to-end%3Apydantic-1080.mp4`,
  integrationLlamaIndex: `${BUCKETS.confidentBucket}/end-to-end%3Allama-index-1080.mp4`,
  integrationLanggraph: `${BUCKETS.confidentBucket}/end-to-end%3Alanggraph.mp4`,
  integrationLangchain: `${BUCKETS.confidentBucket}/end-to-end%3Alangchain.mp4`,
  integrationCrewai: `${BUCKETS.confidentDocs}/end-to-end%3Acrewai-4k-no-zoom.mp4`,
};
