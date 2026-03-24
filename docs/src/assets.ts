const BUCKETS = {
  deepevalDocs: "https://deepeval-docs.s3.amazonaws.com",
  deepevalDocsRegion: "https://deepeval-docs.s3.us-east-1.amazonaws.com",
  confidentDocs: "https://confident-docs.s3.us-east-1.amazonaws.com",
  confidentBucket: "https://confident-bucket.s3.us-east-1.amazonaws.com",
};

export const ASSETS = {
  // ---- Shared Concept Diagrams ----
  llmTestCase: `${BUCKETS.deepevalDocs}/docs:llm-test-case.png`,
  conversationalTestCase: `${BUCKETS.deepevalDocs}/docs:conversational-test-case.png`,
  componentLevelEvals: `${BUCKETS.deepevalDocsRegion}/component-level-evals.png`,
  evaluationDataset: `${BUCKETS.deepevalDocsRegion}/docs:evaluation-dataset.png`,
  endToEndLlmEvals: `${BUCKETS.deepevalDocsRegion}/docs:end-to-end-llm-evals.png`,
  llmTrace: `${BUCKETS.deepevalDocs}/docs:llm-trace.png`,
  mcpArchitecture: `${BUCKETS.deepevalDocs}/mcp-architecture.png`,
  evaluationMcpTools: `${BUCKETS.deepevalDocsRegion}/docs:evaluation-mcp-tools.png`,

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

  // ---- Metrics Images ----
  gEvalAlgorithm: `${BUCKETS.deepevalDocs}/metrics:g-eval:algorithm.png`,
  gEvalResults: `${BUCKETS.deepevalDocs}/metrics:g-eval:results.png`,
  dagSummarization: `${BUCKETS.deepevalDocs}/metrics:dag:summarization.png`,
  dagConversational: `${BUCKETS.deepevalDocsRegion}/metrics:dag:conversational-dag.png`,
  dagTurnWindows: `${BUCKETS.deepevalDocsRegion}/metrics:dag:turn-windows.png`,

  // ---- Evaluation Videos ----
  testCaseToolsCalled: `${BUCKETS.deepevalDocsRegion}/test-case-tools-called.mp4`,

  // ---- Synthesizer Assets ----
  generationFiltration: `${BUCKETS.deepevalDocs}/generation-filtration.svg`,
  evolutions: `${BUCKETS.deepevalDocs}/evolutions.svg`,
  synthesizeFromScratch: `${BUCKETS.deepevalDocs}/synthesize-from-scratch.svg`,
  synthesizeFromContexts: `${BUCKETS.deepevalDocs}/synthesize-from-contexts.svg`,
  synthesizeFromDocs: `${BUCKETS.deepevalDocs}/synthesize-from-docs.svg`,
  filteringContext: `${BUCKETS.deepevalDocs}/filtering_context.svg`,
  goldensFromGoldens: `${BUCKETS.deepevalDocsRegion}/goldens_from_goldens.svg`,
  synthesizerOverview: `${BUCKETS.deepevalDocs}/synthesizer.png`,

  // ---- Red Teaming Assets ----
  redTeamingDeepeval: `${BUCKETS.deepevalDocs}/red_teaming_deepeval.svg`,
  redTeamingIteration: `${BUCKETS.deepevalDocs}/red_teaming_iteration.svg`,

  // ---- Tutorial: Setup ----
  tutorialSetup01: `${BUCKETS.deepevalDocs}/tutorial_setup_01.svg`,

  // ---- Tutorial: Summarization Agent ----
  tutorialSummarizationOverview: `${BUCKETS.deepevalDocsRegion}/tutorials:summarization-agent:summarizer-overview.png`,
  tutorialSummarizationDemo1: `${BUCKETS.deepevalDocsRegion}/tutorials:summarization-agent:summarizer-demo-1.png`,
  tutorialSummarizationDemo2: `${BUCKETS.deepevalDocsRegion}/tutorials:summarization-agent:summarizer-demo-2.png`,
  tutorialSummarizationEvalResults: `${BUCKETS.deepevalDocs}/tutorials:summarization-agent:eval-results.png`,
  tutorialSummarizationHyperparameters: `${BUCKETS.deepevalDocsRegion}/tutorial-legal-document-summarizer-hyperparameters.mp4`,

  // ---- Tutorial: RAG QA Agent ----
  tutorialQaAgentOverview: `${BUCKETS.deepevalDocsRegion}/tutorials:qa-agent:qa-agent-overview.png`,
  tutorialQaAgentDemo1: `${BUCKETS.deepevalDocsRegion}/tutorials:qa-agent:qa-agent-demo-1.png`,
  tutorialQaAgentDemo2: `${BUCKETS.deepevalDocsRegion}/tutorials:qa-agent:qa-agnet-demo-2.png`,
  tutorialRagQaAgentEvalResults: `${BUCKETS.deepevalDocs}/tutorials:rag-qa-agent:eval-results.png`,

  // ---- Tutorial: Medical Chatbot ----
  tutorialMedicalChatbotOverview: `${BUCKETS.deepevalDocsRegion}/tutorials:medical-chatbot:chatbot-overview.png`,

  // ---- Integration: Framework Videos ----
  integrationOpenai: `${BUCKETS.deepevalDocsRegion}/integrations:frameworks:openai.mp4`,
  integrationPydantic: `${BUCKETS.confidentBucket}/end-to-end%3Apydantic-1080.mp4`,
  integrationLlamaIndex: `${BUCKETS.confidentBucket}/end-to-end%3Allama-index-1080.mp4`,
  integrationLanggraph: `${BUCKETS.confidentBucket}/end-to-end%3Alanggraph.mp4`,
  integrationLangchain: `${BUCKETS.confidentBucket}/end-to-end%3Alangchain.mp4`,
  integrationCrewai: `${BUCKETS.confidentDocs}/end-to-end%3Acrewai-4k-no-zoom.mp4`,

  // ---- Integration: Vector Database Images ----
  pgvector: `${BUCKETS.deepevalDocsRegion}/pgvector.png`,
};
