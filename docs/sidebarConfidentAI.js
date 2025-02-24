module.exports = {
    confidentAI: [
      "confident-ai-introduction",
      {
        type: "category",
        label: "Datasets",
        items: [
          "confident-ai-evaluation-dataset-management",
          "confident-ai-evaluation-dataset-evaluation",
          "confident-ai-evaluation-dataset-improvement",
        ],
        collapsed: false,
      },
      {
        type: "category",
        label: "LLM Evaluation",
        items: [
          "confident-ai-testing-n-evaluation-test-run",
          "confident-ai-testing-n-evaluation-unit-testing",
          "confident-ai-testing-n-evaluation-metric-collections",
        ],
        collapsed: false,
      },
      {
        type: "category",
        label: "LLM Observability",
        items: [
          "confident-ai-llm-monitoring",
          "confident-ai-llm-monitoring-evaluations",
          {
            type: "category",
            label: "LLM Tracing",
            items: [
              "confident-ai-tracing",
              "confident-ai-tracing-llama-index",
              "confident-ai-tracing-langchain",
            ],
            collapsed: true,
          },
        ],
        collapsed: false,
      },
      {
        type: "category",
        label: "Human-in-the-Loop",
        items: [
          "confident-ai-human-feedback",
          "confident-ai-human-feedback-adding",
          "confident-ai-human-feedback-platform",
        ],
        collapsed: false,
      },
      {
        type: "category",
        label: "Hyperparameters",
        items: ["confident-ai-hyperparameters-prompt-versioning"],
        collapsed: true,
      },
      {
        type: "category",
        label: "Guardrails",
        items: [
          "confident-ai-guardrails",
          {
            type: "category",
            label: "Guards",
            items: [
              "confident-ai-guardrails-prompt-injection",
              "confident-ai-guardrails-privacy",
              "confident-ai-guardrails-jailbreaking",
              "confident-ai-guardrails-topical",
              "confident-ai-guardrails-cybersecurity",
              "confident-ai-guardrails-hallucination",
              "confident-ai-guardrails-graphic-content",
              "confident-ai-guardrails-illegal",
              "confident-ai-guardrails-toxicity",
              "confident-ai-guardrails-syntax",
              "confident-ai-guardrails-modernization",
            ],
            collapsed: false,
          },
  
        ],
        collapsed: true,
      },
      {
        type: "category",
        label: "Advanced",
        items: [
          "confident-ai-advanced-evaluation-model",
          "confident-ai-advanced-llm-connection"
        ],
        collapsed: true,
      },
    ],
  };
  