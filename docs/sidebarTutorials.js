module.exports = {
  tutorials: [
    "tutorial-introduction",
    "tutorial-setup",
    {
      type: "category",
      label: "RAG QA Agent ",
      items: [
        "qa-agent-introduction",
        "qa-agent-generating-a-synthetic-dataset",
        "qa-agent-defining-an-evaluation-criteria",
        "qa-agent-choosing-metrics",
        "qa-agent-running-evaluations",
        "qa-agent-improving-hyperparameters",
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "Medical Chatbot",
      items: [
        "tutorial-llm-application-example",
        "tutorial-metrics-defining-an-evaluation-criteria",
        "tutorial-metrics-selection",
        "tutorial-evaluations-running-an-evaluation",
        "tutorial-evaluations-hyperparameters",
        "tutorial-evaluations-catching-regressions",
        "tutorial-dataset-synthesis",
        "tutorial-dataset-confident",
        "tutorial-production-monitoring",
        "tutorial-production-evaluation",
      ],
      collapsed: true,
    },
  ],
};
