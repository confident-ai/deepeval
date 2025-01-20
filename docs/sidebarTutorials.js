module.exports = {
    tutorials: [
      "tutorial-introduction",
      {
        type: "category",
        label: "Before We Begin",
        items: ["tutorial-setup", "tutorial-llm-application-example"],
        collapsed: false,
      },
      {
        type: "category",
        label: "Choosing Your Metrics",
        items: [
          "tutorial-metrics-defining-an-evaluation-criteria",
          "tutorial-metrics-selection",
        ],
        collapsed: false,
      },
      {
        type: "category",
        label: "Evaluating Your LLM",
        items: [
          "tutorial-evaluations-running-an-evaluation",
          "tutorial-evaluations-hyperparameters",
          "tutorial-evaluations-catching-regressions",
        ],
        collapsed: false,
      },
      {
        type: "category",
        label: "Improving Your Evaluation Dataset",
        items: [
          "tutorial-dataset-synthesis",
          "tutorial-dataset-confident",
        ],
        collapsed: false,
      },
      {
        type: "category",
        label: "Continous Evaluation in Production",
        items: [
          "tutorial-production-monitoring",
          "tutorial-production-evaluation",
        ],
        collapsed: false,
      },
    ],
  };
  