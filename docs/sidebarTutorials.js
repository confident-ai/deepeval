module.exports = {
  tutorials: [
    {
      type: "category",
      label: "Getting Started",
      className: "sidebar-item-getting-started",
      items: [
        "tutorial-introduction",
        "tutorial-setup"
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "Meeting Summarizer",
      className: "sidebar-item-icon-agent",
      items: [
        "summarization-agent/tutorial-summarization-introduction", 
        "summarization-agent/tutorial-summarization-development",
        "summarization-agent/tutorial-summarization-evaluation",
        "summarization-agent/tutorial-summarization-improvement",
        "summarization-agent/tutorial-summarization-deployment",
      ],
      collapsed: true,
    },
    // {
    //   type: "category",
    //   label: "Legal Doc Summarization",
    //   items: [
    //     "doc-summarization-introduction",
    //     "doc-summarization-defining-a-criteria",
    //     "doc-summarization-selecting-metrics",
    //     "doc-summarization-running-an-evaluation",
    //     "doc-summarization-iterating-on-hyperparameters",
    //     "doc-summarization-catching-llm-regressions",
    //     "doc-summarization-annotating-datasets",
    //     "doc-summarization-using-datasets",
    //   ],
    //   collapsed: true,
    // },
    {
      type: "category",
      label: "RAG QA Agent",
      items: [
        "qa-agent-introduction",
        "qa-agent-generating-a-synthetic-dataset",
        "qa-agent-defining-an-evaluation-criteria",
        "qa-agent-choosing-metrics",
        "qa-agent-running-evaluations",
        "qa-agent-improving-hyperparameters",
      ],
      collapsed: true,
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
