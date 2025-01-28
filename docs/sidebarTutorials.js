module.exports = {
    tutorials: [
      "tutorial-introduction",
      "tutorial-setup", 
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
          "tutorial-production-evaluation"
        ],
        collapsed: false,
      },
    ],
  };
  