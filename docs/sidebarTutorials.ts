module.exports = {
  tutorials: [
    {
      type: "category",
      label: "Getting Started",
      className: "sidebar-item-getting-started",
      items: ["tutorial-introduction", "tutorial-setup"],
      collapsed: false,
    },
    {
      type: "category",
      label: "Meeting Summarizer",
      className: "sidebar-item-icon-agent",
      items: [
        "summarization-agent/introduction",
        "summarization-agent/development",
        "summarization-agent/evaluation",
        "summarization-agent/improvement",
        "summarization-agent/evals-in-prod",
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "RAG QA Agent",
      className: "sidebar-item-icon-rag",
      items: [
        "rag-qa-agent/introduction", 
        "rag-qa-agent/development",
        "rag-qa-agent/evaluation",
        "rag-qa-agent/improvement",
        "rag-qa-agent/evals-in-prod",
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "Medical Chatbot",
      className: "sidebar-item-icon-chatbot",
      items: [
        "medical-chatbot/introduction", 
        "medical-chatbot/development",
        "medical-chatbot/evaluation",
        "medical-chatbot/improvement",
        "medical-chatbot/evals-in-prod",
      ],
      collapsed: false,
    },
  ],
};
