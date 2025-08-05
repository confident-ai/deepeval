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
        "summarization-agent/tutorial-summarization-introduction",
        "summarization-agent/tutorial-summarization-development",
        "summarization-agent/tutorial-summarization-evaluation",
        "summarization-agent/tutorial-summarization-improvement",
        "summarization-agent/tutorial-summarization-deployment",
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "RAG QA Agent",
      className: "sidebar-item-icon-rag",
      items: [
        "rag-qa-agent/tutorial-rag-qa-introduction", 
        "rag-qa-agent/tutorial-rag-qa-development",
        "rag-qa-agent/tutorial-rag-qa-evaluation",
        "rag-qa-agent/tutorial-rag-qa-improvement",
        "rag-qa-agent/tutorial-rag-qa-deployment",
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
