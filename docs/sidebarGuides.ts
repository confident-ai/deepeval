module.exports = {
  guides: [
    {
      type: 'category',
      label: 'AI Agents',
      className: 'sidebar-item-icon-agent',
      items: [
        'guides-ai-agent-evaluation',
        'guides-ai-agent-evaluation-metrics',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Multi-Turn (chatbots)',
      className: 'sidebar-item-icon-chatbot',
      items: [
        'guides-multi-turn-evaluation',
        'guides-multi-turn-evaluation-metrics',
        'guides-multi-turn-simulation',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Retrieval Augmented Generation',
      className: 'sidebar-item-icon-rag',
      items: [
        'guides-rag-evaluation',
        'guides-rag-triad',
        'guides-using-synthesizer',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Customizations',
      className: 'sidebar-item-icon-customizations',
      items: [
        'guides-using-custom-llms',
        'guides-using-custom-embedding-models',
        'guides-building-custom-metrics',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Tracing + Evals',
      className: 'sidebar-item-icon-tracing',
      items: [
        'guides-tracing-multi-turn',
        'guides-tracing-ai-agents',
        'guides-tracing-rag',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Others',
      className: 'sidebar-item-icon-others',
      items: [
        'guides-optimizing-hyperparameters',
        'guides-regression-testing-in-cicd',
        'guides-llm-observability',
        'guides-red-teaming',
        'guides-answer-correctness-metric',
      ],
      collapsed: false,
    },
  ],
};
