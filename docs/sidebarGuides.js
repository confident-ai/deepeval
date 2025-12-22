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
