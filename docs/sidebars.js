module.exports = {
  md: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
          'getting-started',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Evaluation',
      items: [
        'evaluation-introduction',
        'evaluation-test-cases',
        'evaluation-datasets',
        {
          type: 'category',
          label: 'Metrics',
          items: [
              'metrics-introduction',
              'metrics-llm-evals',
              'metrics-hallucination',
              'metrics-answer-relevancy',
              'metrics-faithfulness',
              'metrics-contextual-precision',
              'metrics-contextual-relevancy',
              'metrics-contextual-recall',
              'metrics-ragas',
              'metrics-bias',
              'metrics-toxicity',
              'metrics-judgemental',
              'metrics-custom',
              'metrics-others',
          ],
          collapsed: true,
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Confident AI',
      items: [
        'confident-ai-introduction',
        'confident-ai-manage-datasets',
        'confident-ai-evaluate-datasets',
        'confident-ai-analyze-evaluations',
        'confident-ai-debug-evaluations',
        'confident-ai-track-events'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Integrations',
      items: [
        'integrations-introduction',
        'integrations-llamaindex'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Others',
      items: [
        'data-privacy',
      ],
      collapsed: false,
    },
  ]
}