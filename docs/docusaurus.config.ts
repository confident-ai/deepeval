import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { themes as prismThemes } from 'prism-react-renderer';

const confidentRefPagePath = require.resolve(
  './src/clientModules/confident-ref-page'
);

const config: Config = {
  clientModules: [confidentRefPagePath],
  plugins: [
    'docusaurus-plugin-sass',
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'tutorials',
        path: 'tutorials',
        routeBasePath: 'tutorials',
        sidebarPath: require.resolve('./sidebarTutorials.js'),
        editUrl: 'https://github.com/confident-ai/deepeval/edit/main/docs/',
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'guides',
        path: 'guides',
        routeBasePath: 'guides',
        sidebarPath: require.resolve('./sidebarGuides.js'),
        editUrl: 'https://github.com/confident-ai/deepeval/edit/main/docs/',
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'integrations',
        path: 'integrations',
        routeBasePath: 'integrations',
        sidebarPath: require.resolve('./sidebarIntegrations.js'),
        editUrl: 'https://github.com/confident-ai/deepeval/edit/main/docs/',
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'changelog',
        path: 'changelog',
        routeBasePath: 'changelog',
        sidebarPath: require.resolve('./sidebarChangelog.js'),
        editUrl: 'https://github.com/confident-ai/deepeval/edit/main/docs/',
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      '@docusaurus/plugin-content-blog',
      {
        id: 'blogs',
        path: 'blog',
        routeBasePath: 'blog',
        blogSidebarCount: 0,
      },
    ],
  ],

  title: 'DeepEval by Confident AI - The LLM Evaluation Framework',
  tagline: 'Evaluation Framework for LLMs',
  favicon: 'img/favicon.ico',

  url: 'https://deepeval.com',
  baseUrl: '/',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      '@docusaurus/preset-classic',
      {
        // blog: {
        //   blogTitle: 'DeepEval Blog',
        //   blogDescription: 'The official LLM evaluation blog',
        //   blogSidebarCount: 'ALL',
        // },
        blog: false,
        docs: {
          path: 'docs',
          editUrl: 'https://github.com/confident-ai/deepeval/edit/main/docs/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          sidebarPath: require.resolve('./sidebars.js'),
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: require.resolve('./src/css/custom.scss'),
        },
        gtag: {
          trackingID: 'G-N2EGDDYG9M',
          anonymizeIP: false,
        },
      } satisfies Preset.Options, // <-- Type checking applied here
    ],
  ],
  themes: ['@docusaurus/theme-mermaid'],
  markdown: {
    mermaid: true,
  },
  scripts: [
    {
      src: 'https://plausible.io/js/script.tagged-events.js',
      defer: true,
      'data-domain': 'deepeval.com',
    },
    {
      src: 'https://widget.kapa.ai/kapa-widget.bundle.js',
      'data-website-id': 'a3177869-c654-4b86-9c92-e4b4416f66e0',
      'data-project-name': 'DeepEval',
      'data-button-position-bottom': '2rem',
      'data-button-position-right': '2rem',
      'data-project-color': '#fff',
      'data-button-text-color': '#000',
      'data-project-logo':
        'https://pbs.twimg.com/profile_images/1888060560161574912/qbw1-_2g_400x400.png',
      'data-modal-title': 'Ask DeepEval',
      'data-modal-disclaimer':
        "All the following results are AI generated, if you can't find the solution you're looking for, ping us in [Discord](https://discord.gg/a3K9c8GRGt) we'd be happy to have you!",
      'data-modal-example-questions':
        'Can I create a dataset using my knowledge base?, Can I create a custom metrics for my use-case?',
      'data-uncertain-answer-callout':
        "It would be better to ask this question directly in DeepEval's [Discord](https://discord.gg/a3K9c8GRGt) channel.",
      async: true,
    },
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
    {
      href: 'https://fonts.googleapis.com/css2?family=Lexend+Deca:wght@500&display=swap',
      type: 'text/css',
    },
  ],
  headTags: [
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'Organization',
        name: 'DeepEval by Confident AI',
        alternateName: 'DeepEval - The LLM Evaluation Framework',
        url: 'https://deepeval.com',
        logo: 'https://deepeval.com/icons/DeepEval.svg',
        sameAs: [
          'https://github.com/confident-ai/deepeval',
          'https://x.com/deepeval',
          'https://discord.gg/a3K9c8GRGt',
        ],
      }),
    },
  ],
  themeConfig: {
    image: 'img/social_card.png',
    navbar: {
      logo: {
        alt: 'DeepEval Logo',
        src: 'icons/DeepEval.svg',
      },
      items: [
        {
          to: 'docs/getting-started',
          position: 'left',
          label: 'Docs',
          activeBasePath: 'docs',
        },
        {
          to: 'guides/guides-ai-agent-evaluation',
          position: 'left',
          label: 'Guides',
          activeBasePath: 'guides',
        },
        {
          to: 'tutorials/tutorial-introduction',
          position: 'left',
          label: 'Tutorials',
          activeBasePath: 'tutorials',
        },
        {
          to: 'integrations/models/openai',
          position: 'left',
          label: 'Integrations',
          activeBasePath: 'integrations',
        },
        { to: 'blog', label: 'Blog', position: 'left' },
        {
          to: 'changelog',
          position: 'left',
          label: 'Changelog',
          activeBasePath: 'changelog',
        },
        {
          href: 'https://discord.gg/a3K9c8GRGt',
          className: 'header-discord-link',
          position: 'right',
        },
        {
          href: 'https://github.com/confident-ai/deepeval',
          position: 'right',
          className: 'header-github-link',
        },
        {
          href: 'https://x.com/deepeval',
          className: 'header-twitter-link',
          position: 'right',
        },
      ],
    },
    algolia: {
      appId: '7U9PQIW1ZA',
      apiKey: 'fb799aeac8bcd0f6b9e0e233a385ad33',
      indexName: 'confident-ai',
      contextualSearch: true,
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    announcementBar: {
      id: 'announcementBar-1',
      content:
        '⭐️ If you like DeepEval, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/confident-ai/deepeval">GitHub</a>! ⭐️',
      backgroundColor: '#fff',
      textColor: '#091E42',
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            { label: 'Introduction', to: '/docs/getting-started' },
            {
              label: 'Confident AI',
              to: 'https://www.confident-ai.com/docs',
              className: 'utm--footer_documentation',
            },
            { label: 'Tutorials', to: '/tutorials/tutorial-introduction' },
            { label: 'Guides', to: '/guides/guides-ai-agent-evaluation' },
          ],
        },
        {
          title: 'Articles You Must Read',
          items: [
            {
              label: 'LLM evaluation metrics',
              to: 'https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation',
              className: 'utm--footer_articles',
            },
            {
              label: 'LLM-as-a-judge',
              to: 'https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method',
              className: 'utm--footer_articles',
            },
            {
              label: 'LLM testing',
              to: 'https://www.confident-ai.com/blog/llm-testing-in-2024-top-methods-and-strategies',
              className: 'utm--footer_articles',
            },
            {
              label: 'LLM chatbot evaluation',
              to: 'https://www.confident-ai.com/blog/llm-chatbot-evaluation-explained-top-chatbot-evaluation-metrics-and-testing-techniques',
              className: 'utm--footer_articles',
            },
          ],
        },
        {
          title: 'Evaluation Community',
          items: [
            { label: 'GitHub', to: 'https://github.com/confident-ai/deepeval' },
            { label: 'Discord', to: 'https://discord.gg/a3K9c8GRGt' },
            {
              label: 'Newsletter',
              to: 'https://confident-ai.com/blog',
              className: 'utm--footer_newsletter',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Confident AI Inc. Built with ❤️ and confidence.`,
    },
    prism: {
      theme: prismThemes.nightOwl,
      additionalLanguages: ['python'],
      magicComments: [
        {
          className: 'theme-code-block-highlighted-line',
          line: 'highlight-next-line',
          block: { start: 'highlight-start', end: 'highlight-end' },
        },
        {
          className: 'code-block-error-message',
          line: 'highlight-next-line-error-message',
        },
        {
          className: 'code-block-info-line',
          line: 'highlight-next-line-info',
          block: {
            start: 'highlight-info-start',
            end: 'highlight-info-end',
          },
        },
      ],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
