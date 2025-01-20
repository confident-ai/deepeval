// const remarkMath = require('remark-math');
// const rehypeKatex = require('rehype-katex');
const remarkMathModule = import("remark-math");
const rehypeKatexModule = import("rehype-katex");
/** @type {import('@docusaurus/types').Config} */

module.exports = {
  plugins: [
    [
      "posthog-docusaurus",
      {
        apiKey: "phc_xuPeotBDIsi9uNBRS1ZqFD24IaOTx6lmt4ZCLfOnaee",
        appUrl: "https://us.i.posthog.com",
        enableInDevelopment: false,
      },
    ],
    "docusaurus-plugin-sass",
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "tutorials",
        path: "tutorials",
        routeBasePath: "tutorials", 
        sidebarPath: require.resolve("./sidebarTutorials.js"), 
        editUrl:
          "https://github.com/confident-ai/deepeval/edit/main/docs/",
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "confident-ai", 
        path: "confident-ai",
        routeBasePath: "confident-ai",
        sidebarPath: require.resolve("./sidebarConfidentAI.js"), 
        editUrl:
          "https://github.com/confident-ai/deepeval/edit/main/docs/",
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "guides", 
        path: "guides",
        routeBasePath: "guides",
        sidebarPath: require.resolve("./sidebarGuides.js"), 
        editUrl:
          "https://github.com/confident-ai/deepeval/edit/main/docs/",
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
  ],

  title: "DeepEval - The Open-Source LLM Evaluation Framework",
  tagline: "Evaluation Framework for LLMs",
  favicon: "img/fav.ico",

  // Set the production url of your site here
  url: "https://docs.confident-ai.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "@docusaurus/preset-classic",
      {
        docs: {
          path: "docs",
          editUrl:
            "https://github.com/confident-ai/deepeval/edit/main/docs/",
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          sidebarPath: require.resolve("./sidebars.js"),
          remarkPlugins: [remarkMathModule],
          rehypePlugins: [rehypeKatexModule],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.scss"),
        },
        gtag: {
          trackingID: "G-N2EGDDYG9M",
          anonymizeIP: false,
        },
      },
    ],
  ],
  scripts: [
    {
      src: "https://plausible.io/js/script.tagged-events.js",
      defer: true,
      "data-domain": "confident-ai.com",
    },
  ],
  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM",
      crossorigin: "anonymous",
    },
    {
      href: "https://fonts.googleapis.com/css2?family=Lexend+Deca:wght@500&display=swap",
      type: "text/css",
    },
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: "img/social_card.png",
      navbar: {
        logo: {
          alt: "DeepEval Logo",
          src: "icons/DeepEval.svg",
        },
        items: [
          {
            to: "docs/getting-started",
            position: "left",
            label: "Docs",
            activeBasePath: 'docs',
          },
          {
            to: "confident-ai/confident-ai-introduction",
            position: "left",
            label: "Confident AI",
            activeBasePath: 'confident-ai',
          },
          {
            to: "tutorials/tutorial-introduction",
            position: "left",
            label: "Tutorials",
            activeBasePath: 'tutorials',
          },
          {
            to: "guides/guides-rag-evaluation",
            position: "left",
            label: "Guides",
            activeBasePath: 'guides',
          },
          {
            href: "https://confident-ai.com/blog",
            position: "left",
            label: "Blogs",
          },
          {
            href: "https://confident-ai.com",
            className: "header-confident-link",
            position: "right",
          },
          {
            href: "https://discord.gg/a3K9c8GRGt",
            className: "header-discord-link",
            position: "right",
          },
          {
            href: "https://github.com/mr-gpt/deepeval",
            position: "right",
            className: "header-github-link",
          },
        ],
      },
      algolia: {
        appId: "7U9PQIW1ZA",
        apiKey: "fb799aeac8bcd0f6b9e0e233a385ad33",
        indexName: "confident-ai",
        contextualSearch: true,
      },
      colorMode: {
        defaultMode: "light",
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
      announcementBar: {
        id: "announcementBar-1",
        content:
          '⭐️ If you like DeepEval, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/confident-ai/deepeval">GitHub</a>! ⭐️',
        backgroundColor: "#fff",
        textColor: "#091E42",
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Documentation",
            items: [
              {
                label: "Introduction",
                to: "/docs/getting-started",
              },
              {label: "Confident AI", to: "/confident-ai/confident-ai-introduction"},
              {
                label: "Tutorials",
                to: "/tutorials/tutorial-introduction",
              },
              {
                label: "Guides",
                to: "/guides/guides-rag-evaluation",
              },
            ],
          },
          {
            title: "Community",
            items: [
              {
                label: "GitHub",
                to: "https://github.com/confident-ai/deepeval",
              },
              {
                label: "Discord",
                to: "https://discord.gg/a3K9c8GRGt",
              },
              {
                label: "Newsletter",
                to: "https://confident-ai.com/blog",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Confident AI Inc. Built with ❤️ and confidence.`,
      },
      prism: {
        additionalLanguages: ["python"],
        magicComments: [
          {
            className: "theme-code-block-highlighted-line",
            line: "highlight-next-line",
            block: { start: "highlight-start", end: "highlight-end" },
          },
          {
            className: "code-block-error-message",
            line: "highlight-next-line-error-message",
          },
          {
            className: "code-block-info-line",
            line: "highlight-next-line-info",
            block: {
              start: "highlight-info-start",
              end: "highlight-info-end",
            },
          },
        ],
      },
    }),
};
