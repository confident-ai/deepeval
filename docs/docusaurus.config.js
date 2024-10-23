// const remarkMath = require('remark-math');
// const rehypeKatex = require('rehype-katex');

/** @type {import('@docusaurus/types').Config} */

module.exports = async function createConfigAsync() {
  const remarkMathModule = await import("remark-math");
  const rehypeKatexModule = await import("rehype-katex");

  return {
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
    plugins: [
      "docusaurus-plugin-sass",
      [
        "posthog-docusaurus",
        {
          apiKey: "phc_qyCLAQQVAYN4mDhkZYSzBvFP2SMoiij0QIbnOwhosp0",
          appUrl: "https://docs.confident-ai.com", // optional
          enableInDevelopment: false, // optional
          // other options are passed to posthog-js init as is
        },
      ],
    ],
    presets: [
      [
        "@docusaurus/preset-classic",
        {
          docs: {
            path: "docs",
            editUrl:
              "https://github.com/confident-ai/deepeval/edit/main/docs/docs/",
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
            src: "icons/DeepEval..svg",
          },
          items: [
            {
              to: "docs/getting-started",
              position: "left",
              label: "Docs",
            },
            // {
            //   to: "docs/tutorial",
            //   position: 'left',
            //   label: 'Tutorials',
            // },

            {
              to: "https://confident-ai.com/blog",
              position: "left",
              label: "Blog",
            },
            {
              href: "https://confident-ai.com",
              position: "left",
              label: "Confident AI",
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
                // {
                //   label: 'Tutorial',
                //   to: '/docs/intro',
                // },
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
        scripts: [
          {
            src: "https://cdn.getkoala.com/v1/pk_c2f9444e23cd5fd7c2b34105b24d241bfdad/sdk.js",
            async: true,
            "data-cfasync": "false",
            defer: true,
          },
        ],
      }),
  };
};
