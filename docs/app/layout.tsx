import type { Metadata } from 'next';
import Script from 'next/script';
import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import 'katex/dist/katex.css';
import { Geist, Space_Grotesk } from 'next/font/google';
import UtmCapture from '@/src/layouts/UtmCapture';
import SchemaInjector from '@/src/components/SchemaInjector/SchemaInjector';
import { buildWebSiteSchema } from '@/src/utils/schema-helpers';
import {
  appName,
  kapaConfig,
  siteDescription,
  siteTitle,
  siteUrl,
} from '@/lib/shared';

const sans = Geist({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

const heading = Space_Grotesk({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-heading',
  display: 'swap',
});

const disabledSearchHotKey = [
  {
    key: "__disabled__",
    display: null,
  },
];

// `%s` template mirrors Docusaurus' default `<title>` format so every
// SERP entry still reads "Page Title | {siteTitle}".
//
// `openGraph` / `twitter` defaults here set the site-wide baseline that
// every section inherits (Next's `generateMetadata` deep-merges onto
// this object). Per-page routes can override individual fields — the
// blog section adds `openGraph.type = 'article'` + publishedTime, the
// docs section adds per-page OG images, etc.
export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: siteTitle,
    template: `%s | ${siteTitle}`,
  },
  description: siteDescription,
  alternates: { canonical: '/' },
  openGraph: {
    type: 'website',
    siteName: appName,
    url: siteUrl,
    title: siteTitle,
    description: siteDescription,
  },
  twitter: {
    card: 'summary_large_image',
    site: '@deepeval',
    creator: '@deepeval',
    title: siteTitle,
    description: siteDescription,
  },
};

// Organization schema mirrored from the old Docusaurus `headTags` block
// (docusaurus.config.ts:161-181). Rendered once in <head> via the App
// Router layout — Next will keep JSON-LD scripts where they are placed.
const organizationJsonLd = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'DeepEval by Confident AI',
  alternateName: 'DeepEval - The LLM Evaluation Framework',
  url: siteUrl,
  logo: `${siteUrl}/icons/DeepEval.svg`,
  sameAs: [
    'https://github.com/confident-ai/deepeval',
    'https://x.com/deepeval',
    'https://discord.gg/a3K9c8GRGt',
  ],
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${heading.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/*
          Two site-wide JSON-LD blocks rendered once per page:
          `Organization` (mirrored from the old Docusaurus `headTags`)
          and `WebSite` (so crawlers have a canonical top-level entity
          to hang everything else off of). Both use the shared
          `SchemaInjector` helper, which safely escapes `</` inside the
          serialized JSON.
        */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationJsonLd) }}
        />
        <SchemaInjector schema={buildWebSiteSchema()} />
        {/*
          Kapa.ai "Ask AI" widget.

          Deliberately rendered as a native `<script async>` in <head>
          rather than `next/script` — Kapa's bundle reads its config
          off `document.currentScript.dataset` during its initial
          parse, and `next/script`'s loader rewrites / relocates the
          tag in a way that drops those attributes from the
          currentScript reference at runtime. A plain `<script async>`
          lands in the initial HTML exactly the way Docusaurus' old
          `scripts` block did, which is what the widget expects.

          Attribute names here match Kapa's current (post-2024) widget
          API, which renamed several of the legacy attributes the old
          deepeval.com config used:

            legacy                         → current
            data-button-hide               → data-launcher-button-hidden
            data-button-text-color         → (removed — use component styles)
            data-modal-disclaimer          → data-chat-disclaimer
            data-modal-example-questions   → data-example-questions
            data-modal-override-selector   → data-modal-override-open-selector
                                              (or -open-class for bare names)

          `data-launcher-button-hidden="true"` hides the floating
          launcher; `data-modal-override-open-class="ask-ai-trigger"`
          opens the modal on any click that hits an element with that
          class — see `<AskAIButton>`. Using the `-class` variant
          instead of `-selector` keeps the config value as a bare class
          name (no leading dot to escape).
        */}
        {/*
          Kapa component-style overrides (see the component table at
          docs.kapa.ai/integrations/website-widget/configuration/component-styles).
          The default modal, inner wrapper, and the example-question
          buttons all ship with generous rounded corners; we flatten
          every layer so Kapa's modal reads as one sharp rectangle
          that matches the rest of the DeepEval site (buttons,
          callouts, and code blocks are all square).
        */}
        <script
          async
          src="https://widget.kapa.ai/kapa-widget.bundle.js"
          data-website-id={kapaConfig.websiteId}
          data-project-name={kapaConfig.projectName}
          data-project-color={kapaConfig.projectColor}
          data-project-logo={kapaConfig.projectLogo}
          data-modal-title={kapaConfig.modalTitle}
          data-chat-disclaimer={kapaConfig.chatDisclaimer}
          data-example-questions={kapaConfig.exampleQuestions}
          data-uncertain-answer-callout={kapaConfig.uncertainAnswerCallout}
          data-launcher-button-hidden="true"
          data-modal-override-open-class={kapaConfig.triggerClass}
          data-modal-border-radius="0"
          data-modal-inner-border-radius="0"
          data-modal-content-border-radius="0"
          data-modal-header-border-radius="0"
          data-example-question-button-border-radius="0"
          data-query-input-border-radius="0"
        />
      </head>
      <body className="flex flex-col min-h-screen font-sans">
        <UtmCapture />
        <RootProvider search={{ hotKey: disabledSearchHotKey }}>
          {children}
        </RootProvider>
        {/*
          Analytics parity with the old Docusaurus site
          (docusaurus.config.ts:111-127). `afterInteractive` keeps these
          out of the critical path while still firing on every page
          navigation — same effective behavior as the old
          `<script defer>` tags.
        */}
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-N2EGDDYG9M"
          strategy="afterInteractive"
        />
        <Script id="ga-init" strategy="afterInteractive">
          {`window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-N2EGDDYG9M');`}
        </Script>
        <Script
          src="https://plausible.io/js/script.tagged-events.js"
          data-domain="deepeval.com"
          strategy="afterInteractive"
        />
      </body>
    </html>
  );
}
