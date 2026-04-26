import Link from "next/link";
import { ExternalLink } from "lucide-react";
import { gitConfig } from "@/lib/shared";
import styles from "./Footer.module.scss";

type FooterLink = {
  label: string;
  href: string;
};

type FooterColumn = {
  heading: string;
  links: FooterLink[];
};

const COLUMNS: FooterColumn[] = [
  {
    heading: "Product",
    links: [
      { label: "Getting Started", href: "/docs/getting-started" },
      { label: "Metrics", href: "/docs/metrics-introduction" },
      { label: "Golden Synthesizer", href: "/docs/golden-synthesizer" },
      {
        label: "Prompt Optimization",
        href: "/docs/prompt-optimization-introduction",
      },
      { label: "Benchmarks", href: "/docs/benchmarks-introduction" },
    ],
  },
  {
    heading: "Very Useful Reads",
    links: [
      {
        label: "AI Agent Evaluation",
        href: "/guides/guides-ai-agent-evaluation",
      },
      {
        label: "Multi-Turn Simulation",
        href: "/guides/guides-multi-turn-simulation",
      },
      {
        label: "LLM Tracing + Evals",
        href: "/guides/guides-llm-observability",
      },
      { label: "Evaluating RAG", href: "/guides/guides-rag-evaluation" },
    ],
  },
  {
    heading: "Ecosystem",
    links: [
      { label: "Integrations", href: "/integrations/models/openai" },
      { label: "Confident AI", href: "https://www.confident-ai.com" },
      { label: "DeepTeam", href: "https://trydeepteam.com" },
    ],
  },
];

const isExternal = (href: string) => /^https?:\/\//i.test(href);

const GithubMark = ({ className }: { className?: string }) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
  >
    <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
  </svg>
);

const FooterLinkItem = ({ link }: { link: FooterLink }) => {
  const external = isExternal(link.href);
  const content = (
    <>
      {link.label}
      {external ? (
        <ExternalLink className={styles.externalIcon} aria-hidden="true" />
      ) : null}
    </>
  );

  return (
    <li>
      {external ? (
        <a href={link.href} target="_blank" rel="noopener noreferrer">
          {content}
        </a>
      ) : (
        <Link href={link.href}>{content}</Link>
      )}
    </li>
  );
};

const Footer = () => {
  return (
    <footer className={styles.footer}>
      <div className={styles.shell}>
        <div className={styles.inner}>
          <div className={styles.brand}>
            {/* Rendered as a masked <span> (see `.logo` in the module)
             *  so `background-color: var(--color-fd-foreground)` drives
             *  the fill — keeps the mark legible in both light and dark
             *  modes without forking the SVG asset. `role="img"` + aria
             *  label preserves the <img>'s accessibility semantics. */}
            <span
              className={styles.logo}
              role="img"
              aria-label="DeepEval"
            />
            <p className={styles.tagline}>
              Open-source LLM evaluation framework. Apache 2.0 licensed.
            </p>
            <a
              className={styles.starButton}
              href={`https://github.com/${gitConfig.user}/${gitConfig.repo}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <GithubMark className={styles.starIcon} />
              <span>Star us on GitHub</span>
            </a>
            <span>
              &copy; {new Date().getFullYear()} Confident AI Inc. Made with{" "}
              <span className={styles.heart} aria-hidden="true">
                💜
              </span>{" "}
              and confidence.
            </span>
          </div>

          <nav className={styles.columns} aria-label="Footer">
            {COLUMNS.map((column) => (
              <div key={column.heading} className={styles.column}>
                <h4 className={styles.heading}>{column.heading}</h4>
                <ul className={styles.list}>
                  {column.links.map((link) => (
                    <FooterLinkItem key={link.label} link={link} />
                  ))}
                </ul>
              </div>
            ))}
          </nav>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
