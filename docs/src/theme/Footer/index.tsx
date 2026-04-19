import React from 'react';
import Link from '@docusaurus/Link';
import { ExternalLink } from 'lucide-react';
import styles from './styles.module.scss';

type FooterLink = { label: string; to: string };

const LINKS: Record<'product' | 'reads' | 'ecosystem', FooterLink[]> = {
  product: [
    { label: 'Getting Started', to: '/docs/getting-started' },
    { label: 'Metrics', to: '/docs/metrics-introduction' },
    { label: 'Synthesizer', to: '/docs/synthesizer-introduction' },
    {
      label: 'Prompt Optimization',
      to: '/docs/prompt-optimization-introduction',
    },
    { label: 'Benchmarks', to: '/docs/benchmarks-introduction' },
  ],
  reads: [
    { label: 'AI Agent Evaluation', to: '/guides/guides-ai-agent-evaluation' },
    {
      label: 'Multi-Turn Simulation',
      to: '/guides/guides-multi-turn-simulation',
    },
    { label: 'LLM Tracing + Evals', to: '/guides/guides-llm-observability' },
    { label: 'Evaluating RAG', to: '/guides/guides-rag-evaluation' },
  ],
  ecosystem: [
    { label: 'Integrations', to: '/integrations/models/openai' },
    { label: 'Confident AI', to: 'https://www.confident-ai.com' },
    { label: 'DeepTeam', to: 'https://trydeepteam.com' },
  ],
};

function isExternal(to: string) {
  return /^https?:\/\//i.test(to);
}

function FooterLinkItem({ link }: { link: FooterLink }) {
  const external = isExternal(link.to);
  return (
    <li>
      <Link to={link.to} className={styles.link}>
        {link.label}
        {external && (
          <ExternalLink
            size={12}
            strokeWidth={2}
            className={styles.externalIcon}
            aria-hidden="true"
          />
        )}
      </Link>
    </li>
  );
}

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
}

function Footer() {
  return (
    <footer className={styles.footer} data-utm-content="footer">
      <div className={styles.inner}>
        <div className={styles.top}>
          <div className={styles.brand}>
            <img
              src="/icons/DeepEval.svg"
              alt="DeepEval"
              className={styles.logo}
            />
            <p className={styles.tagline}>
              Open-source LLM evaluation framework. Apache 2.0 licensed.
            </p>
            <Link
              href="https://github.com/confident-ai/deepeval"
              className={styles.starButton}
            >
              <GitHubIcon /> Star us on GitHub
            </Link>
          </div>

          <div className={styles.columns}>
            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Product</h4>
              <ul className={styles.columnList}>
                {LINKS.product.map((link) => (
                  <FooterLinkItem key={link.label} link={link} />
                ))}
              </ul>
            </div>

            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Very Useful Reads</h4>
              <ul className={styles.columnList}>
                {LINKS.reads.map((link) => (
                  <FooterLinkItem key={link.label} link={link} />
                ))}
              </ul>
            </div>

            <div className={styles.column}>
              <h4 className={styles.columnTitle}>Ecosystem</h4>
              <ul className={styles.columnList}>
                {LINKS.ecosystem.map((link) => (
                  <FooterLinkItem key={link.label} link={link} />
                ))}
              </ul>
            </div>
          </div>
        </div>

        <div className={styles.divider} />

        <div className={styles.bottom}>
          <span className={styles.copyright}>
            &copy; {new Date().getFullYear()} Confident AI Inc. Made with 💜 and
            confidence.
          </span>
        </div>
      </div>
    </footer>
  );
}

export default React.memo(Footer);
