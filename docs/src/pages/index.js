import React, { useEffect, useRef } from 'react';
import styles from './index.module.scss';
import LayoutProvider from '@theme/Layout/Provider';
import Footer from '@theme/Footer';
import Link from '@docusaurus/Link';
import Navbar from '@theme/Navbar';
import { useColorMode } from '@docusaurus/theme-common';
import { Envelope } from '../components';
import {
  Terminal,
  Scale,
  MessageSquare,
  Image,
  Database,
  Blocks,
  ClipboardList,
  Network,
  MessageCircleQuestion,
} from 'lucide-react';

const integrationCards = [
  {
    name: 'OpenAI',
    logo: 'https://www.svgrepo.com/show/306500/openai.svg',
    link: '/integrations/frameworks/openai',
    description:
      "Direct integration with OpenAI's chat completion and responses APIs.",
  },
  {
    name: 'LangChain',
    logo: 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/langchain.png',
    link: '/integrations/frameworks/langchain',
    description:
      'Popular framework for developing applications with large language models.',
  },
  {
    name: 'Pydantic AI',
    logo: 'https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/pydantic-srs7pxjs9skodrjb64x86f.png/pydantic-ae96ag6mv67bf6hz5726v8.png?_a=DATAg1AAZAA0',
    link: '/integrations/frameworks/pydanticai',
    description:
      'Type-safe agent framework built on Pydantic for Python applications.',
  },
  {
    name: 'LlamaIndex',
    logo: 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/llamaindex.png',
    link: '/integrations/frameworks/llamaindex',
    description:
      'Framework that makes it easy to build knowledge agents from complex data.',
  },
  // Center cell is index 4 - reserved for logo
  {
    name: 'LangGraph',
    logo: 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/langgraph.png',
    link: '/integrations/frameworks/langgraph',
    description:
      'Graph-based approach to building stateful multi-actor applications.',
  },
  {
    name: 'OpenAI Agents',
    logo: 'https://www.svgrepo.com/show/306500/openai.svg',
    link: '/integrations/frameworks/openai-agents',
    description:
      "Integration with OpenAI's agent framework for intelligent assistants.",
  },
  {
    name: 'Crew AI',
    logo: 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/crewai.png',
    link: '/integrations/frameworks/crewai',
    description:
      'Multi-agent orchestration framework for collaborative AI workflows.',
  },
  {
    name: 'Anthropic',
    logo: 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/anthropic.png',
    link: '/integrations/frameworks/anthropic',
    description:
      'Unified API for calling 100+ LLM APIs using the OpenAI format.',
  },
];

const featureCards = [
  {
    icon: Terminal,
    title: 'Unit-Testing for LLMs',
    link: '/docs/evaluation-end-to-end-llm-evals#use-deepeval-test-run-in-cicd-pipelines',
    description:
      'Native integration with Pytest, that fits right in your CI workflow.',
  },
  {
    icon: Scale,
    title: 'LLM-as-a-Judge Metrics',
    link: '/docs/metrics-introduction',
    description:
      '50+ research-backed metrics, including custom G-Eval and deterministic metrics.',
  },
  {
    icon: MessageSquare,
    title: 'Single and Multi-Turn Evals',
    link: '/docs/evaluation-test-cases',
    description:
      'Covers any use cases, any system architecture, including multi-turn.',
  },
  {
    icon: Image,
    title: 'Native Multi-Modal Support',
    link: '/docs/evaluation-test-cases-multimodal',
    description:
      'Evaluate text, images, and audio with built-in multi-modal test cases.',
  },
  {
    icon: Database,
    title: 'Generation & Simulation',
    link: '/docs/synthesizer-introduction',
    description:
      'No test data? No problem. Generate synthetic data and simulate conversations.',
  },
  {
    icon: Blocks,
    title: 'Auto-Optimize Prompts',
    link: '/docs/prompt-optimization-introduction',
    description:
      'No need to manually tweak prompts. DeepEval automatically optimizes prompts for you.',
  },
];

const sotaCards = [
  {
    icon: ClipboardList,
    title: 'G-Eval',
    link: '/docs/metrics-llm-evals',
    description:
      'Criteria-based, chain-of-thought reasoning for nuanced, subjective scoring via form-filling paradigms.',
  },
  {
    icon: Network,
    title: 'DAG',
    link: '/docs/metrics-dag',
    description:
      'A tree-based, directed acyclic graph approach for evaluating objective multi-step conditional scoring.',
  },
  {
    icon: MessageCircleQuestion,
    title: 'QAG',
    link: '/docs/metrics-qag',
    description:
      'Question-Answer Generation for equation-based scoring based on close-ended questions.',
  },
];

function Corners() {
  return (
    <>
      <span className={styles.corner} />
      <span className={styles.corner} />
      <span className={styles.corner} />
      <span className={styles.corner} />
    </>
  );
}

function DashboardStack({ images }) {
  const count = images.length;
  return (
    <div className={styles.dashboardStack} style={{ '--stack-count': count }}>
      {images.map((src, index) => (
        <img
          key={index}
          src={src}
          alt={`Dashboard screenshot ${index + 1}`}
          className={styles.dashboardImage}
          style={{
            '--i': index,
            zIndex: index + 1,
          }}
        />
      ))}
    </div>
  );
}

function FeatureCard({ title, link, description, icon: Icon }) {
  return (
    <Link to={link} className={styles.featureCard}>
      <span className={styles.iconWrapper}>
        <Icon size={28} strokeWidth={1.5} />
      </span>
      <div className={styles.featureCardContainer}>
        <span className={styles.title}>
          {title}
          <img src="icons/right-arrow.svg" />
        </span>
      </div>
      <p className={styles.description}>{description}</p>
    </Link>
  );
}

function IntegrationCard({ name, logo, link, isLastColumn, isLastRow }) {
  const classes = [
    styles.integrationCard,
    isLastColumn && styles.noRightBorder,
    isLastRow && styles.noBottomBorder,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <Link to={link} className={classes}>
      <span className={styles.integrationIconWrapper}>
        <img src={logo} alt={`${name} logo`} />
      </span>
      <span className={styles.integrationName}>{name}</span>
    </Link>
  );
}

class Index extends React.Component {
  handleConfident = () => {
    window.open('https://confident-ai.com', '_blank');
  };

  render() {
    return (
      <div className={styles.mainRapper}>
        <div className={styles.section} style={{ paddingBottom: '0' }}>
          <div className={styles.heroContainer}>
            <h1 className={styles.mainTitle}>
              <Corners />
              The LLM Evaluation Framework
            </h1>
            <p className={styles.description}>
              <span className={styles.dollarSign}>$</span> used by some of the
              world's leading AI companies, DeepEval enables you to build{' '}
              <strong>reliable evaluation pipelines</strong> to test any AI
              system
              <span className={styles.cursor}>_</span>
            </p>
            <div className={styles.ctaContainer}>
              <Link to={'/docs/getting-started'} className={styles.button}>
                Get Started
              </Link>
            </div>
            <div className={styles.envelopeWrapper}>
              <Envelope />
            </div>
          </div>
        </div>
        {/* <div className={styles.logoCloud}>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
          <div className={styles.logoItem}>
            <span className={styles.logoPlaceholder}>Company</span>
          </div>
        </div> */}
        <div className={styles.section}>
          <div className={styles.container}>
            <div className={styles.title}>
              <strong>Why DeepEval?</strong> The most comprehensive framework to
              evaluate AI apps.
            </div>
            <div className={styles.featuresContainer}>
              <Corners />
              {featureCards.map((feature, index) => (
                <FeatureCard key={index} {...feature} />
              ))}
            </div>
          </div>
        </div>
        <div className={styles.section}>
          <div className={styles.container}>
            <div className={styles.title}>
              <strong>SOTA Evaluation Techniques</strong> Research-backed
              metrics to ensure utmost reliability.
            </div>
            <div className={styles.sotaFeaturesContainer}>
              <Corners />
              {sotaCards.map((feature, index) => (
                <FeatureCard key={index} {...feature} />
              ))}
            </div>
          </div>
        </div>
        <div className={styles.section}>
          <div
            className={`${styles.container} ${styles.productionizeContainer}`}
            style={{ alignItems: 'flex-start' }}
          >
            <div className={styles.title}>
              <strong>An All-in-One Eval Ecosystem</strong> Use DeepEval on
              Confident AI
            </div>
            <div className={styles.pillsContainer}>
              <div className={styles.pill}>Regression Testing</div>
              <div className={styles.pill}>AI Experiments</div>
              <div className={styles.pill}>Dataset Management</div>
              <div className={styles.pill}>Observability & Tracing</div>
              <div className={styles.pill}>Online Monitoring</div>
              <div className={styles.pill}>Human Annotations</div>
            </div>
            <div className={styles.productionizeContent}>
              <div>
                <p className={styles.description}>
                  By the authors of DeepEval, Confident AI is a cloud LLM
                  evaluation platform. It allows you to use DeepEval for
                  team-wide, collaborative AI testing.
                </p>
                <button className={styles.button}>
                  Try DeepEval Free on Confident AI
                </button>
              </div>
              <DashboardStack
                images={[
                  '/img/white-3.png',
                  '/img/white.png',
                  '/img/white-2.png',
                ]}
              />
            </div>
          </div>
        </div>
        <div className={styles.section}>
          <div className={styles.container}>
            <div className={styles.title}>
              <strong>Built for Production-Grade Standards</strong> Integrates
              with your AI stack seamlessly.
            </div>
            <div className={styles.integrationGrid}>
              <Corners />
              {Array.from({ length: 9 }).map((_, index) => {
                const isCenterCell = index === 4;
                const row = Math.floor(index / 3);
                const col = index % 3;
                const isLastColumn = col === 2;
                const isLastRow = row === 2;

                // Map grid index to integration card index (skip center)
                const cardIndex = index < 4 ? index : index - 1;
                const integration = integrationCards[cardIndex];

                if (isCenterCell) {
                  return (
                    <div
                      key={index}
                      className={`${styles.integrationCell} ${styles.centerCell}`}
                    >
                      <div className={styles.logoCell}>
                        <img
                          src="/icons/deepeval-logo.svg"
                          alt="DeepEval"
                          className={styles.deepevalLogo}
                        />
                      </div>
                    </div>
                  );
                }

                return (
                  <IntegrationCard
                    key={index}
                    name={integration.name}
                    logo={integration.logo}
                    link={integration.link}
                    description={integration.description}
                    isLastColumn={isLastColumn}
                    isLastRow={isLastRow}
                  />
                );
              })}
            </div>
          </div>
        </div>
        <div className={styles.section}>
          <div className={styles.container}>
            <h2 className={styles.heading}>
              The Framework of Choice When Reliability Matters
            </h2>
            <div className={styles.codeBox}>
              <Corners />
              <span className={styles.dollarSign}>$</span> pip install deepeval
            </div>
            <div className={styles.ctaContainer}>
              <Link to={'/docs/getting-started'} className={styles.button}>
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

function HomePageContent(props) {
  const { colorMode, setColorMode } = useColorMode();
  const originalColorMode = useRef(colorMode);

  useEffect(() => {
    setColorMode('dark');
    return () => {
      setColorMode(originalColorMode.current);
    };
  }, []);

  return (
    <>
      <Navbar {...props} />
      <Index {...props} />
      <Footer {...props} />
    </>
  );
}

export default function (props) {
  return (
    <LayoutProvider>
      <HomePageContent {...props} />
    </LayoutProvider>
  );
}
