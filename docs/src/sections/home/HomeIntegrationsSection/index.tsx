import styles from "./HomeSections.module.scss";

const integrations = [
  { name: "OpenAI", logo: "/icons/companies/openai.svg", alt: "OpenAI logo" },
  { name: "LangChain", logo: "/icons/frameworks/langchain.png", alt: "LangChain logo" },
  { name: "Pydantic AI", logo: "/icons/frameworks/pydanticai.png", alt: "Pydantic AI logo" },
  { name: "LlamaIndex", logo: "/icons/frameworks/llamaindex.png", alt: "LlamaIndex logo" },
  { name: "DeepEval", logo: "/icons/deepeval-logo.svg", alt: "DeepEval logo", deepeval: true },
  { name: "LangGraph", logo: "/icons/frameworks/langgraph.png", alt: "LangGraph logo" },
  { name: "OpenAI Agents", logo: "/icons/companies/openai.svg", alt: "OpenAI Agents logo" },
  { name: "Crew AI", logo: "/icons/frameworks/crewai.png", alt: "Crew AI logo" },
  { name: "Anthropic", logo: "/icons/frameworks/anthropic.png", alt: "Anthropic logo" },
];

const HomeIntegrationsSection: React.FC = () => {
  return (
    <section className={styles.section}>
      <div className={styles.sectionShell}>
        <div className={styles.sectionIntro}>
          <p className={styles.sectionEyebrow}>1 line integration</p>
          <h2 className={styles.sectionTitle}>
            Built for Production-Grade Standards Fits right in your existing AI
            stack.
          </h2>
        </div>

        <div className={styles.integrationsGrid}>
          {integrations.map((integration) => (
            <div key={integration.name} className={styles.integrationCard}>
              <span className={styles.integrationLogoWrap}>
                <img
                  className={integration.deepeval ? styles.deepevalMark : styles.integrationLogo}
                  src={integration.logo}
                  alt={integration.alt}
                />
              </span>
              <span className={styles.integrationName}>{integration.name}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};


export default HomeIntegrationsSection;
