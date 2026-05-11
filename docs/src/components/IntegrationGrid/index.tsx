import Image from "next/image";
import Link from "next/link";
import type { ComponentType, SVGProps } from "react";
import {
  CircleCIMark,
  GitHubMark,
  OpenAIMark,
  VercelAISDKMark,
} from "@site/src/components/BrandMarks";
import styles from "./IntegrationGrid.module.scss";

type Integration = {
  name: string;
  /**
   * Static file in /public for use with next/image. Kept as a fallback
   * and so the icon exists on disk even for integrations we render
   * inline (next/image is still nice for preloading / link previews).
   */
  logo: string;
  /** Docs page the card links to. */
  href: string;
  /**
   * Optional inline SVG component. When set, the icon is rendered as
   * real SVG in the React tree so `fill="currentColor"` picks up the
   * page's foreground color and survives light/dark mode toggles.
   * Loading via <img>/next/image puts the SVG in a separate document
   * context where `currentColor` can't inherit from the host page,
   * which is why monochrome brand marks (OpenAI, GitHub, CircleCI,
   * Vercel AI SDK) need to be inlined.
   */
  inline?: ComponentType<SVGProps<SVGSVGElement>>;
};

type Category = {
  label: string;
  items: Integration[];
  columns: number;
};

/* ---------- Category config ---------- */

/* Destination docs pages. Mistral, Vercel AI SDK, and OpenTelemetry
 * don't have dedicated pages yet and fall back to the integrations
 * index. CI/CD tools all share the generic CI/CD unit-testing guide.
 * Swap any of these to a dedicated page when one lands. */
const CI_CD_DOCS = "/docs/evaluation-unit-testing-in-ci-cd";
const INTEGRATIONS_INDEX = "/integrations";

const MODEL_PROVIDERS: Category = {
  label: "Model Providers",
  columns: 4,
  items: [
    { name: "OpenAI", logo: "/icons/integrations/openai.svg", href: "/integrations/models/openai", inline: OpenAIMark },
    { name: "Claude", logo: "/icons/integrations/claude.svg", href: "/integrations/models/anthropic" },
    { name: "Gemini", logo: "/icons/integrations/gemini.svg", href: "/integrations/models/gemini" },
    { name: "Azure OpenAI", logo: "/icons/integrations/azure.svg", href: "/integrations/models/azure-openai" },
    { name: "AWS Bedrock", logo: "/icons/integrations/bedrock.svg", href: "/integrations/models/amazon-bedrock" },
    { name: "Vertex AI", logo: "/icons/integrations/vertext_ai.svg", href: "/integrations/models/vertex-ai" },
    { name: "Mistral", logo: "/icons/integrations/mistral.svg", href: INTEGRATIONS_INDEX },
    { name: "LiteLLM", logo: "/icons/integrations/litellm.svg", href: "/integrations/models/litellm" },
    { name: "Portkey", logo: "/icons/integrations/portkey.svg", href: "/integrations/models/portkey" },
  ],
};

const FRAMEWORKS: Category = {
  label: "Frameworks",
  columns: 3,
  items: [
    { name: "LangChain", logo: "/icons/integrations/langchain.svg", href: "/integrations/frameworks/langchain" },
    { name: "LlamaIndex", logo: "/icons/integrations/llamaindex.svg", href: "/integrations/frameworks/llamaindex" },
    { name: "CrewAI", logo: "/icons/integrations/crewai.svg", href: "/integrations/frameworks/crewai" },
    { name: "OpenAI Agents", logo: "/icons/integrations/openai.svg", href: "/integrations/frameworks/openai-agents", inline: OpenAIMark },
    { name: "LangGraph", logo: "/icons/integrations/langgraph.svg", href: "/integrations/frameworks/langgraph" },
    { name: "PydanticAI", logo: "/icons/integrations/pydanticai.svg", href: "/integrations/frameworks/pydanticai" },
    { name: "Anthropic", logo: "/icons/integrations/claude.svg", href: "/integrations/frameworks/anthropic" },
    { name: "Google ADK", logo: "/icons/integrations/google-adk.png", href: "/integrations/frameworks/google-adk" },
    { name: "AgentCore", logo: "/icons/integrations/agentcore.svg", href: "/integrations/frameworks/agentcore" },
    { name: "Strands", logo: "/icons/integrations/strands.svg", href: "/integrations/frameworks/strands" },
    { name: "Vercel AI SDK", logo: "/icons/integrations/ai-sdk.svg", href: INTEGRATIONS_INDEX, inline: VercelAISDKMark },
    { name: "OpenTelemetry", logo: "/icons/integrations/otel.svg", href: INTEGRATIONS_INDEX },
  ],
};

const CI_CD: Category = {
  label: "CI / CD",
  columns: 3,
  items: [
    { name: "GitHub Actions", logo: "/icons/integrations/github.svg", href: CI_CD_DOCS, inline: GitHubMark },
    { name: "GitLab CI", logo: "/icons/integrations/gitlab.svg", href: CI_CD_DOCS },
    { name: "Jenkins", logo: "/icons/integrations/jenkins.svg", href: CI_CD_DOCS },
    { name: "CircleCI", logo: "/icons/integrations/circleci.svg", href: CI_CD_DOCS, inline: CircleCIMark },
    { name: "Buildkite", logo: "/icons/integrations/buildkite.svg", href: CI_CD_DOCS },
    { name: "Azure Pipelines", logo: "/icons/integrations/azure-pipelines.svg", href: CI_CD_DOCS },
  ],
};

const IntegrationTile: React.FC<{ item: Integration }> = ({ item }: { item: Integration }) => {
  const Inline = item.inline;
  return (
    <Link
      href={item.href}
      className={styles.tile}
      aria-label={`${item.name} integration docs`}
    >
      <div
        className={`${styles.logoWrap}${Inline ? ` ${styles.logoWrapInline}` : ""}`}
      >
        {Inline ? (
          <Inline className={styles.logoInline} aria-label={`${item.name} logo`} />
        ) : (
          <Image
            src={item.logo}
            alt={`${item.name} logo`}
            width={32}
            height={32}
            className={styles.logo}
          />
        )}
      </div>
      <span className={styles.tileName}>{item.name}</span>
    </Link>
  );
};

const Panel: React.FC<{
  category: Category;
  className?: string;
}> = ({
  category,
  className,
}: {
  category: Category;
  className?: string;
}) => {
  return (
    <section
      className={`${styles.panel}${className ? ` ${className}` : ""}`}
      aria-labelledby={`integration-${category.label}`}
    >
      <header className={styles.panelHeader}>
        <span id={`integration-${category.label}`} className={styles.panelLabel}>
          {category.label}
        </span>
      </header>
      <div
        className={styles.tiles}
        style={{ ["--tile-cols" as string]: category.columns }}
      >
        {category.items.map((item) => (
          <IntegrationTile key={item.name} item={item} />
        ))}
      </div>
    </section>
  );
};

const IntegrationGrid: React.FC = () => {
  return (
    <div className={styles.grid}>
      <Panel category={FRAMEWORKS} className={styles.tall} />
      <Panel category={MODEL_PROVIDERS} className={styles.top} />
      <Panel category={CI_CD} className={styles.bottom} />
    </div>
  );
};


export default IntegrationGrid;
