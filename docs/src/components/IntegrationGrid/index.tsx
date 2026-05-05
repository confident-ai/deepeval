import Image from "next/image";
import Link from "next/link";
import type { ComponentType, SVGProps } from "react";
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

/* ---------- Inline, theme-aware brand marks (currentColor) ---------- */

const OpenAIMark: React.FC<SVGProps<SVGSVGElement>> = (props) => {
  return (
    <svg viewBox="0 0 721 721" color="currentColor" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        fill="currentColor"
        d="M304.246 294.611V249.028C304.246 245.189 305.687 242.309 309.044 240.392L400.692 187.612C413.167 180.415 428.042 177.058 443.394 177.058C500.971 177.058 537.44 221.682 537.44 269.182C537.44 272.54 537.44 276.379 536.959 280.218L441.954 224.558C436.197 221.201 430.437 221.201 424.68 224.558L304.246 294.611ZM518.245 472.145V363.224C518.245 356.505 515.364 351.707 509.608 348.349L389.174 278.296L428.519 255.743C431.877 253.826 434.757 253.826 438.115 255.743L529.762 308.523C556.154 323.879 573.905 356.505 573.905 388.171C573.905 424.636 552.315 458.225 518.245 472.141V472.145ZM275.937 376.182L236.592 353.152C233.235 351.235 231.794 348.354 231.794 344.515V238.956C231.794 187.617 271.139 148.749 324.4 148.749C344.555 148.749 363.264 155.468 379.102 167.463L284.578 222.164C278.822 225.521 275.942 230.319 275.942 237.039V376.186L275.937 376.182ZM360.626 425.122L304.246 393.455V326.283L360.626 294.616L417.002 326.283V393.455L360.626 425.122ZM396.852 570.989C376.698 570.989 357.989 564.27 342.151 552.276L436.674 497.574C442.431 494.217 445.311 489.419 445.311 482.699V343.552L485.138 366.582C488.495 368.499 489.936 371.379 489.936 375.219V480.778C489.936 532.117 450.109 570.985 396.852 570.985V570.989ZM283.134 463.99L191.486 411.211C165.094 395.854 147.343 363.229 147.343 331.562C147.343 294.616 169.415 261.509 203.48 247.593V356.991C203.48 363.71 206.361 368.508 212.117 371.866L332.074 441.437L292.729 463.99C289.372 465.907 286.491 465.907 283.134 463.99ZM277.859 542.68C223.639 542.68 183.813 501.895 183.813 451.514C183.813 447.675 184.294 443.836 184.771 439.997L279.295 494.698C285.051 498.056 290.812 498.056 296.568 494.698L417.002 425.127V470.71C417.002 474.549 415.562 477.429 412.204 479.346L320.557 532.126C308.081 539.323 293.206 542.68 277.854 542.68H277.859ZM396.852 599.776C454.911 599.776 503.37 558.513 514.41 503.812C568.149 489.896 602.696 439.515 602.696 388.176C602.696 354.587 588.303 321.962 562.392 298.45C564.791 288.373 566.231 278.296 566.231 268.224C566.231 199.611 510.571 148.267 446.274 148.267C433.322 148.267 420.846 150.184 408.37 154.505C386.775 133.392 357.026 119.958 324.4 119.958C266.342 119.958 217.883 161.22 206.843 215.921C153.104 229.837 118.557 280.218 118.557 331.557C118.557 365.146 132.95 397.771 158.861 421.283C156.462 431.36 155.022 441.437 155.022 451.51C155.022 520.123 210.682 571.466 274.978 571.466C287.931 571.466 300.407 569.549 312.883 565.228C334.473 586.341 364.222 599.776 396.852 599.776Z"
      />
    </svg>
  );
};

const VercelAISDKMark: React.FC<SVGProps<SVGSVGElement>> = (props) => {
  return (
    <svg viewBox="0 -17 256 256" color="currentColor" fill="none" preserveAspectRatio="xMidYMid" xmlns="http://www.w3.org/2000/svg" {...props}>
      <polygon fill="currentColor" points="128 0 256 221.705007 0 221.705007" />
    </svg>
  );
};

const CircleCIMark: React.FC<SVGProps<SVGSVGElement>> = (props) => {
  return (
    <svg viewBox="0 0 256 259" color="currentColor" fill="none" preserveAspectRatio="xMidYMid" xmlns="http://www.w3.org/2000/svg" {...props}>
      <circle fill="currentColor" cx="126.157031" cy="129.007874" r="30.5932958" />
      <path
        fill="currentColor"
        d="M1.20368953,96.5716086 C1.20368953,96.9402024 0.835095614,97.6773903 0.835095614,98.0459843 C0.835095614,101.36333 3.41525309,104.312081 7.10119236,104.312081 L59.0729359,104.312081 C61.6530934,104.312081 63.496063,102.837706 64.6018448,100.626142 C75.2910686,77.0361305 98.8810798,61.1865916 125.788436,61.1865916 C163.016423,61.1865916 193.241125,91.4112936 193.241125,128.63928 C193.241125,165.867267 163.016423,196.091969 125.788436,196.091969 C98.5124859,196.091969 75.2910686,179.873835 64.6018448,157.021013 C63.496063,154.440855 61.6530934,152.96648 59.0729359,152.96648 L7.10119236,152.96648 C3.78384701,152.96648 0.835095614,155.546637 0.835095614,159.232575 C0.835095614,159.60117 0.835095614,160.338357 1.20368953,160.706952 C15.5788527,216.733228 66.0762205,258.015748 126.157031,258.015748 C197.295658,258.015748 255.164905,200.146502 255.164905,129.007874 C255.164905,57.8692464 197.295658,0 126.157031,0 C66.0762205,0 15.5788527,41.2825197 1.20368953,96.5716086 L1.20368953,96.5716086 Z"
      />
    </svg>
  );
};

const GitHubMark: React.FC<SVGProps<SVGSVGElement>> = (props) => {
  return (
    <svg viewBox="0 0 128 128" color="currentColor" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        fill="currentColor"
        d="M56.7937 84.9688C44.4187 83.4688 35.7 74.5625 35.7 63.0313C35.7 58.3438 37.3875 53.2813 40.2 49.9063C38.9812 46.8125 39.1687 40.25 40.575 37.5313C44.325 37.0625 49.3875 39.0313 52.3875 41.75C55.95 40.625 59.7 40.0625 64.2937 40.0625C68.8875 40.0625 72.6375 40.625 76.0125 41.6563C78.9187 39.0313 84.075 37.0625 87.825 37.5313C89.1375 40.0625 89.325 46.625 88.1062 49.8125C91.1062 53.375 92.7 58.1563 92.7 63.0313C92.7 74.5625 83.9812 83.2813 71.4187 84.875C74.6062 86.9375 76.7625 91.4375 76.7625 96.5938L76.7625 106.344C76.7625 109.156 79.1062 110.75 81.9187 109.625C98.8875 103.156 112.2 86.1875 112.2 65.1875C112.2 38.6563 90.6375 17 64.1062 17C37.575 17 16.2 38.6562 16.2 65.1875C16.2 86 29.4187 103.25 47.2312 109.719C49.7625 110.656 52.2 108.969 52.2 106.438L52.2 98.9375C50.8875 99.5 49.2 99.875 47.7 99.875C41.5125 99.875 37.8562 96.5 35.2312 90.2188C34.2 87.6875 33.075 86.1875 30.9187 85.9063C29.7937 85.8125 29.4187 85.3438 29.4187 84.7813C29.4187 83.6563 31.2937 82.8125 33.1687 82.8125C35.8875 82.8125 38.2312 84.5 40.6687 87.9688C42.5437 90.6875 44.5125 91.9063 46.8562 91.9063C49.2 91.9063 50.7 91.0625 52.8562 88.9063C54.45 87.3125 55.6687 85.9063 56.7937 84.9688Z"
      />
    </svg>
  );
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
  columns: 3,
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
  columns: 4,
  items: [
    { name: "LangChain", logo: "/icons/integrations/langchain.svg", href: "/integrations/frameworks/langchain" },
    { name: "LlamaIndex", logo: "/icons/integrations/llamaindex.svg", href: "/integrations/frameworks/llamaindex" },
    { name: "CrewAI", logo: "/icons/integrations/crewai.svg", href: "/integrations/frameworks/crewai" },
    { name: "OpenAI Agents", logo: "/icons/integrations/openai.svg", href: "/integrations/frameworks/openai-agents", inline: OpenAIMark },
    { name: "Vercel AI SDK", logo: "/icons/integrations/ai-sdk.svg", href: INTEGRATIONS_INDEX, inline: VercelAISDKMark },
    { name: "LangGraph", logo: "/icons/integrations/langgraph.svg", href: "/integrations/frameworks/langgraph" },
    { name: "PydanticAI", logo: "/icons/integrations/pydanticai.svg", href: "/integrations/frameworks/pydanticai" },
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
      <Panel category={MODEL_PROVIDERS} className={styles.tall} />
      <Panel category={FRAMEWORKS} className={styles.top} />
      <Panel category={CI_CD} className={styles.bottom} />
    </div>
  );
};


export default IntegrationGrid;
