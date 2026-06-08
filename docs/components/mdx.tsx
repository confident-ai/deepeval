import defaultMdxComponents from "fumadocs-ui/mdx";
import { Tabs, Tab } from "fumadocs-ui/components/tabs";
import { Card, Cards } from "fumadocs-ui/components/card";
import { Steps, Step } from "fumadocs-ui/components/steps";
import type { MDXComponents } from "mdx/types";
import { DEFAULT_LLM_MODEL } from "@/lib/defaults";
import { MdxAnchor } from "@/components/mdx-anchor";
import { C } from "@/components/lang/code-term";

// Site-specific MDX components — globally registered so MDX authors
// don't have to `import` them in every file.
import VideoDisplayer from "@site/src/components/VideoDisplayer";
import ImageDisplayer from "@site/src/components/ImageDisplayer";
import Callout from "@site/src/components/Callout";
import Equation from "@site/src/components/Equation";
import Mermaid from "@site/src/components/Mermaid";
import MetricTagsDisplayer from "@site/src/components/MetricTagsDisplayer";
import IntegrationTagsDisplayer from "@site/src/components/IntegrationTagsDisplayer";
import AgentTraceTerminal from "@site/src/components/AgentTraceTerminal";
import FeatureComparisonTable from "@site/src/components/FeatureComparisonTable";
import LinkCards from "@site/src/components/LinkCards";
import TechStackCards from "@site/src/components/TechStackCards";
import { FAQs } from "@site/src/components/FAQ";
import BlogPostMeta from "@site/src/components/BlogPostMeta";
import ChangelogContributors from "@site/src/components/ChangelogContributors";
import SectionLabel from "@site/src/components/SectionLabel";
import EnterpriseComparisonTable from "@site/src/sections/enterprise/EnterpriseComparisonTable";
import EnterprisePlatformMockup from "@site/src/sections/enterprise/EnterprisePlatformMockup";
import RepoContributors from "@site/src/sections/home/RepoContributors";

function DefaultLLMModel() {
  return <code>{DEFAULT_LLM_MODEL}</code>;
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    a: MdxAnchor,
    // Fumadocs primitives
    Tabs,
    Tab,
    Card,
    Cards,
    Steps,
    Step,
    // Site components
    VideoDisplayer,
    ImageDisplayer,
    Callout,
    Equation,
    Mermaid,
    MetricTagsDisplayer,
    IntegrationTagsDisplayer,
    AgentTraceTerminal,
    FeatureComparisonTable,
    LinkCards,
    TechStackCards,
    FAQs,
    BlogPostMeta,
    ChangelogContributors,
    SectionLabel,
    EnterpriseComparisonTable,
    EnterprisePlatformMockup,
    RepoContributors,
    DefaultLLMModel,
    // Inline language-aware code term, e.g. <C id="class::LLMTestCase"/>
    C,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
