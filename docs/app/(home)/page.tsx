import type { Metadata } from "next";
import { DocsBody } from "fumadocs-ui/layouts/notebook/page";
import { getMDXComponents } from "@/components/mdx";
import ReadMe from "@/home/read-me.mdx";
import HomeLayout from "@/src/layouts/HomeLayout";
import HomeHeroSection from "@/src/sections/home/HomeHeroSection";
import { siteTitle } from "@/lib/shared";

// Homepage sets `title.absolute` so the root layout's `%s | …` template
// doesn't double up the site name. The tagline here mirrors the old
// Docusaurus `tagline` ("Evaluation Framework for LLMs") expanded into
// a proper meta-description sentence.
export const metadata: Metadata = {
  title: { absolute: siteTitle },
  description:
    "DeepEval is the open-source LLM evaluation framework for testing and benchmarking LLM applications — 50+ plug-and-play metrics for AI agents, RAG, chatbots, and more.",
  alternates: { canonical: "/" },
};

export default function HomePage() {
  return (
    <HomeLayout
      leftContent={<HomeHeroSection />}
      rightContent={
        <div className="docs-page-surface">
          <DocsBody>
            <ReadMe components={getMDXComponents()} />
          </DocsBody>
        </div>
      }
    />
  );
}
