import type { Metadata } from "next";
import { DocsBody } from "fumadocs-ui/layouts/notebook/page";
import { getMDXComponents } from "@/components/mdx";
import EnterpriseReadMe from "@/enterprise/read-me.mdx";
import HomePageShell from "@/src/layouts/HomePageShell";
import HomeLayout from "@/src/layouts/HomeLayout";
import EnterpriseHeroSection from "@/src/sections/enterprise/EnterpriseHeroSection";

export const metadata: Metadata = {
  title: "Enterprise",
  description:
    "Scale DeepEval with enterprise observability, shared workflows, and production-grade LLM evaluation on Confident AI.",
  alternates: { canonical: "/enterprise" },
};

export default function EnterprisePage() {
  return (
    <HomePageShell>
      <HomeLayout
        leftContent={<EnterpriseHeroSection />}
        rightContent={
          <div className="docs-page-surface">
            <DocsBody>
              <EnterpriseReadMe components={getMDXComponents()} />
            </DocsBody>
          </div>
        }
      />
    </HomePageShell>
  );
}
