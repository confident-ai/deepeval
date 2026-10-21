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
    "Standardize AI quality across your organization with Confident AI: one quality standard for every team, AI governance as release gates, and evals on every production trace.",
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
