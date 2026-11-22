import type { Metadata } from "next";
import { DocsBody } from "fumadocs-ui/layouts/notebook/page";
import { getMDXComponents } from "@/components/mdx";
import CommunityReadMe from "@/community/read-me.mdx";
import HomePageShell from "@/src/layouts/HomePageShell";
import HomeLayout from "@/src/layouts/HomeLayout";
import CommunityHeroSection from "@/src/sections/community/CommunityHeroSection";

// Rebuild the page each hour, so an event moves from "Upcoming" to "Past"
// within an hour, without a new deploy.
export const revalidate = 3600;

export const metadata: Metadata = {
  title: "Community",
  description:
    "Find a DeepEval meetup near you, or host one in your city. Engineers get together to show how they test and debug their AI agents.",
  alternates: { canonical: "/community" },
};

export default function CommunityPage() {
  return (
    <HomePageShell>
      <HomeLayout
        leftContent={<CommunityHeroSection />}
        rightContent={
          <div className="docs-page-surface">
            <DocsBody>
              <CommunityReadMe components={getMDXComponents()} />
            </DocsBody>
          </div>
        }
      />
    </HomePageShell>
  );
}
