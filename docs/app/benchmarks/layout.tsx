import type { ReactNode } from "react";
import { Geist_Mono } from "next/font/google";
import HomePageShell from "@/src/layouts/HomePageShell";
import Footer from "@/src/layouts/Footer";

// Used for every number on the benchmark pages, so it is loaded for this
// route group and not site-wide.
const numericFont = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-numeric",
  display: "swap",
});

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className={numericFont.variable} style={{ display: "contents" }}>
      <HomePageShell>
        {children}
        <Footer />
      </HomePageShell>
    </div>
  );
}
