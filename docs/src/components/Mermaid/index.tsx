"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useTheme } from "next-themes";

type MermaidRenderResult = {
  svg: string;
  bindFunctions?: (element: Element) => void;
};

type MermaidProps = {
  chart: string;
};

const Mermaid: React.FC<MermaidProps> = ({ chart }) => {
  const id = useId().replace(/:/g, "");
  const containerRef = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();
  const [result, setResult] = useState<MermaidRenderResult | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function renderChart() {
      const mermaid = (await import("mermaid")).default;

      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        fontFamily: "inherit",
        theme: resolvedTheme === "dark" ? "dark" : "default",
      });

      const renderResult = await mermaid.render(
        `mermaid-${id}-${resolvedTheme ?? "light"}`,
        chart.replaceAll("\\n", "\n")
      );

      if (!cancelled) {
        setResult({
          svg: renderResult.svg,
          bindFunctions: renderResult.bindFunctions,
        });
      }
    }

    void renderChart();

    return () => {
      cancelled = true;
    };
  }, [chart, id, resolvedTheme]);

  useEffect(() => {
    if (!containerRef.current) return;

    containerRef.current.style.width = "100%";
    containerRef.current.style.maxHeight = "60vh";
    containerRef.current.style.overflow = "auto";

    const svg = containerRef.current.querySelector("svg");
    if (svg instanceof SVGSVGElement) {
      svg.style.display = "block";
      svg.style.maxWidth = "100%";
      svg.style.maxHeight = "60vh";
      svg.style.width = "auto";
      svg.style.height = "auto";
      svg.style.margin = "0 auto";
    }

    if (!result?.bindFunctions) return;
    result.bindFunctions(containerRef.current);
  }, [result]);

  if (!result) return null;

  return (
    <div
      ref={containerRef}
      dangerouslySetInnerHTML={{ __html: result.svg }}
    />
  );
};


export default Mermaid;
