import type { LogoProps } from "./types";
import { loadInlineSvg } from "./_loadInlineSvg";

/**
 * Builds a React component for a brand whose SVG should be inlined into the
 * SSR HTML at build time. The returned component is structurally identical
 * to the hand-coded brand marks (Uber, AWS, …): a real `<svg>` element in
 * the tree, no extra HTTP request, no client-side hydration cost beyond
 * what every other inline brand mark already pays.
 */
export function createInlineLogo(slug: string): React.FC<LogoProps> {
  const { inner, viewBox, rootFill, rootStroke } = loadInlineSvg(slug);

  const Component: React.FC<LogoProps> = (props) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox={viewBox}
      fill={rootFill}
      stroke={rootStroke}
      dangerouslySetInnerHTML={{ __html: inner }}
      {...props}
    />
  );
  Component.displayName = `InlineLogo(${slug})`;
  return Component;
}
