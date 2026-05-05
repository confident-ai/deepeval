import fs from "node:fs";
import path from "node:path";
import type { LogoProps } from "./types";
import styles from "./CompanyLogos.module.scss";

const SVG_PATH = path.join(
  process.cwd(),
  "public",
  "icons",
  "companies",
  "benz.svg"
);

const raw = fs.readFileSync(SVG_PATH, "utf8");

const processed = raw.replaceAll("fill:#131822", "fill:var(--benz-wordmark)");

const innerMatch = processed.match(/<svg[^>]*>([\s\S]*)<\/svg>/);
const inner = innerMatch?.[1] ?? "";

const viewBoxMatch = processed.match(/viewBox="([^"]+)"/);
const widthMatch = processed.match(/\swidth="([\d.]+)"/);
const heightMatch = processed.match(/\sheight="([\d.]+)"/);
const viewBox =
  viewBoxMatch?.[1] ??
  (widthMatch && heightMatch
    ? `0 0 ${widthMatch[1]} ${heightMatch[1]}`
    : undefined);

const Benz: React.FC<LogoProps> = ({ className, ...rest }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox={viewBox}
    className={[styles.benzRoot, className].filter(Boolean).join(" ")}
    dangerouslySetInnerHTML={{ __html: inner }}
    {...rest}
  />
);

export default Benz;
