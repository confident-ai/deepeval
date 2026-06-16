import React, { ReactNode } from "react";
import { ChevronRight } from "lucide-react";
import SchemaInjector from "../SchemaInjector/SchemaInjector";
import { buildFAQPageSchema } from "@/src/utils/schema-helpers";

export interface QA {
  question: string;
  answer: ReactNode;
}

interface FAQsProps {
  qas: QA[];
}

/**
 * Walks a ReactNode tree and concatenates its visible text. Used to
 * flatten rich MDX answers into a plain string for the FAQPage JSON-LD,
 * which expects `text` to be a single string per crawler spec.
 */
function extractText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (React.isValidElement(node)) {
    return extractText((node.props as { children?: ReactNode }).children);
  }
  return "";
}

/**
 * Mirrors the typography plugin's `.prose code` chip (border, muted
 * background, rounded corners, normal weight). The accordion header is
 * marked `not-prose`, so prose code styling never reaches the title —
 * we replicate it here, sized relative to the trigger text.
 */
const questionCodeStyle: React.CSSProperties = {
  padding: "0px 4px",
  border: "solid 1px var(--color-fd-border)",
  borderRadius: "4px",
  fontSize: "0.8em",
  fontWeight: 400,
  background: "var(--color-fd-muted)",
  color: "var(--color-fd-foreground)",
};

/**
 * Renders a question authored with Markdown-style `backticks` as inline
 * <code> spans for the accordion title, leaving the rest as plain text.
 * Authors get code formatting in questions without hand-writing JSX.
 */
function renderQuestion(question: string): ReactNode {
  const segments = question.split(/(`[^`]+`)/g);
  return segments.map((segment, index) =>
    segment.length > 1 && segment.startsWith("`") && segment.endsWith("`") ? (
      <code key={index} style={questionCodeStyle}>
        {segment.slice(1, -1)}
      </code>
    ) : (
      segment
    )
  );
}

/**
 * Strips backticks so the question can be used as a plain string for the
 * `<details>` key and the FAQPage JSON-LD `name`, both of which must be
 * plain text rather than rich nodes.
 */
function plainQuestion(question: string): string {
  return question.replace(/`/g, "");
}

/**
 * Derives a stable, page-safe group name for the `<details name>` so a
 * single FAQ block behaves like a single-open accordion (the browser
 * auto-closes siblings sharing a name). Keyed off the first question so
 * two separate FAQ blocks on the same page don't fight over open state.
 */
function groupName(qas: QA[]): string {
  const seed = plainQuestion(qas[0]?.question ?? "faq");
  return `faq-${seed.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48)}`;
}

/**
 * FAQ list rendered with native `<details>/<summary>` so every answer
 * ships inside the server-rendered HTML body (crawlable + visible with
 * JavaScript disabled — search People-Also-Ask and LLM crawlers read
 * the text without executing JS), unlike a Radix accordion which only
 * mounts answer content client-side on expand.
 *
 * `name` on each `<details>` gives native single-open accordion behavior
 * with zero JS; older browsers without `name` support simply allow more
 * than one panel open at a time (graceful degradation). The classes
 * mirror Fumadocs' accordion so the UI stays visually identical.
 *
 * A schema.org FAQPage JSON-LD block is still emitted alongside for rich
 * results; the schema emission stays inside this wrapper so callers don't
 * have to remember to pair the two manually.
 */
export const FAQs: React.FC<FAQsProps> = ({ qas }) => {
  const schema = buildFAQPageSchema(
    qas.map(({ question, answer }) => ({
      question: plainQuestion(question),
      answer: extractText(answer).replace(/\s+/g, " ").trim(),
    }))
  );

  const name = groupName(qas);

  return (
    <>
      <SchemaInjector schema={schema} />
      <div className="divide-y divide-fd-border overflow-hidden rounded-lg border bg-fd-card">
        {qas.map(({ question, answer }) => {
          const plain = plainQuestion(question);
          return (
            <details
              key={plain}
              name={name}
              className="group scroll-m-24 not-prose"
            >
              <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-start font-medium text-fd-card-foreground focus-visible:bg-fd-accent focus-visible:outline-none [&::-webkit-details-marker]:hidden">
                <ChevronRight className="size-4 shrink-0 text-fd-muted-foreground transition-transform duration-200 group-open:rotate-90" />
                {renderQuestion(question)}
              </summary>
              <div className="px-4 pb-2 text-[0.9375rem] prose-no-margin">
                {answer}
              </div>
            </details>
          );
        })}
      </div>
    </>
  );
};

export default FAQs;
