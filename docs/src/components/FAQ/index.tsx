import React, { ReactNode } from "react";
import { Accordion, Accordions } from "fumadocs-ui/components/accordion";
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
 * accordion `value` (deep-link/open state) and the FAQPage JSON-LD `name`,
 * both of which must be plain text rather than rich nodes.
 */
function plainQuestion(question: string): string {
  return question.replace(/`/g, "");
}

/**
 * Accordion-style FAQ list that also emits a schema.org FAQPage JSON-LD
 * block. The UI is delegated to Fumadocs' `Accordions` component so we
 * inherit Radix-powered a11y, keyboard nav, and deep-link support for
 * free. The schema emission stays inside this wrapper so callers don't
 * have to remember to pair the two manually.
 */
export const FAQs: React.FC<FAQsProps> = ({ qas }) => {
  const schema = buildFAQPageSchema(
    qas.map(({ question, answer }) => ({
      question: plainQuestion(question),
      answer: extractText(answer).replace(/\s+/g, " ").trim(),
    }))
  );

  return (
    <>
      <SchemaInjector schema={schema} />
      <Accordions type="single">
        {qas.map(({ question, answer }) => {
          const plain = plainQuestion(question);
          return (
            <Accordion
              key={plain}
              title={renderQuestion(question)}
              value={plain}
            >
              {answer}
            </Accordion>
          );
        })}
      </Accordions>
    </>
  );
};

export default FAQs;
