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
 * Accordion-style FAQ list that also emits a schema.org FAQPage JSON-LD
 * block. The UI is delegated to Fumadocs' `Accordions` component so we
 * inherit Radix-powered a11y, keyboard nav, and deep-link support for
 * free. The schema emission stays inside this wrapper so callers don't
 * have to remember to pair the two manually.
 */
export const FAQs: React.FC<FAQsProps> = ({ qas }) => {
  const schema = buildFAQPageSchema(
    qas.map(({ question, answer }) => ({
      question,
      answer: extractText(answer).replace(/\s+/g, " ").trim(),
    })),
  );

  return (
    <>
      <SchemaInjector schema={schema} />
      <Accordions type="single">
        {qas.map(({ question, answer }) => (
          <Accordion key={question} title={question}>
            {answer}
          </Accordion>
        ))}
      </Accordions>
    </>
  );
};

export default FAQs;
