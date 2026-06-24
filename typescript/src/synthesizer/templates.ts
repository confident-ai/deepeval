import nunjucks from "nunjucks";

const env = new nunjucks.Environment(null, {
  autoescape: false,
  throwOnUndefined: true,
});

function compile(text: string): nunjucks.Template {
  const trimmed = text.endsWith("\n") ? text.slice(0, -1) : text;
  return nunjucks.compile(trimmed, env);
}

export abstract class EvolutionTemplate {
  protected static baseInstruction = `I want you to act as an input rewriter.
Your objective is to rewrite a given \`Input\` and must be factually correct according to the supporting information in \`Context\`.
You MUST complicate the given \`Input\` using the following method:`;

  static reasoning_evolution(input: string, context: string | null): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

{% if context %}
Context:
{{ context }}
{% endif %}

Rewritten input (under 15 words, no "based on the context" phrasing):
`).render({
      _base: this.baseInstruction,
      _method: "Rewrite the input to explicitly require multiple-step reasoning. The evolved input should demand the model to reason through a problem before arriving at an answer, rather than simply recalling information.",
      input,
      context: context ?? "",
    });
  }

  static multi_context_evolution(input: string, context: string): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

Context:
{{ context }}

Rewritten input (under 15 words, no "based on the context" phrasing):
`).render({
      _base: this.baseInstruction,
      _method: "The input MUST be rewritten to require the reader to use information from ALL elements of the provided Context. The evolved input should implicitly demand cross-referencing multiple pieces of information.",
      input,
      context,
    });
  }

  static concretizing_evolution(input: string, context: string | null): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

{% if context %}
Context:
{{ context }}
{% endif %}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Replace general concepts, entities, or inquiries with more specific, concrete ones. The evolved input should reference real-world specifics instead of abstract categories.",
      input,
      context: context ?? "",
    });
  }

  static constrained_evolution(input: string, context: string | null): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

{% if context %}
Context:
{{ context }}
{% endif %}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Add at least one additional constraint or requirement to the input. This could be a format restriction, length limit, stylistic requirement, or specific condition that must be met.",
      input,
      context: context ?? "",
    });
  }

  static comparative_question_evolution(input: string, context: string | null): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

{% if context %}
Context:
{{ context }}
{% endif %}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Rewrite the input to require comparing or contrasting two or more entities, concepts, processes, or perspectives. The evolved input must demand an explicit comparative analysis.",
      input,
      context: context ?? "",
    });
  }

  static hypothetical_scenario_evolution(input: string, context: string | null): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

{% if context %}
Context:
{{ context }}
{% endif %}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Incorporate a hypothetical or speculative 'what if' scenario that is relevant to the input. The evolved input should explore a counterfactual situation.",
      input,
      context: context ?? "",
    });
  }

  static in_breadth_evolution(input: string, context: string | null): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

{% if context %}
Context:
{{ context }}
{% endif %}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Create a brand new input in the same general domain as the original, but exploring a rarer, more niche angle. The evolved input should be a different question entirely, not a modification of the original.",
      input,
      context: context ?? "",
    });
  }
}

export class PromptEvolutionTemplate {
  protected static baseInstruction = `I want you to act as an input rewriter.
Your objective is to rewrite a given \`Input\`.
You MUST complicate the given \`Input\` using the following method:`;

  static reasoning_evolution(input: string): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Rewrite the input to require multiple-step reasoning before arriving at an answer.",
      input,
    });
  }

  static concretizing_evolution(input: string): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Replace general concepts with more specific, concrete ones. Reference real-world specifics.",
      input,
    });
  }

  static constrained_evolution(input: string): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Add at least one additional constraint or requirement to the input.",
      input,
    });
  }

  static comparative_question_evolution(input: string): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Rewrite the input to require comparing or contrasting two or more entities, concepts, or processes.",
      input,
    });
  }

  static hypothetical_scenario_evolution(input: string): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Incorporate a hypothetical 'what if' scenario relevant to the input.",
      input,
    });
  }

  static in_breadth_evolution(input: string): string {
    return compile(`
{{ _base }}

{{ _method }}

Input:
{{ input }}

Rewritten input (under 15 words):
`).render({
      _base: this.baseInstruction,
      _method: "Create a brand new input in the same domain but a rarer, more niche angle. Entirely new question, not a modification.",
      input,
    });
  }
}

export class SynthesizerPromptTemplate {
  static generate_inputs(context: string, numInputs: number): string {
    return compile(`
Generate {{ num_inputs }} realistic user inputs that a user might ask about the following context.
Each input should be a standalone question or request that tests understanding of the content.

Context:
{{ context }}

Return a JSON array of objects with an "input" field each.
`).render({ num_inputs: numInputs, context });
  }

  static generate_expected_output(input: string, context: string): string {
    return compile(`
Given the following input and context, generate the correct expected output.

Input:
{{ input }}

Context:
{{ context }}

Return only the expected output text, no explanation.
`).render({ input, context });
  }

  static evaluate_input_quality(input: string): string {
    return compile(`
Evaluate the quality of the following synthetic input for testing an LLM.
Consider: realism, specificity, complexity, and how well it tests understanding.

Input:
"{{ input }}"

Return a JSON object with:
- "score": a number between 0.0 and 1.0
- "feedback": a brief explanation
`).render({ input });
  }

  static rewrite_input(input: string, feedback: string): string {
    return compile(`
Rewrite the following input to be more realistic, specific, and useful for testing an LLM.

Original input:
"{{ input }}"

Feedback on quality issues:
{{ feedback }}

Return only the rewritten input.
`).render({ input, feedback });
  }
}
