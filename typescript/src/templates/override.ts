/**
 * User-supplied prompt overrides (`evaluationTemplate`).
 *
 * Prompt text lives in the Jinja bundle shared with the Python SDK, where both
 * template method names and template variables are snake_case. This module is
 * the boundary that turns that into the camelCase surface the rest of the TS SDK
 * uses, so an override reads like ordinary TypeScript:
 *
 * ```ts
 * new AnswerRelevancyMetric({
 *   evaluationTemplate: {
 *     generateStatements: ({ actualOutput }) => `...${actualOutput}...`,
 *   },
 * })
 * ```
 *
 * The method names are derived from the bundle's own keys, so a metric added
 * later gets a correctly typed override surface with no changes here.
 */

import metricsBundle from "@/templates/metrics/templates.json";

/** `generate_statements` -> `generateStatements` (type level). */
export type SnakeToCamel<S extends string> =
  S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<SnakeToCamel<Tail>>}`
    : S;

type MetricsBundle = typeof metricsBundle;

/** Bundle keys that own templates, e.g. `"AnswerRelevancyMetric"`. */
export type TemplateClassName = keyof MetricsBundle;

/** The camelCase template method names available for one bundle key. */
export type TemplateMethodsOf<K extends TemplateClassName> = SnakeToCamel<
  Extract<keyof MetricsBundle[K], string>
>;

/** The render context handed to an override. Keys are camelCase. */
export type TemplateVars = Record<string, unknown> & { multimodal: boolean };

/**
 * One prompt override.
 *
 * `renderDefault` renders the template DeepEval ships, so a prompt can be
 * extended rather than replaced:
 * `(vars, renderDefault) => renderDefault(vars) + "\n\nExtra rule."`
 */
export type TemplateFn<V extends TemplateVars = TemplateVars> = (
  vars: V,
  renderDefault: (vars?: V) => string,
) => string;

/**
 * A partial set of prompt overrides for a metric. Unlisted methods keep
 * rendering from the shared bundle.
 */
export type MetricTemplateOverride<K extends TemplateClassName> = Partial<
  Record<TemplateMethodsOf<K>, TemplateFn>
>;

/** `generate_statements` -> `generateStatements` (value level). */
export function snakeToCamel(name: string): string {
  return name.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());
}

/**
 * Re-key a render context for an override.
 *
 * Leading underscores are dropped (`_additional_context` -> `additionalContext`)
 * since they mark template-internal variables, not something a caller should
 * have to spell.
 */
export function camelizeVars(
  vars: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(vars)) {
    out[snakeToCamel(key.replace(/^_+/, ""))] = value;
  }
  return out;
}

/** Invert {@link camelizeVars} so `renderDefault` can reach the Jinja names. */
export function decamelizeVars(
  vars: Record<string, unknown>,
  originalKeys: string[],
): Record<string, unknown> {
  const bySnake = new Map(
    originalKeys.map((key) => [snakeToCamel(key.replace(/^_+/, "")), key]),
  );
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(vars)) {
    out[bySnake.get(key) ?? key] = value;
  }
  return out;
}

/**
 * Find the override for a snake_case bundle method, if the caller supplied one.
 * Kept lenient about the container's type: it arrives from user code.
 */
export function findOverride(
  template: unknown,
  method: string,
): TemplateFn | undefined {
  if (template == null || typeof template !== "object") return undefined;
  const fn = (template as Record<string, unknown>)[snakeToCamel(method)];
  return typeof fn === "function" ? (fn as TemplateFn) : undefined;
}
