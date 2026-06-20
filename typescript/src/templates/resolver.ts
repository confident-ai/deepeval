import nunjucks from "nunjucks";
// Per-feature bundles, mirroring deepeval/templates/<feature>/templates.json.
// scripts/compile_metric_templates.py emits these into BOTH packages.
import metricsBundle from "./metrics/templates.json";
const FRAGMENTS_KEY = "_fragments";

type TemplateBundle = Record<string, Record<string, string>>;
const BUNDLES: Record<string, TemplateBundle> = {
  metrics: metricsBundle as unknown as TemplateBundle,
};

function getBundle(feature: string): TemplateBundle {
  const b = BUNDLES[feature];
  if (!b) {
    throw new MetricTemplateNotFoundError(
      `No template bundle for feature '${feature}'. ` +
        `Available: ${Object.keys(BUNDLES).join(", ")}`,
    );
  }
  return b;
}

// --- Jinja2 parity helpers -------------------------------------------------
// Nunjucks and Jinja2 differ in two output details; we normalize the TS side to
// match Python so the SAME template renders byte-identically in both engines:
//   1. Arrays: Jinja prints a list as Python `repr` (`['a', 'b']`); Nunjucks
//      joins with commas (`a,b`). We give array values a Python-style toString
//      while leaving them real arrays (so `list[0]` / `{% for %}` still work).
//   2. Trailing newline: Jinja's default `keep_trailing_newline=False` drops one
//      trailing "\n"; Nunjucks keeps it. We strip one at compile time.

/** Python `repr()` of a value, as it appears INSIDE a list/dict (strings quoted). */
function pyRepr(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (typeof v === "string") {
    return (
      "'" +
      v
        .replace(/\\/g, "\\\\")
        .replace(/'/g, "\\'")
        .replace(/\n/g, "\\n")
        .replace(/\r/g, "\\r")
        .replace(/\t/g, "\\t") +
      "'"
    );
  }
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return String(v);
  if (Array.isArray(v)) return "[" + v.map(pyRepr).join(", ") + "]";
  if (typeof v === "object") {
    return (
      "{" +
      Object.entries(v as Record<string, unknown>)
        .map(([k, val]) => `${pyRepr(k)}: ${pyRepr(val)}`)
        .join(", ") +
      "}"
    );
  }
  return String(v);
}

/**
 * Recursively give arrays a Python-`repr` `toString` so `{{ list }}` matches
 * Jinja, without losing array behavior (indexing, iteration). Non-arrays pass
 * through unchanged so booleans/null stay usable in `{% if %}` and objects keep
 * property access (e.g. `_fragments.x`, `obj.attr`).
 */
function toJinjaParityValue(v: unknown): unknown {
  if (Array.isArray(v)) {
    const arr = v.map(toJinjaParityValue);
    Object.defineProperty(arr, "toString", {
      value: () => pyRepr(arr),
      enumerable: false,
      configurable: true,
    });
    return arr;
  }
  return v;
}

export class MetricTemplateNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MetricTemplateNotFoundError";
  }
}

export class MetricTemplateInterpolationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MetricTemplateInterpolationError";
  }
}

// --- cached environments + compiled templates (templates are static) ---
let _envStrict: nunjucks.Environment | null = null;
let _envLenient: nunjucks.Environment | null = null;
const _compiled = new Map<string, nunjucks.Template>();

function getEnv(strict: boolean): nunjucks.Environment {
  // autoescape MUST be off — these are LLM prompts, not HTML (matches the
  // Python jinja2.Environment default). `throwOnUndefined` mirrors StrictUndefined.
  if (strict) {
    if (_envStrict === null) {
      _envStrict = new nunjucks.Environment(null, {
        autoescape: false,
        throwOnUndefined: true,
      });
    }
    return _envStrict;
  }
  if (_envLenient === null) {
    _envLenient = new nunjucks.Environment(null, {
      autoescape: false,
      throwOnUndefined: false,
    });
  }
  return _envLenient;
}

/** Return the raw (un-rendered) template string for a class/method. */
export function getRawTemplate(
  feature: string,
  className: string,
  method: string,
): string {
  const bundle = getBundle(feature);
  const body = bundle[className]?.[method];
  if (typeof body !== "string" || body.length === 0) {
    throw new MetricTemplateNotFoundError(
      `No template found for '${feature}'.'${className}'.'${method}'. ` +
        `Available classes: ${Object.keys(bundle).join(", ")}`,
    );
  }
  return body;
}

function getCompiled(
  feature: string,
  className: string,
  method: string,
  strict: boolean,
): nunjucks.Template {
  const key = `${feature} ${className} ${method} ${strict}`;
  let template = _compiled.get(key);
  if (template === undefined) {
    // Mirror Jinja's default `keep_trailing_newline=False`: drop one trailing newline.
    let raw = getRawTemplate(feature, className, method);
    if (raw.endsWith("\n")) raw = raw.slice(0, -1);
    template = nunjucks.compile(raw, getEnv(strict));
    _compiled.set(key, template);
  }
  return template;
}

export interface ResolveTemplateOptions {
  /** Throw on undefined variables (mirrors Python's StrictUndefined). Default true. */
  strict?: boolean;
}

/**
 * Render a metric template to a final prompt.
 *
 * `multimodal` (default false) and `_fragments` (shared snippets) are always
 * available to the template; everything else is supplied via `vars`.
 *
 * @example resolveTemplate("metrics", "AnswerRelevancyMetric", "generate_verdicts", { input, statements })
 */
export function resolveTemplate(
  feature: string,
  className: string,
  method: string,
  vars: Record<string, unknown> = {},
  opts: ResolveTemplateOptions = {},
): string {
  const strict = opts.strict ?? true;
  const fragments = getBundle(feature)[FRAGMENTS_KEY] ?? {};
  const context: Record<string, unknown> = { multimodal: false };
  for (const [k, v] of Object.entries(vars)) {
    context[k] = toJinjaParityValue(v);
  }
  context._fragments = fragments;
  try {
    return getCompiled(feature, className, method, strict).render(context);
  } catch (e) {
    if (e instanceof MetricTemplateNotFoundError) throw e;
    throw new MetricTemplateInterpolationError(
      `Failed to render template '${feature}'.'${className}'.'${method}': ${(e as Error).message}`,
    );
  }
}

/** Clear cached environments + compiled templates (mainly for tests). */
export function clearMetricTemplateCache(): void {
  _envStrict = null;
  _envLenient = null;
  _compiled.clear();
}
