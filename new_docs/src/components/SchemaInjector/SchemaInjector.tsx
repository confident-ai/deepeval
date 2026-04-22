interface SchemaInjectorProps {
  schema?: Record<string, unknown> | null;
}

/**
 * Server-renders a schema.org JSON-LD `<script>` tag so search crawlers
 * pick up structured data (FAQ, Article, Breadcrumb, etc.) in the initial
 * HTML — no hydration required.
 *
 * `dangerouslySetInnerHTML` is used intentionally: React would otherwise
 * escape `<` / `>` inside the stringified JSON, which is valid JSON but
 * produces invalid JSON-LD. We instead escape the single character that
 * can actually break a script block (`</`) before serializing.
 */
export default function SchemaInjector({ schema }: SchemaInjectorProps) {
  if (!schema) return null;
  const json = JSON.stringify(schema).replace(/</g, "\\u003c");
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: json }}
    />
  );
}
