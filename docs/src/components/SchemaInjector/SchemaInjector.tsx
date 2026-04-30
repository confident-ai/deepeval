interface SchemaInjectorProps {
  // Accepts any object — the schema-builder helpers in
  // `src/utils/schema-helpers.ts` return `object` (not
  // `Record<string, unknown>`), and widening the prop type here avoids
  // forcing every call site to cast.
  schema?: object | null;
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
const SchemaInjector: React.FC<SchemaInjectorProps> = ({ schema }) => {
  if (!schema) return null;
  const json = JSON.stringify(schema).replace(/</g, "\\u003c");
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: json }}
    />
  );
};


export default SchemaInjector;
