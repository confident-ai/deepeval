import { z } from "zod";
import { chartColor } from "@/lib/charts/schema";

// Data for SchemaFrame: objects with named fields, such as an LLM test case.
// Named schema-card.ts so it isn't mistaken for a zod schema module.

const nonEmpty = z.string().trim().min(1);
const objectId = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, "use lowercase-kebab-case object ids");
/** `"<object id>.<field name>"`. */
const fieldRef = z.string().regex(/^[a-z0-9-]+\.[^.\s]+$/, 'refer to a field as "<object id>.<field name>"');

const field = z.strictObject({
  name: nonEmpty,
  optional: z.boolean().default(false),
  example: nonEmpty.optional(),
  note: nonEmpty.optional(),
});

export const schemaCardData = z.strictObject({
  objects: z
    .array(
      z.strictObject({
        id: objectId,
        title: nonEmpty,
        subtitle: nonEmpty.optional(),
        color: chartColor.default("blue"),
        fields: z.array(field).min(1),
      })
    )
    .min(1),
  highlights: z
    .array(z.strictObject({ label: nonEmpty, color: chartColor, fields: z.array(fieldRef).min(1) }))
    .default([]),
  links: z.array(z.strictObject({ from: fieldRef, to: fieldRef })).default([]),
});

export type SchemaCardData = z.infer<typeof schemaCardData>;
export type SchemaObject = SchemaCardData["objects"][number];
export type SchemaField = SchemaObject["fields"][number];

export const SCHEMA_TYPES = ["card", "compare", "match"] as const;
export type SchemaType = (typeof SCHEMA_TYPES)[number];

export const fieldRefOf = (object: SchemaObject, field: SchemaField) => `${object.id}.${field.name}`;

export const schemaProblems = (data: SchemaCardData): string[] => {
  const problems: string[] = [];
  const ids = data.objects.map(o => o.id);
  if (new Set(ids).size !== ids.length) problems.push("object ids must be unique");
  const refs = new Set<string>();
  data.objects.forEach((o, i) => {
    const names = o.fields.map(f => f.name);
    if (new Set(names).size !== names.length) problems.push(`"objects.${i}" has duplicate field names`);
    for (const f of o.fields) refs.add(fieldRefOf(o, f));
  });
  const check = (ref: string, at: string) => {
    if (!refs.has(ref)) problems.push(`${at} names unknown field "${ref}"`);
  };
  data.highlights.forEach((h, i) => h.fields.forEach((ref, j) => check(ref, `"highlights.${i}.fields.${j}"`)));
  data.links.forEach((l, i) => {
    check(l.from, `"links.${i}.from"`);
    check(l.to, `"links.${i}.to"`);
  });
  return problems;
};

export const schemaTypeProblems = (type: SchemaType, data: SchemaCardData): string[] => {
  const n = data.objects.length;
  if (type === "card") return n === 1 ? [] : [`card shows exactly 1 object, this data has ${n}`];
  // Highlights are drawn as boxes with lines to the fields they read, around a single object.
  const highlights = data.highlights.length ? [`${type} doesn't draw highlights; use a card, or remove "highlights"`] : [];
  if (type === "compare") return [...(n >= 2 && n <= 4 ? [] : [`compare needs 2 to 4 objects, this data has ${n}`]), ...highlights];
  const problems = n === 2 ? [...highlights] : [`match needs exactly 2 objects, this data has ${n}`, ...highlights];
  if (!data.links.length) problems.push('match needs at least one entry in "links"');
  const [left, right] = data.objects;
  data.links.forEach((l, i) => {
    if (left && right && !(l.from.startsWith(`${left.id}.`) && l.to.startsWith(`${right.id}.`))) {
      problems.push(`"links.${i}" must go from "${left.id}" to "${right.id}"`);
    }
  });
  return problems;
};
