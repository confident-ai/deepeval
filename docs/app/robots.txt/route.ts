import { siteUrl } from '@/lib/shared';

// Hand-rolled instead of Next's `robots.ts` metadata file because that
// convention can't emit `Content-Signal` directives
// (https://contentsignals.org/). The signal declares we WANT AI
// systems to search, retrieve, and train on this content — the same
// posture as publishing llms.txt. Everything else matches the old
// generated output (allow all, advertise the sitemap).
export const revalidate = false;

export function GET() {
  const body = [
    'User-Agent: *',
    'Allow: /',
    'Content-Signal: search=yes, ai-input=yes, ai-train=yes',
    '',
    `Host: ${siteUrl}`,
    `Sitemap: ${siteUrl}/sitemap.xml`,
    '',
  ].join('\n');

  return new Response(body, {
    headers: { 'Content-Type': 'text/plain' },
  });
}
