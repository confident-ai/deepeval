import { setupScript } from '@/lib/wizard/setup-script';

export const revalidate = false;

export function GET() {
  return new Response(setupScript, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'public, max-age=3600, s-maxage=86400',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}
