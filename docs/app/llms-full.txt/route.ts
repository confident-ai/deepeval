import { getLLMText, source } from '@/lib/source';
import { LLMS_ATTRIBUTION } from '@/lib/llms-attribution';

export const revalidate = false;

export async function GET() {
  const scan = source.getPages().map(getLLMText);
  const scanned = await Promise.all(scan);

  return new Response([`> ${LLMS_ATTRIBUTION}`, ...scanned].join('\n\n'));
}
