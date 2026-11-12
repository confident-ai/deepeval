import { source } from '@/lib/source';
import { llms } from 'fumadocs-core/source';
import { LLMS_ATTRIBUTION } from '@/lib/llms-attribution';

export const revalidate = false;

export function GET() {
  return new Response([`> ${LLMS_ATTRIBUTION}`, llms(source).index()].join('\n\n'));
}
