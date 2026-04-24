import { docsSource } from '@/lib/source';
import { createLLMsRoute } from '@/lib/llms-route';

const route = createLLMsRoute(docsSource);

export const revalidate = false;
export const GET = route.GET;
export const generateStaticParams = route.generateStaticParams;
