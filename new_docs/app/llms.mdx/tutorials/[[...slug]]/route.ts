import { tutorialsSource } from '@/lib/source';
import { createLLMsRoute } from '@/lib/llms-route';

const route = createLLMsRoute(tutorialsSource);

export const revalidate = route.revalidate;
export const GET = route.GET;
export const generateStaticParams = route.generateStaticParams;
