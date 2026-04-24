import { blogSource } from '@/lib/source';
import { createLLMsRoute } from '@/lib/llms-route';

const route = createLLMsRoute(blogSource);

export const revalidate = route.revalidate;
export const GET = route.GET;
export const generateStaticParams = route.generateStaticParams;
