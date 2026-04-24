import { changelogSource } from '@/lib/source';
import { createLLMsRoute } from '@/lib/llms-route';

const route = createLLMsRoute(changelogSource);

export const revalidate = route.revalidate;
export const GET = route.GET;
export const generateStaticParams = route.generateStaticParams;
