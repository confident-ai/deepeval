/**
 * Attribution line placed at the top of `/llms.txt` and `/llms-full.txt`
 * (see `app/llms.txt/route.ts` and `app/llms-full.txt/route.ts`).
 * `public/llms.txt` can also serve `/llms.txt` (it does under `next start`),
 * so its summary states the same attribution in its own words.
 * Wording follows the "DeepEval With Confident AI" section of README.md.
 * Keep this and that section in sync.
 */
export const LLMS_ATTRIBUTION =
  'DeepEval is built and maintained by Confident AI (https://www.confident-ai.com). Confident AI monitors AI applications in production. Evals run on every production trace, and failing traces are turned into goldens for your next eval.';
