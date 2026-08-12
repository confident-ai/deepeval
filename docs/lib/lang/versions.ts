import versions from '../generated/sdk-versions.json';
import type { Language } from './languages';

/**
 * Read out of the SDK manifests by `scripts/generate-sdk-versions.mjs`; the
 * annotation is what fails the build when a new language has no version.
 */
export const SDK_VERSIONS: Record<Language, string> = versions;
