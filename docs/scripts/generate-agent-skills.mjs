import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Publishes the repo's agent skills (`<repo root>/skills/`) at
 * `/.well-known/agent-skills/` per the Agent Skills Discovery RFC
 * v0.2.0 (https://github.com/cloudflare/agent-skills-discovery-rfc):
 * copies each skill folder into `public/` and writes an `index.json`
 * whose sha256 digests are computed from the copied SKILL.md files, so
 * the digests can never drift from the served artifacts.
 *
 * Runs as part of `prebuild`. If the build environment doesn't include
 * files outside `docs/` (Vercel root-directory isolation), it leaves
 * the committed copies in `public/` untouched instead of failing.
 */
const docsDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const skillsDir = join(docsDir, '..', 'skills');
const outDir = join(docsDir, 'public', '.well-known', 'agent-skills');
const siteUrl = 'https://deepeval.com';

if (!existsSync(skillsDir)) {
  console.log('generate-agent-skills: ../skills not found, keeping committed copies');
  process.exit(0);
}

function frontmatterField(markdown, field) {
  const fm = markdown.match(/^---\n([\s\S]*?)\n---/)?.[1];
  if (!fm) return undefined;

  const folded = fm.match(new RegExp(`^${field}: *[>|][+-]?\\n((?:[ \\t]+.*\\n?)+)`, 'm'));
  if (folded) return folded[1].replace(/\s+/g, ' ').trim();

  const inline = fm.match(new RegExp(`^${field}: *(.+)$`, 'm'));
  return inline?.[1].trim().replace(/^["']|["']$/g, '');
}

const skills = [];

for (const entry of readdirSync(skillsDir, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const skillFile = join(skillsDir, entry.name, 'SKILL.md');
  if (!existsSync(skillFile)) continue;

  const markdown = readFileSync(skillFile);
  const name = frontmatterField(markdown.toString(), 'name') ?? entry.name;
  const description = frontmatterField(markdown.toString(), 'description') ?? '';

  cpSync(join(skillsDir, entry.name), join(outDir, name), { recursive: true });

  skills.push({
    name,
    type: 'skill-md',
    description,
    url: `${siteUrl}/.well-known/agent-skills/${name}/SKILL.md`,
    digest: `sha256:${createHash('sha256').update(markdown).digest('hex')}`,
  });
}

skills.sort((a, b) => a.name.localeCompare(b.name));

mkdirSync(outDir, { recursive: true });
for (const stale of readdirSync(outDir, { withFileTypes: true })) {
  if (stale.isDirectory() && !skills.some((s) => s.name === stale.name)) {
    rmSync(join(outDir, stale.name), { recursive: true });
  }
}

writeFileSync(
  join(outDir, 'index.json'),
  JSON.stringify(
    {
      $schema: 'https://schemas.agentskills.io/discovery/0.2.0/schema.json',
      skills,
    },
    null,
    2,
  ) + '\n',
);

console.log(`generate-agent-skills: published ${skills.length} skills`);
