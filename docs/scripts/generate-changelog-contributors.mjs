import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const CHANGELOG_DIR = path.join(ROOT, "content", "changelog");
const OUT_PATH = path.join(
  ROOT,
  "lib",
  "generated",
  "changelog-contributors.json",
);

const yearFileRe = /^changelog-(\d{4})\.mdx$/;
const githubProfileRe = /\[([^\]]+)\]\(https:\/\/github\.com\/([^)\/]+)\)/g;

function collectYearContributors(filePath) {
  const text = fs.readFileSync(filePath, "utf8");
  const byLogin = new Map();
  let match;

  while ((match = githubProfileRe.exec(text)) !== null) {
    const [, name, login] = match;
    if (login === "confident-ai") continue;

    const current = byLogin.get(login) ?? {
      login,
      name,
      url: `https://github.com/${login}`,
      avatarUrl: `https://github.com/${login}.png?size=64`,
      contributions: 0,
    };

    current.contributions += 1;
    byLogin.set(login, current);
  }

  return Array.from(byLogin.values()).sort((a, b) => {
    if (b.contributions !== a.contributions) {
      return b.contributions - a.contributions;
    }
    return a.login.localeCompare(b.login);
  });
}

function main() {
  const manifest = {};
  const files = fs.readdirSync(CHANGELOG_DIR);

  for (const file of files) {
    const match = yearFileRe.exec(file);
    if (!match) continue;

    const year = match[1];
    manifest[year] = collectYearContributors(
      path.join(CHANGELOG_DIR, file),
    );
  }

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(
    `[changelog-contributors] wrote ${Object.keys(manifest).length} year(s)`,
  );
}

main();
