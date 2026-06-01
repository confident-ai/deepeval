#!/usr/bin/env node
/**
 * One-off generator for the README hero assets.
 *
 * Produces 4 SVGs under `assets/hero/`:
 *   - wordmark-light.svg / wordmark-dark.svg  (DeepEval icon + "DeepEval.")
 *   - tagline-light.svg  / tagline-dark.svg   ("The LLM Evaluation Framework")
 *
 * The README references them via <picture> + prefers-color-scheme so GitHub
 * swaps the right variant for each viewer's theme. Each SVG embeds CSS
 * keyframe animations (logo scale-in, letter stagger fade, tagline fade-up)
 * that GitHub's image proxy preserves for animated SVGs.
 *
 * Quicksand SemiBold is downloaded from the Google Fonts CDN and cached
 * locally under `docs/scripts/fonts/`. Glyphs are converted to outline
 * <path> elements via opentype.js so no font dependency exists at render
 * time — the SVG is pixel-identical in every renderer (GitHub camo,
 * Safari, etc.) regardless of font availability.
 *
 * Run:  yarn build-readme-hero  (from docs/)
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import opentype from 'opentype.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');
const OUT_DIR = path.join(REPO_ROOT, 'assets/hero');
const FONT_DIR = path.join(__dirname, 'fonts');
const FONT_PATH = path.join(FONT_DIR, 'Quicksand-SemiBold.ttf');

// google-webfonts-helper packages each Google Font weight as a static TTF
// (Google's own repo only ships the variable Quicksand[wght].ttf, whose
// default instance is Light/300 — opentype.js can't reliably interpolate
// that to SemiBold without bundling the full variable-axis math). The
// gwfh zip contains a single static `quicksand-v37-latin-600.ttf` with
// the SemiBold outlines we want.
const FONT_ZIP_URL =
  'https://gwfh.mranftl.com/api/fonts/quicksand?download=zip&subsets=latin&variants=600&formats=ttf';

const ACCENT = '#4400FF'; // DeepEval brand purple
const LIGHT_FG = '#0A0A0A';
const DARK_FG = '#FAFAFA';

async function ensureFont() {
  try {
    await fs.access(FONT_PATH);
    return;
  } catch {}
  await fs.mkdir(FONT_DIR, { recursive: true });
  console.log('Downloading Quicksand SemiBold…');
  const res = await fetch(FONT_ZIP_URL);
  if (!res.ok) throw new Error(`Font download failed: ${res.status}`);
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'quicksand-'));
  const zipPath = path.join(tmp, 'q.zip');
  await fs.writeFile(zipPath, Buffer.from(await res.arrayBuffer()));
  execFileSync('unzip', ['-q', '-o', zipPath, '-d', tmp]);
  const entries = await fs.readdir(tmp);
  const ttfName = entries.find((n) => n.endsWith('.ttf'));
  if (!ttfName) throw new Error('No TTF inside font zip');
  await fs.copyFile(path.join(tmp, ttfName), FONT_PATH);
  await fs.rm(tmp, { recursive: true, force: true });
}

/**
 * Build the wordmark SVG: just "DeepEval." with the period in the brand
 * purple. Each letter sits in its own <g> so we can stagger the entrance
 * animation per-letter.
 */
async function buildWordmark(font, mode) {
  const fg = mode === 'dark' ? DARK_FG : LIGHT_FG;
  const text = 'DeepEval.';
  const fontSize = 96;

  // Lay out each glyph individually so we can animate them separately.
  // We use opentype's per-glyph advance widths so kerning is preserved.
  const glyphs = [];
  let cursor = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const p = font.getPath(ch, cursor, 0, fontSize);
    glyphs.push({ ch, d: p.toPathData(2), x: cursor });
    const advance =
      (font.charToGlyph(ch).advanceWidth / font.unitsPerEm) * fontSize;
    if (i < text.length - 1) {
      const kern = font.getKerningValue(
        font.charToGlyph(ch),
        font.charToGlyph(text[i + 1]),
      );
      cursor += advance + (kern / font.unitsPerEm) * fontSize;
    } else {
      cursor += advance;
    }
  }

  const wholePath = font.getPath(text, 0, 0, fontSize);
  const bbox = wholePath.getBoundingBox();
  const textWidth = bbox.x2 - bbox.x1;
  const textHeight = bbox.y2 - bbox.y1;

  const padX = 4;
  const padY = 12;
  const totalW = textWidth + padX * 2;
  const totalH = textHeight + padY * 2;
  // Translate so the text's visual bbox sits inside the padded canvas:
  // text glyphs are positioned with baseline at y=0, so we translate by
  // (padY - bbox.y1) on Y to drop the top of the bbox at padY.
  const tx = padX - bbox.x1;
  const ty = padY - bbox.y1;

  let letterGs = '';
  glyphs.forEach((g, i) => {
    const delay = (i * 45 + 100).toFixed(0);
    const fill = g.ch === '.' ? ACCENT : fg;
    letterGs += `<g class="hero-letter" style="animation-delay:${delay}ms" fill="${fill}"><path d="${g.d}" transform="translate(${tx.toFixed(2)},${ty.toFixed(2)})"/></g>`;
  });

  const css = `
    .hero-letter { opacity: 0; transform: translateY(8px); animation: heroLetterIn 500ms cubic-bezier(.2,.8,.2,1) forwards; }
    @keyframes heroLetterIn { to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) {
      .hero-letter { opacity: 1 !important; transform: none !important; animation: none !important; }
    }
  `;

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${Math.ceil(totalW)} ${Math.ceil(totalH)}" width="${Math.ceil(totalW)}" height="${Math.ceil(totalH)}" role="img" aria-label="DeepEval.">
  <style>${css}</style>
  ${letterGs}
</svg>
`;
}

async function main() {
  await ensureFont();
  await fs.mkdir(OUT_DIR, { recursive: true });

  const font = opentype.parse((await fs.readFile(FONT_PATH)).buffer);

  for (const mode of ['light', 'dark']) {
    const wordmark = await buildWordmark(font, mode);
    await fs.writeFile(path.join(OUT_DIR, `wordmark-${mode}.svg`), wordmark);
    console.log(`wrote wordmark-${mode}.svg`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
