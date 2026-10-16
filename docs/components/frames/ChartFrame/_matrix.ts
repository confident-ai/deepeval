import type { MatrixData } from "@/lib/charts/schema";
import { labelWidth, wrapFullLabel } from "./_labels";
import { type Canvas, emptyScene, type Scene } from "./_scene";

export type MatrixType = "heatmap" | "confusion-matrix";

// Rows are y categories, columns x categories. In a confusion matrix rows are
// the truth and columns the prediction, and the first category is "positive".

const AXIS_TITLE = 24;
const LINE = 15;
// Below this a column can't hold its number legibly, so the canvas widens and scrolls.
const MIN_CELL = 52;

const formatter = (data: MatrixData) => (v: number) =>
  `${v.toLocaleString("en-US", { minimumFractionDigits: data.value.decimals, maximumFractionDigits: data.value.decimals })}${data.value.unit}`;

const percent = (share: number) => `${share > 0 && share < 0.01 ? "<1" : Math.round(share * 100)}%`;

/** Names of a 2×2 confusion matrix's cells, by [actual][predicted]. */
const CELL_NAMES = [
  ["true positive", "false negative"],
  ["false positive", "true negative"],
];

export const confusionStats = (values: number[][]) => {
  const total = values.flat().reduce((sum, v) => sum + v, 0);
  const agree = values.reduce((sum, row, i) => sum + (row[i] ?? 0), 0);
  const stats: { label: string; value: string }[] = [{ label: "accuracy", value: percent(total ? agree / total : 0) }];
  if (values.length === 2) {
    const [[tp, fn], [fp]] = values;
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    stats.push(
      { label: "precision", value: precision.toFixed(2) },
      { label: "recall", value: recall.toFixed(2) },
      { label: "F1", value: f1.toFixed(2) }
    );
  }
  return stats;
};

/** Everything the canvas size and the layout share. */
const plan = (type: MatrixType, data: MatrixData, width: number) => {
  const cols = data.x.categories.length;
  const rows = data.y.categories.length;
  const titleLeft = data.y.label ? AXIS_TITLE : 0;
  // Row labels get at least their longest word, so they wrap between words, never inside one.
  const longestRowWord = Math.max(...data.y.categories.flatMap(c => c.split(/\s+/).map(w => labelWidth(w))));
  const rowLabelWidth = Math.min(
    Math.max(width * 0.28, longestRowWord + 14),
    width * 0.4,
    Math.max(...data.y.categories.map(c => labelWidth(c))) + 12
  );
  const rowLines = data.y.categories.map(c => wrapFullLabel(c, rowLabelWidth - 10));
  const left = titleLeft + rowLabelWidth;
  // Column labels wrap at spaces; when a single word can't fit its column, they angle instead.
  const roughCell = Math.max(MIN_CELL, (width - left - 4) / cols);
  const longestWord = Math.max(...data.x.categories.flatMap(c => c.split(/\s+/).map(w => labelWidth(w))));
  const rotate = longestWord > roughCell - 8;
  // Angled labels rise to the right, so the last one needs room past the grid.
  const lastLabel = labelWidth(data.x.categories[cols - 1]) * Math.SQRT1_2 + 8;
  const right = rotate ? Math.max(4, lastLabel - roughCell / 2) : 4;
  const layoutWidth = Math.max(width, left + cols * MIN_CELL + right);
  const cellW = (layoutWidth - left - right) / cols;
  const colLines = data.x.categories.map(c => (rotate ? [c] : wrapFullLabel(c, cellW - 8)));
  const labelsHeight = rotate
    ? Math.max(...data.x.categories.map(c => labelWidth(c))) * Math.SQRT1_2 + 6
    : Math.max(...colLines.map(l => l.length)) * LINE;
  const top = (data.x.label ? AXIS_TITLE : 0) + labelsHeight + 8;
  const confusion = type === "confusion-matrix";
  const big = confusion && rows <= 3;
  const labelsTall = Math.max(...rowLines.map(l => l.length)) * LINE + 8;
  const cellH = Math.max(labelsTall, big ? Math.min(Math.max(cellW * 0.55, 72), 104) : Math.min(Math.max(cellW * 0.45, 36), 56));
  const bottom = confusion ? 34 : 6;
  return { cols, rows, titleLeft, rowLabelWidth, rowLines, left, layoutWidth, cellW, cellH, colLines, rotate, top, bottom, big };
};

export const matrixCanvas = (type: MatrixType, data: MatrixData, width: number): Canvas => {
  const p = plan(type, data, width);
  return {
    width: p.layoutWidth,
    height: p.top + p.rows * p.cellH + p.bottom,
    layoutWidth: width,
    ...(p.layoutWidth > width ? { minWidth: p.layoutWidth } : {}),
  };
};

export const layoutMatrix = (type: MatrixType, data: MatrixData, canvas: Canvas): Scene => {
  const s = emptyScene();
  const p = plan(type, data, canvas.layoutWidth ?? canvas.width);
  const fmt = formatter(data);
  const gridBottom = p.top + p.rows * p.cellH;
  const gridRight = p.left + p.cols * p.cellW;

  if (data.x.label) s.axisLabels.push({ x: p.left + (gridRight - p.left) / 2, y: 14, text: data.x.label });
  if (data.y.label) s.axisLabels.push({ x: 12, y: p.top + (gridBottom - p.top) / 2, text: data.y.label, rotate: true });

  data.x.categories.forEach((_, c) => {
    const lines = p.colLines[c];
    s.categoryLabels.push(
      p.rotate
        ? { x: p.left + p.cellW * (c + 0.5) + 4, y: p.top - 8, lines, fullText: data.x.categories[c], anchor: "start", rotate: -45 }
        : { x: p.left + p.cellW * (c + 0.5), y: p.top - 8 - (lines.length - 1) * LINE, lines, fullText: data.x.categories[c], anchor: "middle" }
    );
  });
  data.y.categories.forEach((category, r) => {
    const lines = p.rowLines[r];
    s.categoryLabels.push({
      x: p.left - 10,
      y: p.top + p.cellH * (r + 0.5) + 4.5 - ((lines.length - 1) * LINE) / 2,
      lines,
      fullText: category,
      anchor: "end",
    });
  });

  const flat = data.values.flat();
  const lo = Math.min(...flat);
  const hi = Math.max(...flat);
  const total = flat.reduce((sum, v) => sum + v, 0);
  // Tints stay light enough for the numbers on them to read in both themes.
  const tint = (v: number, max: number, min = 0) => (max > min ? 0.1 + 0.5 * ((v - min) / (max - min)) : 0.35);

  data.values.forEach((row, r) =>
    row.forEach((v, c) => {
      const x = p.left + c * p.cellW;
      const y = p.top + r * p.cellH;
      const i = r * p.cols + c;
      // Cells fade in one by one; their numbers follow row by row.
      const li = r;
      const cx = x + p.cellW / 2;
      const cy = y + p.cellH / 2;
      if (type === "heatmap") {
        s.cells.push({ x, y, w: p.cellW, h: p.cellH, color: data.color, opacity: tint(v, hi, lo), i });
        s.labels.push({ x: cx, y: cy + 4.5, text: fmt(v), color: "ink", anchor: "middle", i: li, plain: true });
        return;
      }
      const agree = r === c;
      s.cells.push({ x, y, w: p.cellW, h: p.cellH, color: agree ? "teal" : "pink", opacity: tint(v, hi), i });
      const name = p.rows === 2 && p.big ? CELL_NAMES[r][c] : null;
      const share = percent(total ? v / total : 0);
      if (p.big) {
        if (name) s.labels.push({ x: cx, y: cy - 16, text: name, color: "ink", anchor: "middle", i: li, plain: true, small: true });
        s.labels.push({ x: cx, y: cy + (name ? 9 : 4), text: fmt(v), color: "ink", anchor: "middle", i: li, plain: true, large: true });
        s.labels.push({ x: cx, y: cy + (name ? 27 : 22), text: share, color: "ink", anchor: "middle", i: li, plain: true, small: true });
      } else {
        s.labels.push({ x: cx, y: cy + 4.5, text: `${fmt(v)} · ${share}`, color: "ink", anchor: "middle", i: li, plain: true });
      }
    })
  );

  if (type === "confusion-matrix") {
    const text = confusionStats(data.values)
      .map(st => `${st.label} ${st.value}`)
      .join("   ·   ");
    s.labels.push({ x: p.left + (gridRight - p.left) / 2, y: gridBottom + 24, text, color: "ink", anchor: "middle", i: p.rows, plain: true, small: true });
  }
  return s;
};
