import type { Canvas, Scene } from "./_scene";

// Conservative estimate for the inherited proportional font. Leave breathing
// room instead of shrinking type to fit an arbitrary number of labels.
export const labelWidth = (text: string, size = 13) => Array.from(text).length * size * 0.72;

export const fitLabel = (text: string, width: number, size = 13): string => {
  if (labelWidth(text, size) <= width) return text;
  const count = Math.max(0, Math.floor(width / (size * 0.72)) - 1);
  return `${Array.from(text).slice(0, count).join("")}…`;
};

export const wrapLabel = (text: string, width: number): string[] => {
  const words = text.trim().split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (let i = 0; i < words.length; i++) {
    const next = line ? `${line} ${words[i]}` : words[i];
    if (labelWidth(next) <= width) {
      line = next;
    } else if (line && lines.length === 0) {
      lines.push(line);
      line = words[i];
    } else {
      const rest = [next, ...words.slice(i + 1)].join(" ");
      lines.push(fitLabel(rest, width));
      return lines;
    }
  }
  if (line) lines.push(fitLabel(line, width));
  return lines;
};

/** Keep labels inside the canvas and omit ticks whose text would collide. */
export const fitTicks = (ticks: Scene["ticks"], canvas: Canvas, inset = 4): Scene["ticks"] => {
  const kept: Scene["ticks"] = [];
  const boxes: { left: number; right: number; top: number; bottom: number }[] = [];
  for (const tick of ticks) {
    const available = tick.anchor === "end" ? tick.x - inset : tick.anchor === "start" ? canvas.width - tick.x - 4 : canvas.width - 8;
    let compact = tick.text;
    if (labelWidth(compact, 11) > available) {
      const match = compact.match(/^(-?\d+(?:\.\d+)?)(.*)$/);
      if (match) {
        const value = Number(match[1]);
        const number = Math.abs(value) >= 10000
          ? new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value)
          : value !== 0 && Math.abs(value) < 0.001 ? value.toExponential(1) : match[1];
        compact = number + match[2];
      }
    }
    const text = fitLabel(compact, Math.max(8, available), 11);
    const width = labelWidth(text, 11);
    const offset = tick.anchor === "end" ? width : tick.anchor === "middle" ? width / 2 : 0;
    const left = Math.max(inset, Math.min(canvas.width - width - 4, tick.x - offset));
    const box = { left, right: left + width, top: tick.y - 11, bottom: tick.y + 3 };
    if (boxes.some(b => box.left < b.right + 8 && box.right + 8 > b.left && box.top < b.bottom + 4 && box.bottom + 4 > b.top)) continue;
    boxes.push(box);
    kept.push({ ...tick, x: left + offset, text, fullText: tick.text });
  }
  return kept;
};

/** Wrap without losing text; break long unspaced names as a last resort. */
export const wrapFullLabel = (text: string, width: number): string[] => {
  const max = Math.max(1, Math.floor(width / (13 * 0.72)));
  const lines: string[] = [];
  let line = "";
  for (const word of text.trim().split(/\s+/)) {
    if (line && Array.from(`${line} ${word}`).length <= max) {
      line += ` ${word}`;
      continue;
    }
    if (line) lines.push(line);
    const chars = Array.from(word);
    while (chars.length > max) lines.push(chars.splice(0, max).join(""));
    line = chars.join("");
  }
  if (line) lines.push(line);
  return lines;
};
