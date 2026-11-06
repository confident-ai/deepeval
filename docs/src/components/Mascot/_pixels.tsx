import type { Sprite } from "./_sprites";

export const pixelRuns = (sprite: Sprite, row = 0, shift = 0) => {
  const runs: Record<string, string> = {};
  sprite.forEach((line, y) => {
    let x = 0;
    while (x < line.length) {
      const key = line[x];
      let end = x + 1;
      while (line[end] === key) end++;
      if (key !== ".") {
        const w = end - x;
        runs[key] = `${runs[key] ?? ""}M${x + shift} ${y + row}h${w}v1h-${w}z`;
      }
      x = end;
    }
  });
  return runs;
};

export const pixelPaths = (
  sprite: Sprite,
  palette: Record<string, string>,
  row = 0,
  shift = 0
) =>
  Object.entries(pixelRuns(sprite, row, shift)).map(([key, d]) => (
    <path key={key} d={d} fill={palette[key]} />
  ));
