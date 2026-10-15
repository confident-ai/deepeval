export const niceTicks = (min: number, max: number, step?: number): number[] => {
  if (!step) {
    const raw = (max - min) / 4;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const norm = raw / magnitude;
    step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * magnitude;
  }
  const ticks: number[] = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) {
    ticks.push(+v.toFixed(10));
  }
  return ticks;
};
