// Rounds a rough step up to 1, 2 or 5 times a power of ten.
export function niceStep(rough: number): number {
  const power = 10 ** Math.floor(Math.log10(rough));
  const unit = rough / power;
  if (unit <= 1) return power;
  if (unit <= 2) return 2 * power;
  if (unit <= 5) return 5 * power;
  return 10 * power;
}

// Evenly spaced ticks from zero to a round number at or above `max`.
export function zeroBasedTicks(max: number, target = 4): number[] {
  const step = niceStep(max / target || 1);
  const top = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = 0; value <= top + step / 2; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}
